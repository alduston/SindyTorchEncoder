import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from sindy_utils import z_derivative, z_derivative_order2,\
    get_initialized_weights, sindy_library_torch, sindy_library_torch_order2
import warnings
from copy import copy, deepcopy
import tensorflow as tf
warnings.filterwarnings("ignore")


class SindyNet(nn.Module):
    def __init__(self, params, device = None):
        super().__init__()
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        params['device'] = self.device
        self.params = params
        self.activation_f = self.get_activation_f(params)

        encoder, encoder_layers = self.Encoder(self.params)
        self.encoder = encoder
        self.encoder_layers = encoder_layers

        decoder, decoder_layers = self.Decoder(self.params)
        self.decoder = decoder
        self.decoder_layers = decoder_layers

        self.iter_count = torch.tensor(0, device = device)
        self.epoch = torch.tensor(0, device = device)

        self.sindy_coeffs = torch.nn.Parameter(self.init_sindy_coefficients(), requires_grad = True)
        self.coefficient_mask = torch.tensor(params['coefficient_mask'], dtype = torch.float32, device = self.device)

        self.damping_mask = torch.tensor(params['coefficient_mask'], dtype=torch.float32, device=self.device)
        self.activation_mask = torch.tensor(params['coefficient_mask'], dtype=torch.float32, device=self.device)

        self.num_active_coeffs = torch.sum(copy(self.coefficient_mask)).cpu().detach().numpy()
        self.exp_label = params['exp_label']
        self.true_coeffs = torch.tensor(params['true_coeffs'], dtype=torch.float32, device=self.device)

        self.sub_model_coeffs = {}
        self.sub_model_masks = {}
        self.sub_model_losses_dict = {}


    def Encoder(self, params):
        activation_function = self.get_activation_f(params)
        input_dim = params['input_dim']
        latent_dim = params['latent_dim']
        widths = params['widths']

        layers = []
        for output_dim in widths:
            encoder = nn.Linear(input_dim, output_dim)
            nn.init.xavier_uniform(encoder.weight)
            nn.init.constant_(encoder.bias.data, 0)

            input_dim = output_dim
            layers.append(encoder)
            layers.append(activation_function)


        encoder = nn.Linear(input_dim, latent_dim)
        nn.init.xavier_uniform(encoder.weight)
        nn.init.constant_(encoder.bias.data, 0)
        layers.append(encoder)
        Encoder = nn.Sequential(*layers)
        return Encoder, layers


    def Decoder(self, params):
        activation_function = self.get_activation_f(params)
        final_dim = params['input_dim']
        input_dim = params['latent_dim']
        widths = params['widths']

        layers = []
        for output_dim in widths[::-1]:
            decoder = nn.Linear(input_dim, output_dim)
            nn.init.xavier_uniform(decoder.weight)
            nn.init.constant_(decoder.bias.data, 0)

            input_dim = output_dim
            layers.append(decoder)
            layers.append(activation_function)

        decoder = nn.Linear(input_dim, final_dim)
        nn.init.xavier_uniform(decoder.weight)
        nn.init.constant_(decoder.bias.data, 0)
        layers.append(decoder)
        Decoder = nn.Sequential(*layers)
        return Decoder, layers


    def get_activation_f(self, params):
        activation = params['activation']
        if activation == 'relu':
            activation_function = torch.nn.ReLU()
        elif activation == 'elu':
            activation_function = torch.nn.ELU()
        elif activation == 'sigmoid':
            activation_function = torch.nn.Sigmoid()
        return activation_function


    def decoder_weights(self):
        decoder_weights = []
        decoder_biases = []
        for layer in self.decoder_layers:
            try:
                decoder_weights.append(layer.weight)
                decoder_biases.append(layer.bias)
            except AttributeError:
                pass
        return decoder_weights, decoder_biases


    def encoder_weights(self):
        encoder_weights = []
        encoder_biases = []
        for layer in self.encoder_layers:
            try:
                encoder_weights.append(layer.weight)
                encoder_biases.append(layer.bias)
            except AttributeError:
                pass
        return encoder_weights, encoder_biases


    def dz(self, x, dx):
        activation = self.params['activation']
        encoder_weights, encoder_biases = self.encoder_weights()
        return z_derivative(x, dx, encoder_weights, encoder_biases, activation)


    def ddz(self, x , dx, ddx):
        activation = self.params['activation']
        encoder_weights, encoder_biases = self.encoder_weights()
        dz, ddz = z_derivative_order2(x, dx, ddx, encoder_weights, encoder_biases, activation=activation)
        return dz, ddz


    def initializer(self):
        init_param = None
        init = self.params['coefficient_initialization']
        if init == 'xavier':
            intializer = torch.nn.init.xavier_uniform
        elif init == 'specified':
            intializer = torch.nn.init.xavier_uniform
        elif init == 'constant':
            intializer = torch.nn.init.constant_
            init_param = [1]
        elif init == 'normal':
            intializer = torch.nn.init.normal_
        return intializer, init_param


    def init_sindy_coefficients(self):
        library_dim = self.params['library_dim']
        latent_dim = self.params['latent_dim']
        initializer, init_param = self.initializer()
        return get_initialized_weights([library_dim, latent_dim], initializer,
                                       init_param = init_param, device = self.device)


    def Theta(self, z, x = None, dx = None):
        model_order = self.params['model_order']
        poly_order = self.params['poly_order']
        include_sine = self.params['include_sine']
        latent_dim = self.params['latent_dim']
        if model_order == 1:
            #dz = self.dz(x, dx)
            Theta = sindy_library_torch(z, latent_dim, poly_order, include_sine, device = self.device)
        if model_order == 2:
            #dz, ddz = self.ddz(x, dx)
            dz = self.dz(x, dx)
            Theta = sindy_library_torch_order2(z, dz, latent_dim, poly_order, include_sine, device = self.device)
        return Theta


    def dist_mult(self, A_tensor, B_tensor):
        xa,ya = A_tensor.shape
        xb,yb,zb = B_tensor.shape
        A_tensor = A_tensor.reshape(xb, xa//xb, ya)
        output_tensor = torch.einsum('bij,bjk->bik', A_tensor, B_tensor)
        return output_tensor.reshape(xa, zb)

    def masked_predict(self, Theta, coeffs):
        masks = self.params['coeff_masks']
        sindy_predict = torch.zeros(masks[0].shape, device = self.device)
        for idx,coeff_m in enumerate(coeffs):
            mask = masks[idx]
            sub_predict = mask * torch.matmul(Theta, coeff_m)
            sindy_predict += sub_predict
        return sindy_predict


    def sindy_predict(self, z, x = None, dx = None, idx = None,  scramble = False):
        scramble = False
        Theta = self.Theta(z, x, dx)
        epoch = self.epoch
        if idx == None:
            sindy_coefficients = self.sindy_coeffs
        else:
            sindy_coefficients = self.sub_model_coeffs[idx]
        if self.params['sequential_thresholding']:
            if epoch and (epoch % self.params['threshold_frequency'] == 0):
                self.coefficient_mask = self.coefficient_mask * torch.tensor(torch.abs(sindy_coefficients) >= self.params['coefficient_threshold'], device=self.device)
                self.num_active_coeffs = torch.sum(copy(self.coefficient_mask)).cpu().detach().numpy()
        if scramble:
            return self.masked_predict(Theta,  self.sub_model_coeffs)
        return torch.matmul(Theta, self.coefficient_mask * sindy_coefficients)



    def calc_coefficient_mask(self):
        sindy_coefficients = self.sindy_coeffs
        coefficient_mask = self.coefficient_mask * torch.tensor(
            torch.abs(sindy_coefficients) >= self.params['coefficient_threshold'],
            device=self.device)
        self.coefficient_mask = coefficient_mask
        return coefficient_mask


    def active_coeffs(self):
        sindy_coefficients = self.sindy_coeffs
        coefficient_mask = self.coefficient_mask
        return sindy_coefficients * coefficient_mask


    def dx_decode(self,z, x, dx = None, idx = None, scramble = False):
        sindy_predict = self.sindy_predict(z, x, dx, idx, scramble)
        decoder_weights, decoder_biases = self.decoder_weights()
        activation = self.params['activation']
        dx_decode = z_derivative(z, sindy_predict, decoder_weights, decoder_biases, activation=activation)
        return dx_decode


    def ddx_decode(self,z, x, dx, idx = None, scramble = False):
        sindy_predict = self.sindy_predict(z, x, dx, idx, scramble)
        decoder_weights, decoder_biases = self.decoder_weights()
        activation = self.params['activation']
        dz = self.dz(x,dx)
        dx_decode, ddx_decode = z_derivative_order2(z, dz, sindy_predict, decoder_weights, decoder_biases, activation=activation)
        return dx_decode, ddx_decode


    def forward(self, x):
        z = self.encoder(x)
        x_decode = self.decoder(z)
        return x_decode, z


    def decoder_loss(self, x, x_pred):
        criterion = nn.MSELoss()
        return self.params['loss_weight_decoder'] * criterion(x, x_pred)


    def sindy_reg_loss(self, idx = None, penalize_self = False):
        if idx == None:
            sub_coeffs = self.sindy_coeffs
        else:
            if penalize_self:
                sub_coeffs = self.sub_model_coeffs[idx]
            else:
                #sub_coeffs = torch.sum(self.sub_model_coeffs, dim = 0) * (1/self.params['nbags'])
                sub_coeffs = torch.sum(self.sub_model_coeffs, dim=0)
        return self.params['loss_weight_sindy_regularization'] * torch.mean(torch.abs(sub_coeffs))


    def sindy_z_loss(self, z, x, dx, ddx = None, idx = None,  scramble = False):
        criterion = nn.MSELoss()
        if self.params['model_order'] == 1:
            dz = self.dz(x, dx)
            dz_predict = torch.transpose(self.sindy_predict(z, x, dx, idx, scramble),0,1)
            return self.params['loss_weight_sindy_z'] * criterion(dz, dz_predict)
        else:
            ddz = self.ddz(x, dx, ddx)[1]
            ddz_predict = torch.transpose(self.sindy_predict(z, x, dx, idx, scramble),0,1)
            return  self.params['loss_weight_sindy_z'] * criterion(ddz , ddz_predict)


    def sindy_x_loss(self, z, x, dx, ddx = None, idx = None, scramble = False):
        criterion = nn.MSELoss()
        if self.params['model_order'] == 1:
            dx_decode = torch.transpose(self.dx_decode(z, x, dx, idx),0,1)
            return self.params['loss_weight_sindy_x'] * criterion(dx , dx_decode)
        else:
            dx_decode, ddx_decode = self.ddx_decode(z, x, dx, idx)
            ddx_decode = torch.transpose(ddx_decode,0,1)
            return  self.params['loss_weight_sindy_x'] * criterion(ddx , ddx_decode)


    def Loss(self, x, x_decode, z, dx, ddx = None, idx = None,
             penalize_self = False):
        decoder_loss = self.decoder_loss(x, x_decode)
        sindy_z_loss = self.sindy_z_loss(z, x, dx, ddx, idx)
        sindy_x_loss = self.sindy_x_loss(z, x, dx, ddx, idx)
        reg_loss = self.sindy_reg_loss(idx, penalize_self = False)
        if penalize_self:
            self_loss = self.sindy_reg_loss(idx, penalize_self)
            reg_loss += self_loss

        loss_refinement = decoder_loss + sindy_z_loss + sindy_x_loss
        loss = loss_refinement + reg_loss
        losses = {'decoder': decoder_loss, 'sindy_z': sindy_z_loss,
                  'sindy_x': sindy_x_loss, 'reg': reg_loss}
        return loss, loss_refinement, losses


    def loss(self, x, x_decode, z, dx, ddx=None):
        return self.Loss(x, x_decode, z, dx, ddx)[0]


    def auto_Loss(self, x, dx, ddx=None, idx=None, penalize_self = False):
        x_decode, z = self.forward(x)
        return self.Loss(x, x_decode, z, dx, ddx, idx, penalize_self)


    def scramble_Loss(self, x, dx, ddx=None, penalize_self = False, idx = None):
        x_decode, z = self.forward(x)
        decoder_loss = self.decoder_loss(x, x_decode)
        sindy_z_loss = self.sindy_z_loss(z, x, dx, ddx, idx = idx, scramble = True)
        sindy_x_loss = self.sindy_x_loss(z, x, dx, ddx, idx = idx, scramble = True)
        reg_loss = self.sindy_reg_loss(idx = idx, penalize_self = False)
        if penalize_self:
            self_loss = self.sindy_reg_loss(penalize_self)
            reg_loss += self_loss

        loss_refinement = decoder_loss + sindy_z_loss + sindy_x_loss
        loss = loss_refinement + reg_loss
        losses = {'decoder': decoder_loss, 'sindy_z': sindy_z_loss,
                  'sindy_x': sindy_x_loss, 'reg': reg_loss}
        return loss, loss_refinement, losses


    def bag_loss(self, x, dx, ddx=None):
        x_decode, z = self.forward(x)
        sindy_z_loss = self.sindy_z_loss(z, x, dx, ddx) * self.params['bag_loss_weight_sindy_z']
        sindy_x_loss = self.sindy_x_loss(z, x, dx, ddx) * self.params['bag_loss_weight_sindy_x']
        reg_loss = self.sindy_reg_loss() * self.params['bag_loss_weight_sindy_regularization']

        loss_refinement = sindy_z_loss + sindy_x_loss
        loss = loss_refinement + reg_loss

        return loss


'''
TEST: Epoch: 0, Total: 0.035985537, Decoder: 0.034785271, Sindy_x: 0.001199497, Reg: 7.68e-07, Sindy_z: 0.0, Active_coeffs: 60, Coeff: 5
TEST: Epoch: 50, Total: 0.002646168, Decoder: 0.00144647, Sindy_x: 0.001199697, Reg: 1e-09, Sindy_z: 0.0, Active_coeffs: 60, Coeff: 7
TEST: Epoch: 100, Total: 0.001987437, Decoder: 0.000787857, Sindy_x: 0.001199579, Reg: 1e-09, Sindy_z: 0.0, Active_coeffs: 60, Coeff: 7
TEST: Epoch: 150, Total: 0.001857645, Decoder: 0.000658096, Sindy_x: 0.001199548, Reg: 1e-09, Sindy_z: 0.0, Active_coeffs: 50, Coeff: 7
'''