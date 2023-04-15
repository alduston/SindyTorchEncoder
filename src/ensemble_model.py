import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from sindy_utils import z_derivative, z_derivative_order2,\
    get_initialized_weights, sindy_library_torch, sindy_library_torch_order2
import warnings
from copy import copy, deepcopy
import numpy as np
import matplotlib.pyplot as plt
import random
#import tensorflow as tf
warnings.filterwarnings("ignore")


def expand_tensor(tensor, r):
    l,L = tensor.shape
    expanded_tensor = torch.zeros(l*r, L)
    for i in range(l):
        for k in range(r):
            row_idx = i*r +k
            expanded_tensor[row_idx, :] += tensor[i,:]
    return expanded_tensor

def plot_mask(mask, savedir = 'mask.png'):
    transpose = False
    if mask.shape[0] > mask.shape[1]:
        transpose = True
        mask = mask.T
    L = max(mask.shape)
    l = min(mask.shape)
    r = int(L / l)
    print_mask = expand_tensor(mask,r)
    if transpose:
        print_mask = print_mask.T
    plt.imshow(binarize(print_mask))
    plt.savefig(savedir)
    clear_plt()
    return True


def clear_plt():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    return True


def binarize(tensor):
    binary_tensor = (tensor != 0).float() * 1
    binary_tensor[-1,-1]=0
    return binary_tensor



class SindyNetEnsemble(nn.Module):
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
        self.submodels = self.init_submodels()

        decoder, decoder_layers = self.Decoder(self.params)
        self.decoder = decoder
        self.decoder_layers = decoder_layers

        self.iter_count = torch.tensor(0, device = device)
        self.epoch = torch.tensor(0, device = device)

        self.sindy_coeffs = torch.nn.Parameter(self.init_sindy_coefficients())
        self.coefficient_mask = torch.tensor(params['coefficient_mask'], dtype = torch.float32, device = self.device)

        self.num_active_coeffs = torch.sum(copy(self.coefficient_mask)).cpu().detach().numpy()
        self.exp_label = params['exp_label']
        self.true_coeffs = torch.tensor(params['true_coeffs'], dtype=torch.float32, device=self.device)
        self.torch_params =  self.get_params()


    def get_params(self):
        params = list(self.parameters())
        torch_params = []
        for i,model in enumerate(self.submodels):
            params += model['encoder'].parameters()
            #params += model['decoder'].parameters()
        for i,tensor in enumerate(params):
            self.register_parameter(name=f'param{i}', param = tensor)
        return params


    def init_submodel(self, idx):
        encoder, encoder_layers = self.Encoder(self.params)
        #decoder, decoder_layers = self.Decoder(self.params)

        encoder.sindy_coeffs =  nn.Parameter(self.init_sindy_coefficients(), requires_grad = True)
        submodel = {'encoder' : encoder, 'encoder_layers': encoder_layers,
                    'sindy_coeffs': encoder.sindy_coeffs}
                    #'decoder': decoder, 'decoder_layers': decoder_layers,
        return submodel


    def init_submodels(self):
        submodels = []
        for bag_idx in range(self.params['nbags']):
            submodel = self.init_submodel(bag_idx)
            submodels.append(submodel)
        return submodels


    def Encoder(self, params):
        activation_function = self.get_activation_f(params)
        input_dim = params['input_dim']
        latent_dim = params['latent_dim']
        widths = params['widths']
        #widths = [int(width * np.random.uniform(.7,1.5,1)) for width in widths]

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


    def encoder_weights(self, bag_idx):
        encoder_weights = []
        encoder_biases = []
        encoder_layers = self.submodels[bag_idx]['encoder_layers']
        for layer in encoder_layers:
            try:
                encoder_weights.append(layer.weight)
                encoder_biases.append(layer.bias)
            except AttributeError:
                pass
        return encoder_weights, encoder_biases


    def sub_dz_loss(self, x, dx, sub_idx):
        M = x.shape[0] * x.shape[1]
        sub_model = self.submodels[sub_idx]
        encoder = sub_model['encoder']
        z = encoder(x)
        encoder_weights, encoder_biases = self.encoder_weights(sub_idx)
        activation = self.params['activation']
        dz = z_derivative(x, dx, encoder_weights, encoder_biases, activation)
        Theta = self.Theta(z, x, dx)
        coeff_m_p = sub_model['sindy_coeffs']
        sindy_predict = torch.matmul(Theta, self.coefficient_mask * coeff_m_p)
        sub_loss = torch.mean((dz - sindy_predict.T)**2)/M
        return float(sub_loss.detach())


    def agr_dz_loss(self, x, dx):
        activation = self.params['activation']
        M = x.shape[0] * x.shape[1]
        dzs = []
        for sub_idx in range(self.params['nbags']):
            encoder_weights, encoder_biases = self.encoder_weights(sub_idx)
            dz_sub = z_derivative(x, dx, encoder_weights, encoder_biases, activation)
            dzs.append(dz_sub)
        dz = self.aggregate(torch.stack(dzs))
        z = self.agr_forward(x)
        Theta = self.Theta(z, x, dx)
        agr_coeff =  self.aggregate(self.sub_model_coeffs())
        sindy_predict = torch.matmul(Theta, self.coefficient_mask * agr_coeff)
        sub_loss = torch.mean((dz - sindy_predict.T)**2)/M
        return float(sub_loss.detach())


    def dz(self, x, dx):
        activation = self.params['activation']
        sub_models = self.submodels
        dzs = []

        for bag_idx, sub_model in enumerate(sub_models):
            encoder_weights, encoder_biases = self.encoder_weights(bag_idx)
            mask = copy(sub_model['output_mask'].T)

            if self.params['eval']:
                mask = torch.ones(mask.shape, device = self.device)
            if not bag_idx:
                dz  = z_derivative(x, dx , encoder_weights, encoder_biases, activation)
                output_shape = dz.shape
                dz *= self.reshape_mask(mask, output_shape, first = False)
                dzs.append(dz)
            else:
                dz += self.reshape_mask(mask, output_shape) * z_derivative(x , dx , encoder_weights,
                                                                      encoder_biases, activation)
                dzs.append(dz)

        dzs = torch.stack(dzs)
        if self.params['eval']:
            return self.aggregate(dzs)
        return torch.sum(dzs, 0)


    def ddz(self, x , dx, ddx, bag_idx):
        activation = self.params['activation']
        encoder_weights, encoder_biases = self.encoder_weights(bag_idx)
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
            Theta = sindy_library_torch(z, latent_dim, poly_order, include_sine, device = self.device)
        if model_order == 2:
            dz = self.dz(x, dx)
            Theta = sindy_library_torch_order2(z, dz, latent_dim, poly_order, include_sine, device = self.device)
        return Theta


    def masked_predict(self, Theta):
        masks = [copy(submodel['output_mask']) for submodel in self.submodels]
        coeffs = [submodel['sindy_coeffs'] for submodel in self.submodels]
        coeff_mask = self.coefficient_mask
        for idx,coeff_m in enumerate(coeffs):
            mask = masks[idx]
            sub_predict = torch.matmul(Theta, coeff_mask * coeff_m)
            if not idx:
                output_shape = sub_predict.shape
                mask = self.reshape_mask(mask, output_shape, first = False)
                sindy_predict = mask * sub_predict
            else:
                mask = self.reshape_mask(mask, output_shape, first = False)
                sindy_predict += mask * sub_predict
        return sindy_predict


    def sindy_predict(self, z, x = None, dx = None):
        Theta = self.Theta(z, x, dx)
        if self.params['eval']:
            sindy_coefficients = self.sindy_coeffs
            return torch.matmul(Theta, self.coefficient_mask * sindy_coefficients)
        else:
            return self.masked_predict(Theta)


    def calc_coefficient_mask(self):
        sindy_coefficients = self.sindy_coeffs
        coefficient_mask = self.coefficient_mask * torch.tensor(
            torch.abs(sindy_coefficients) >= self.params['coefficient_threshold'],
            device=self.device)
        self.coefficient_mask = coefficient_mask
        return coefficient_mask


    def reshape_mask(self, mask, output_shape, first = True):
        if first:
            return mask[:output_shape[0], :output_shape[-1]]
        else:
            return mask[:output_shape[0], :output_shape[-1]]


    def active_coeffs(self):
        sindy_coefficients = self.sindy_coeffs
        coefficient_mask = self.coefficient_mask
        return sindy_coefficients * coefficient_mask


    def sub_dx(self, bag_idx, x, dx):
        sub_model = self.submodels[bag_idx]
        z_p = sub_model['encoder'](x)
        Theta = self.Theta(z_p, x, dx)
        coeff_m_p = sub_model['sindy_coeffs']
        sindy_predict = torch.matmul(Theta, self.coefficient_mask * coeff_m_p)
        decoder_weights, decoder_biases = self.decoder_weights()
        activation = self.params['activation']
        sub_dx = z_derivative(z_p, sindy_predict, decoder_weights, decoder_biases, activation=activation)
        return sub_dx


    def agr_dx(self, x, dx):
        z_aggregate = self.agr_forward(x)
        Theta = self.Theta(z_aggregate, x, dx)
        agr_coeff =  self.aggregate(self.sub_model_coeffs())
        sindy_predict = torch.matmul(Theta, self.coefficient_mask * agr_coeff)
        decoder_weights, decoder_biases = self.decoder_weights()
        activation = self.params['activation']
        agr_dx = z_derivative(z_aggregate, sindy_predict, decoder_weights, decoder_biases, activation=activation)
        return agr_dx


    def dx_decode(self,z, x, dx = None):
        sindy_predict = self.sindy_predict(z, x, dx)
        decoder_weights, decoder_biases = self.decoder_weights()
        activation = self.params['activation']
        sub_models = self.submodels
        rescale = 1

        dx_decodes = []
        for bag_idx, sub_model in enumerate(sub_models):
            mask = copy(sub_model['output_mask']).T

            if self.params['eval']:
                mask = torch.ones(mask.shape, device=self.device)
                #rescale = (1 / len(sub_models))
            if not bag_idx:
                dx_decode = z_derivative(z, sindy_predict, decoder_weights,
                                                decoder_biases, activation=activation)
                output_shape = dx_decode.shape
                mask = self.reshape_mask(mask, output_shape)
                dx_decode *= mask
                dx_decodes.append(dx_decode)
            else:
                mask = self.reshape_mask(mask, output_shape)
                dx_decodes.append(mask * z_derivative(z, sindy_predict, decoder_weights, decoder_biases, activation=activation))
        dx_decodes = torch.stack(dx_decodes)
        if self.params['eval']:
            dx_decode = self.aggregate(dx_decodes)
        else:
            for bag_idx, sub_model in enumerate(sub_models):
                dx_decode = torch.sum(dx_decodes,0)
                M = dx.shape[0] * dx.shape[1]

                dx_p = self.sub_dx(bag_idx, x, dx)
                error = copy(torch.sum((((dx_p.T - dx)**2))))/M
                self.params['dx_errors'][bag_idx] += float(error.detach())
                if bag_idx == 0:
                    dx_agr = self.agr_dx(x, dx)
                    error =  copy(torch.sum((((dx_agr.T - dx)**2))))/M
                    self.params['dx_errors'][-1] += float(error.detach())
        return dx_decode


    def ddx_decode(self,z, x, dx):
        sindy_predict = self.sindy_predict(z, x, dx)
        decoder_weights, decoder_biases = self.decoder_weights()
        activation = self.params['activation']
        dz = self.dz(x,dx)
        dx_decode, ddx_decode = z_derivative_order2(z, dz, sindy_predict, decoder_weights, decoder_biases, activation=activation)
        return dx_decode, ddx_decode


    def sub_model_coeffs(self):
        return torch.stack([submodel['sindy_coeffs'] for submodel in self.submodels])


    def aggregate(self, tensors, agr_key = 'mean'):
        if agr_key == 'median':
            return torch.median(tensors,0)[0]
        if agr_key == 'mean':
            return torch.mean(tensors,0)


    def agr_forward(self, x):
        submodels = self.submodels
        zs = []
        for i,submodel in enumerate(submodels):
            encoder = submodel['encoder']
            if not i:
                z = encoder(x)
                zs.append(z)
            else:
                z = encoder(x)
                zs.append(z)
        zs = torch.stack(zs)
        return self.aggregate(zs)


    def agr_decode(self, z):
        submodels = self.submodels
        xs = []
        for i,submodel in enumerate(submodels):
            decoder = submodel['decoder']
            x = decoder(z)
            xs.append(x)
        return self.aggregate(torch.stack(xs))


    def masked_forward(self, x):
        submodels = self.submodels
        for i, submodel in enumerate(submodels):
            encoder = submodel['encoder']
            mask = copy(submodel['output_mask'])
            if not i:
                z_p = encoder(x)
                output_shape = z_p.shape
                z_mask = self.reshape_mask(mask, output_shape, first = False)
                z = z_mask * z_p
            else:
                z_mask = self.reshape_mask(mask, output_shape, first = False)
                z += z_mask * encoder(x)
        return z


    def masked_decode(self, z):
        submodels = self.submodels
        for i, submodel in enumerate(submodels):
            decoder = submodel['decoder']
            mask = copy(submodel['output_mask'])
            if not i:
                x_p = decoder(z)
                output_shape = x_p.shape
                x_mask = self.reshape_mask(mask, output_shape, first = False)
                x = x_mask * x_p
            else:
                x_mask = self.reshape_mask(mask, output_shape, first = False)
                x += x_mask * decoder(z)
        return x


    def forward(self, x):
        if self.params['eval']:
            z = self.agr_forward(x)
        else:
            z = self.masked_forward(x)
        x_decode = self.decoder(z)
        return x_decode, z


    def decoder_loss(self, x, x_pred):
        criterion = nn.MSELoss()
        return self.params['loss_weight_decoder'] *  self.params['bagn_factor'] * criterion(x, x_pred)


    def sindy_reg_loss(self, alt = False):
        if self.params['eval']:
            sub_coeffs = self.sindy_coeffs
        else:
            sub_coeffs = self.sub_model_coeffs()
            if alt:
                sub_coeffs = torch.sum(sub_coeffs, dim = 0)
        reg_loss = self.params['loss_weight_sindy_regularization'] * torch.mean(torch.abs(sub_coeffs))
        reg_loss *= (1 / self.params['nbags'])
        return reg_loss


    def sindy_z_loss(self, z, x, dx, ddx = None):
        criterion = nn.MSELoss()
        if self.params['model_order'] == 1:
            dz = self.dz(x, dx)
            dz_predict = torch.transpose(self.sindy_predict(z, x, dx),0,1)
            return self.params['loss_weight_sindy_z'] * criterion(dz, dz_predict)
        else:
            ddz = self.ddz(x, dx, ddx)[1]
            ddz_predict = torch.transpose(self.sindy_predict(z, x, dx),0,1)
            return  self.params['loss_weight_sindy_z'] * criterion(ddz , ddz_predict)


    def sindy_x_loss(self, z, x, dx, ddx = None):
        criterion = nn.MSELoss()
        if self.params['model_order'] == 1:
            dx_decode = torch.transpose(self.dx_decode(z, x, dx),0,1)
            return self.params['loss_weight_sindy_x'] * criterion(dx , dx_decode)
        else:
            dx_decode, ddx_decode = self.ddx_decode(z, x, dx)
            ddx_decode = torch.transpose(ddx_decode,0,1)
            return  self.params['loss_weight_sindy_x'] * criterion(ddx , ddx_decode)


    def Loss(self, x, dx, ddx = None):
        x_decode, z = self.forward(x)
        decoder_loss = self.decoder_loss(x, x_decode)
        sindy_z_loss = self.sindy_z_loss(z, x, dx, ddx)
        sindy_x_loss = self.sindy_x_loss(z, x, dx, ddx)
        reg_loss = self.sindy_reg_loss(alt = False)

        loss_refinement = decoder_loss + sindy_z_loss + sindy_x_loss
        loss = loss_refinement + reg_loss
        losses = {'decoder': decoder_loss, 'sindy_z': sindy_z_loss,
                  'sindy_x': sindy_x_loss, 'reg': reg_loss}
        return loss, loss_refinement, losses


    def loss(self, x, x_decode, z, dx, ddx=None):
        return self.Loss(x, x_decode, z, dx, ddx)[0]


    def auto_Loss(self, x, dx, ddx=None):
        return self.Loss(x, dx, ddx)
