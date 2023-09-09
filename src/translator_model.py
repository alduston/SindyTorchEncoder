import torch
import torch.nn as nn
from sindy_utils import z_derivative, get_initialized_weights, sindy_library_torch, residual_z_derivative
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R


def format(n, n_digits = 6):
    try:
        n = float(n)
        if n > 1e-4:
            return round(n,n_digits)
        a = '%E' % n
        str =  a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]
        scale = str[-5:]
        digits = str[:-5]
        return digits[:min(len(digits),n_digits)] + scale
    except IndexError:
        return float(n)


def dict_mean(dicts):
    mean_dict = {key: torch.mean(torch.stack([sub_dict[key] for sub_dict in dicts])) for key in dicts[0].keys()}
    return mean_dict


def expand_tensor(tensor, r):
    l,L = tensor.shape
    expanded_tensor = torch.zeros(l*r, L)
    for i in range(l):
        for k in range(r):
            row_idx = i*r +k
            expanded_tensor[row_idx, :] += tensor[i,:]
    return expanded_tensor


def clear_plt():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    return True

def double(tensor):
    return torch.concat([tensor,tensor], dim = 1)


def binarize(tensor):
    binary_tensor = (tensor != 0).float() * 1
    return binary_tensor


def avg_criterion(vec, zero_threshold = .1):
    return torch.abs(torch.mean(vec)) >  zero_threshold


def stability_criterion(vec, zero_threshold = .1, accept_threshold = .6):
    return (sum([abs(val) > zero_threshold for val in vec])/len(vec)) > accept_threshold


def halving_widths(in_widht, out_width):
    widths = [in_widht]
    while widths[-1] > 1:
        widths.append(widths[-1]//2)
    return np.asarray(widths[0:])


def diagnolize_weights(w_list):
    n = len(w_list)
    l,w =  w_list[0].shape
    device = w_list[0].device
    dtype = w_list[0].dtype
    diag_tensor = torch.zeros((n * l, n * w), device = device, dtype=dtype)
    for i in range(n):
        diag_tensor[(i) * l:(i+1) * l, (i) * w:(i+1) * w] += w_list[i]
    return diag_tensor


#(9x18 and 9x250)

class DoublingBlock(nn.Module):
    def __init__(self, n_input, device = 'cpu'):
        super().__init__()
        self.device = device
        doubling_tensor = torch.eye(n_input, device = self.device)
        self.doubling_tensor = torch.concat([doubling_tensor, doubling_tensor], dim = 0)
        self.weight =  self.doubling_tensor
        self.bias = torch.zeros(2 * n_input, device = self.device)

    def forward(self, x):
        return  torch.concat([x,x], dim = 1)


class ResidualBlock(nn.Module):
    def __init__(self,  n_indentity, n_input, n_output, base_activation, device = 'cpu'):
        super().__init__()
        self.device = device

        self.activation_f = base_activation
        self.n = n_indentity
        linear_layer = nn.Linear(n_indentity + n_input, n_output, device=self.device)
        nn.init.xavier_uniform(linear_layer.weight)
        nn.init.constant_(linear_layer.bias.data, 0)
        self.lin_layer = linear_layer

        self.weight =  self.lin_layer.weight
        self.bias = self.lin_layer.bias

    def forward(self, x):
        n = self.n
        activation_f = self.activation_f
        x_linear = x[:, :n]
        return torch.concat([x_linear, activation_f(self.lin_layer(x))], dim=1)


class SindyNetTCompEnsemble(nn.Module):
    def __init__(self, indep_models, device = None):
        super().__init__()
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        self.dtype = indep_models.dtype
        params = indep_models.params
        self.params = params
        self.true_coeffs = indep_models.true_coeffs

        self.params['indep_models'] = indep_models
        self.params['loss_weight_sindy_z'] = 0
        self.params['loss_weight_corr'] = 1e-3
        self.params['indep_models'] = indep_models

        self.activation_f = indep_models.activation_f
        self.criterion_f = indep_models.criterion_f

        self.translators = self.init_translators()
        self.decompressor, self.decompressor_layers = self.Residual_Decompressor(self.params)

        self.params['stacked_decoder'], self.params['stacked_decoder_layers'] = self.Stacked_decoder(self.params)
        self.params['stacked_encoder'], self.params['stacked_encoder_layers'] = self.Stacked_encoder(self.params)

        self.sindy_coeffs = torch.nn.Parameter(self.init_sindy_coefficients(), requires_grad=True)
        self.coefficient_mask = torch.tensor(deepcopy(self.params['coefficient_mask']),dtype=self.dtype, device=self.device)
        self.epoch = 0
        self.refresh_val_dict = True


    def Stacked_encoder(self, params):
        indep_models = self.params['indep_models']
        n_encoders = params['n_encoders']
        weights = []
        biases = []

        for j in range(len(indep_models.encoder_weights(0)[0])):
            w_list = [indep_models.encoder_weights(i)[0][j] for i in range(n_encoders)]
            w_stack = diagnolize_weights(w_list)
            weights.append(w_stack)

            b_stack = torch.concat([indep_models.encoder_weights(i)[1][j] for i in range(n_encoders)], dim=0)
            biases.append(b_stack)

        activation_function = self.activation_f
        input_dim = params['input_dim'] * n_encoders
        latent_dim = params['latent_dim'] * n_encoders
        widths = np.asarray(params['widths']) * n_encoders

        layers = []

        for i, output_dim in enumerate(widths):
            layer = nn.Linear(input_dim, output_dim, device=self.device)
            layer.weight.data = weights[i]
            layer.bias.data = biases[i]

            input_dim = output_dim
            layers.append(layer)
            layers.append(activation_function)


        final_layer = nn.Linear(input_dim, latent_dim, device=self.device)
        final_layer.weight.data = weights[-1]
        final_layer.bias.data = biases[-1]
        layers.append(final_layer)

        Stacked_encoder = nn.Sequential(*layers)
        layer_shapes = []
        for layer in layers:
            try:
                layer_shapes.append(layer.weight.shape)
            except AttributeError:
                pass
        return Stacked_encoder, layers


    def Stacked_decoder(self, params):
        indep_models = self.params['indep_models']
        n_decoders = params['n_decoders']
        weights = []
        biases = []

        for j in range(len(indep_models.decoder_weights(0)[0])):
            w_list = [indep_models.decoder_weights(i)[0][j] for i in range(n_decoders)]
            w_stack = diagnolize_weights(w_list)
            weights.append(w_stack)

            b_stack = torch.concat([indep_models.decoder_weights(i)[1][j] for i in range(n_decoders)],dim=0)
            biases.append(b_stack)

        activation_function = self.activation_f
        input_dim = params['latent_dim'] * n_decoders
        final_dim =  params['input_dim'] * n_decoders
        widths = np.asarray(params['widths']) * n_decoders

        layers = []

        for i,output_dim in enumerate(widths[::-1]):
            layer = nn.Linear(input_dim, output_dim, device= self.device)
            layer.weight.data = weights[i]
            layer.bias.data = biases[i]

            input_dim = output_dim
            layers.append(layer)
            layers.append(activation_function)

        final_layer = nn.Linear(input_dim, final_dim, device= self.device)
        final_layer.weight.data = weights[-1]
        final_layer.bias.data = biases[-1]
        layers.append(final_layer)
        Stacked_decoder = nn.Sequential(*layers)

        layer_shapes = []
        for layer in layers:
            try:
                layer_shapes.append(layer.weight.shape)
            except AttributeError:
                pass

        return Stacked_decoder, layers


    def init_translators(self):
        translators = []
        for i in range(self.params['n_encoders']):
            translator, translator_layers = self.Residual_Translator(self.params)
            translators.append({'translator': translator, 'encoder_layers': translator_layers})
            for j, tensor in enumerate(translator.parameters()):
                self.register_parameter(name=f'{"encoder"}{i}{j}', param=tensor)
        return translators


    def Residual_Translator(self, params):
        activation_function = self.activation_f
        latent_dim = deepcopy(params['latent_dim'])
        input_dim = params['latent_dim']
        widths = [2 * latent_dim, 2 * latent_dim]
        layers = [DoublingBlock(latent_dim)]

        for output_dim in widths:
            layer = ResidualBlock(latent_dim, input_dim, output_dim, activation_function, self.device)
            input_dim = output_dim
            layers.append(layer)

        final_layer = nn.Linear(latent_dim + input_dim, latent_dim)
        nn.init.xavier_uniform(final_layer.weight)
        nn.init.constant_(final_layer.bias.data, 0)
        layers.append(final_layer)
        Translator = nn.Sequential(*layers)
        return Translator, layers


    def Residual_Decompressor(self, params):
        activation_function = self.activation_f #self.get_activation_f(params)
        input_dim = params['latent_dim']
        init_dim = deepcopy(input_dim)
        final_dim = params['latent_dim'] * params['n_encoders']
        widths = params['n_encoders'] * halving_widths(params['n_encoders'], 1)
        layers = [DoublingBlock(init_dim)]

        for output_dim in widths:
            layer = ResidualBlock(init_dim, input_dim, output_dim, activation_function, self.device)
            input_dim = output_dim
            layers.append(layer)

        final_layer = nn.Linear(init_dim + output_dim, final_dim)
        nn.init.xavier_uniform(final_layer.weight)
        nn.init.constant_(final_layer.bias.data, 0)
        layers.append(final_layer)
        Decompressor = nn.Sequential(*layers)

        return Decompressor, layers


    def num_active_coeffs(self):
        return torch.sum(self.coefficient_mask)


    def get_comp_weights(self, layers):
        weights = []
        biases = []
        for layer in layers:
            weights.append(layer.weight)
            biases.append(layer.bias)
        return weights, biases


    def get_weights(self, layers):
        weights = []
        biases = []
        for layer in layers[::2]:
            weights.append(layer.weight)
            biases.append(layer.bias)
        return weights, biases


    def initializer(self):
        init_param = None
        init = self.params['coefficient_initialization']
        if init == 'xavier':
            intializer = torch.nn.init.xavier_uniform
        elif init == 'binary_xavier':
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

    def Theta(self, z):
        poly_order = self.params['poly_order']
        include_sine = self.params['include_sine']
        latent_dim = self.params['latent_dim']
        Theta = sindy_library_torch(z, latent_dim, poly_order, include_sine, device = self.device)
        return Theta


    def expand(self, x):
        n = self.params['n_encoders']
        x_stack = torch.concat([x for i in range(n)], dim = 1)
        return x_stack


    def collapse(self,x_stack, agr_key = 'mean', double = False):
        N,m = x_stack.shape
        n = self.params['n_encoders']
        if double:
            n = self.params['n_encoders'] ** 2
        x_stack = x_stack.reshape((N,n,m//n))
        if agr_key == 'mean':
            x = torch.mean(x_stack, dim = 1)
        elif agr_key == 'median':
            x = torch.median(x_stack, dim=1)[0]
        return x


    def split(self, x_stack):
        N, m = x_stack.shape
        n = self.params['n_encoders']
        x_stack = x_stack.reshape((N, n, m // n))
        x_split = x_stack.permute(1,0,2)
        return x_split


    def sindy_predict(self, z, coeffs = [], mask = []):
        Theta = self.Theta(z)
        if not len(mask):
            mask = self.coefficient_mask
        if not len(coeffs):
            coeffs = self.sindy_coeffs
        return torch.matmul(Theta, mask * coeffs)


    def sub_forward(self, x, encode_idx):
        encoder = self.params['indep_models'].Encoders[encode_idx]['encoder']
        stacked_decoder = self.params['stacked_decoder']
        translator = self.translators[encode_idx]['translator']

        x_encode = encoder(x)
        x_translate = translator(x_encode)
        x_decomp = self.decompressor(x_translate)
        x_decomp_decode = stacked_decoder(x_decomp)
        return x_translate, x_decomp_decode


    def stack_forward(self,  x):
        x_decomp_decodes = []
        x_translates = []
        for encode_idx in range(len(self.translators)):
            x_translate, x_decomp_decode = self.sub_forward(x, encode_idx)
            x_decomp_decodes.append(x_decomp_decode)
            x_translates.append(x_translate)

        x_decomp_decode_stack = torch.concat(x_decomp_decodes, dim = 1)
        x_translate_stack = torch.concat(x_translates, dim=1)
        return x_decomp_decode_stack, x_translate_stack


    def dx_decode(self, z, dz):
        decoder_weights, decoder_biases = self.get_weights(self.params['stacked_decoder_layers'])
        decompressor_weights, decompressor_biases = self.get_comp_weights(self.decompressor_layers)
        activation = self.params['activation']

        dz_g = residual_z_derivative(z, dz, decompressor_weights, decompressor_biases, activation).T
        z_g = self.decompressor(z)
        dx_decode = z_derivative(z_g, dz_g, decoder_weights, decoder_biases, activation).T
        return dx_decode


    def stacked_dx_loss(self, x_translate_stack, dx_stack_stack):
        dx_preds = []
        for z in self.split(x_translate_stack):
            dz_pred = self.sindy_predict(z)
            dx_preds.append(self.dx_decode(z, dz_pred))

        dx_pred_stack = torch.concat(dx_preds, dim = 1)

        criterion = nn.MSELoss()
        loss = self.params['loss_weight_sindy_x'] * (criterion(dx_stack_stack, dx_pred_stack))
        return loss, dx_pred_stack


    def decode_loss(self, x, x_pred):
        criterion = nn.MSELoss()
        loss = self.params['loss_weight_decoder'] * criterion(x, x_pred)
        return loss


    def reg_loss(self):
        return self.params['loss_weight_sindy_regularization'] * torch.mean(torch.abs(self.sindy_coeffs))


    def corr_sub_loss(self, Z, encode_idx):
        zs = self.split(Z)
        z_comp = zs[encode_idx]
        z_comp_mean = torch.mean(z_comp, dim=0)
        z_comp_centered = z_comp - z_comp_mean
        corr_sum = 0
        c = torch.trace(z_comp_centered.T  @ z_comp_centered)
        for z  in zs:
            z_mean = torch.mean(z, dim = 0)
            z_centered = z - z_mean

            a = torch.trace(z_centered.T @ z_comp_centered)
            b = torch.trace(z_centered.T @ z_centered)
            corr_sum  += (a/((b * c)**(1/2)))**2

        return -self.params['loss_weight_corr'] * corr_sum * (1/len(zs))


    def corr_loss(self, x_translate_stack):
        corr_loss  = 0
        for encode_idx in range(len(self.translators)):
            corr_loss += self.corr_sub_loss(x_translate_stack, encode_idx)
        return corr_loss/len(self.translators)


    def stacked_dx_decode(self, x_encode):
        decoder_weights, decoder_biases = self.get_weights(self.params['stacked_decoder_layers'])
        indep_masks = self.params['indep_models'].coefficient_masks
        indep_coeffs = self.params['indep_models'].Sindy_coeffs

        x_split = self.split(x_encode)
        z_preds = [self.sindy_predict(x, coeff, mask) for (x,coeff, mask) in zip(x_split, indep_coeffs, indep_masks)]
        z_pred_stack = torch.concat(z_preds , dim = 1)

        activation = self.params['activation']
        stacked_dx_decode = z_derivative(x_encode, z_pred_stack , decoder_weights,
                                         decoder_biases, activation=activation).T
        return stacked_dx_decode

    def val_test(self, x, dx, x_translate_stack, x_decomp_decode_stack, dx_pred_stack, agr_key = 'mean'):
        if self.refresh_val_dict:
            self.val_dict = {'E_Decoder': [],  'E_Sindy_x': [], 'E_agr_Decoder': [], 'E_agr_Sindy_x': []}

        criterion = nn.MSELoss()

        agr_x_translate = self.collapse(x_translate_stack, agr_key=agr_key)
        agr_x_decomp_decode = self.params['stacked_decoder'](self.decompressor(agr_x_translate))

        agr_dz_pred = self.sindy_predict(agr_x_translate)
        agr_dx_pred = self.dx_decode(agr_x_translate, agr_dz_pred)

        agr_decomp_loss = float(self.decode_loss(
            self.collapse(x_decomp_decode_stack, agr_key=agr_key, double=True), x).detach().cpu())
        agr_decomp_loss2 = float(self.decode_loss(
            self.collapse(agr_x_decomp_decode, agr_key=agr_key), x).detach().cpu())

        agr_dx_decomp_loss = criterion(self.collapse(
            dx_pred_stack, agr_key=agr_key, double=True), dx).detach().cpu() * self.params['loss_weight_sindy_x']
        agr_dx_decode_loss2 = criterion(self.collapse(
            agr_dx_pred, agr_key=agr_key), dx).detach().cpu() * self.params['loss_weight_sindy_x']

        self.val_dict['E_Decoder'].append(agr_decomp_loss)
        self.val_dict['E_Sindy_x'].append(agr_dx_decomp_loss)
        self.val_dict['E_agr_Decoder'].append(agr_decomp_loss2)
        self.val_dict['E_agr_Sindy_x'].append(agr_dx_decode_loss2)

        self.refresh_val_dict = False
        return True


    def Loss(self, x, dx):
        x_decomp_decode_stack, x_translate_stack = self.stack_forward(x)

        x_stack = self.expand(x)
        x_stack_stack = self.expand(x_stack)

        dx_stack = self.expand(dx)
        dx_stack_stack = self.expand(dx_stack)

        decoder_loss = self.decode_loss(x_decomp_decode_stack, x_stack_stack)
        reg_loss = self.reg_loss()
        sindy_x_loss, dx_pred_stack = self.stacked_dx_loss(x_translate_stack,dx_stack_stack)
        corr_loss = self.corr_loss(x_translate_stack)

        loss = decoder_loss + sindy_x_loss + corr_loss + reg_loss
        loss_dict = {'decoder': decoder_loss, 'sindy_x': sindy_x_loss, 'sindy_z': corr_loss, 'reg': reg_loss}

        if self.params['eval']:
            self.val_test(x, dx, x_translate_stack, x_decomp_decode_stack, dx_pred_stack)
        return loss, loss_dict




