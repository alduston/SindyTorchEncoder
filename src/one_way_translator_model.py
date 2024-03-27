import torch
import torch.nn as nn
from sindy_utils import z_derivative, get_initialized_weights, sindy_library_torch, residual_z_derivative
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from datetime import datetime as dt
import random


def dict_mean(dicts):
    mean_dict = {key: torch.mean(torch.stack([sub_dict[key] for sub_dict in dicts])) for key in dicts[0].keys()}
    return mean_dict


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
        self.criterion_f = self.criterion__function

        self.translators = self.init_translators()
        self.Decoder, self.Decoder_layers = self.init_decoder(self.params)

        #self.detranslators = self.init_detranslators()

        #self.params['stacked_decoder'], self.params['stacked_decoder_layers'] = self.Stacked_decoder(self.params)
        self.params['stacked_encoder'], self.params['stacked_encoder_layers'] = self.Stacked_encoder(self.params)
        self.sindy_coeffs = self.init_sindy_coeff_stack()

        self.coefficient_mask = torch.tensor(deepcopy(self.params['coefficient_mask']),dtype=self.dtype, device=self.device)
        self.epoch = 0
        self.refresh_val_dict = True
        self.stage = 2
        self.exp_label = params['exp_label']


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


    def init_decoder(self, params):
        activation_function = self.activation_f
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


    def init_translators(self):
        translators = []
        for i in range(self.params['n_encoders']):
            translator, translator_layers = self.Residual_Translator(self.params)
            translators.append({'translator': translator, 'translator_layers': translator_layers})
            for j, tensor in enumerate(translator.parameters()):
                self.register_parameter(name=f'{"translator"}{i}{j}', param=tensor)
        return translators



    def init_sindy_coeff_stack(self):
        n = self.params['n_encoders']
        coeff_stack = torch.stack([self.init_sindy_coefficients() for i in range(n)])
        return torch.nn.Parameter(coeff_stack, requires_grad=True)



    def init_sindy_coefficients(self):
        library_dim = self.params['library_dim']
        latent_dim = self.params['latent_dim']
        initializer, init_param = self.initializer()
        return get_initialized_weights([library_dim, latent_dim], initializer,
                                       init_param = init_param, device = self.device)


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


    def criterion__function(self, vec):
        criterion = self.params['criterion']
        if criterion == 'stability':
            zero_threshold = self.params['zero_threshold']
            accept_threshold = self.params['accept_threshold']
            return stability_criterion(vec, zero_threshold, accept_threshold)
        elif criterion == 'avg':
            zero_threshold = self.params['zero_threshold']
            return avg_criterion(vec, zero_threshold)


    def num_active_coeffs(self):
        return torch.sum(self.coefficient_mask)


    def aggregate(self, tensors, agr_key='mean', agr_dim=0):
        try:
            tensors = torch.stack(tensors)
        except BaseException:
            pass
        if agr_key == 'median':
            return torch.median(tensors, agr_dim)[0]
        if agr_key == 'mean':
            return torch.mean(tensors, agr_dim)


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


    def sub_dz(self, x, dx, encode_idx):
        translator = self.translators[encode_idx]
        encoder = self.params['indep_models'].Encoders[encode_idx]
        encoder_weights, encoder_biases = self.get_weights(encoder['encoder_layers'])
        translator_weights, translator_biases = self.get_comp_weights(translator['translator_layers'])
        activation = self.params['activation']

        zi = encoder['encoder'](x)
        dzi = z_derivative(x, dx, encoder_weights, encoder_biases, activation).T

        dz = residual_z_derivative(zi, dzi, translator_weights, translator_biases, activation).T
        return dz


    def sindy_predict(self, z, encode_idx = None, coeffs = []):
        Theta = self.Theta(z)
        mask = self.coefficient_mask
        if not len(coeffs):
            coeffs = self.sindy_coeffs[encode_idx]
        return torch.matmul(Theta, mask * coeffs)


    def dx_decode(self, z, dz_pred):
        decoder_weights, decoder_biases = self.get_weights(self.Decoder_layers)
        activation = self.params['activation']
        dx_decode = z_derivative(z, dz_pred, decoder_weights, decoder_biases, activation = activation).T
        return dx_decode


    def sub_dx_loss(self, x_translate, dx, dz_pred):
        dx_pred = self.dx_decode(x_translate, dz_pred)
        criterion = nn.MSELoss()
        loss = self.params['loss_weight_sindy_x'] * (criterion(dx_pred, dx))
        return loss


    def decode_loss(self, x, x_pred):
        criterion = nn.MSELoss()
        loss = self.params['loss_weight_decoder'] * criterion(x, x_pred)
        return loss


    def reg_loss(self):
        return self.params['loss_weight_sindy_regularization'] * torch.mean(torch.abs(self.sindy_coeffs))


    def sub_corr_loss(self, Z, encode_idx):
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
            corr_sum  += (a/((b * c)**(1/2)))

        return -self.params['loss_weight_corr'] * corr_sum * (1/len(zs))


    def corr_loss(self, x_translate_stack):
        corr_loss  = 0
        for encode_idx in range(len(self.translators)):
            corr_loss += self.sub_corr_loss(x_translate_stack, encode_idx)
        return corr_loss/len(self.translators)


    def val_test(self, x, dx, x_translate_stack, agr_key='mean'):
        if self.refresh_val_dict:
            self.val_dict = {'E_Decoder': [],  'E_Sindy_x': [], 'E_agr_Decoder': [], 'E_agr_Sindy_x': []}

        criterion = nn.MSELoss()

        agr_x_translate = self.collapse(x_translate_stack, agr_key=agr_key)
        agr_x_decomp_decode = self.Decoder(agr_x_translate)

        agr_coeffs = torch.mean(self.sindy_coeffs, dim = 0)
        agr_dz_pred = self.sindy_predict(agr_x_translate, coeffs=agr_coeffs)
        agr_dx_pred  = self.dx_decode(agr_x_translate, agr_dz_pred)

        agr_decomp_loss2 = float(self.decode_loss(agr_x_decomp_decode, x).detach().cpu())
        agr_dx_decode_loss2 = criterion(agr_dx_pred, dx).detach().cpu() * self.params['loss_weight_sindy_x']

        self.val_dict['E_agr_Decoder'].append(agr_decomp_loss2)
        self.val_dict['E_agr_Sindy_x'].append(agr_dx_decode_loss2)

        self.refresh_val_dict = False
        return True


    def sub_loss(self, x, dx, encode_idx):
        encoder = self.params['indep_models'].Encoders[encode_idx]['encoder']
        translator = self.translators[encode_idx]['translator']
        x_translate = translator(encoder(x))
        loss_dicts = []

        dz_pred = self.sindy_predict(x_translate, encode_idx)
        x_decode = self.Decoder(x_translate)

        decoder_loss = self.decode_loss(x_decode, x)
        sindy_x_loss = self.sub_dx_loss(x_translate, dx, dz_pred)
        loss_dicts.append({'decoder': decoder_loss, 'sindy_x': sindy_x_loss})
        return dict_mean(loss_dicts), x_translate


    def rand_decode_indexes(self, k = 2):
        candidate_indexes = list(range(self.params['n_encoders']))
        decode_indexes = random.choices(candidate_indexes, k = k)
        return decode_indexes[0]


    def Loss(self, x, dx):
        sub_loss_dicts = []
        x_translates = []
        for encode_idx in range(self.params['n_encoders']):
            sub_loss_dict, x_translate = self.sub_loss(x, dx, encode_idx)
            sub_loss_dicts.append(sub_loss_dict)
            x_translates.append(x_translate)

        loss_dict = dict_mean(sub_loss_dicts)
        x_translate_stack = torch.concat(x_translates, dim = 1)
        corr_loss =  self.corr_loss(x_translate_stack)
        loss_dict['reg'] = self.reg_loss()
        loss_dict['sindy_z'] = corr_loss
        loss =  loss_dict['sindy_z'] #loss_dict['decoder'] + loss_dict['sindy_x'] + loss_dict['reg'] + loss_dict['sindy_z']
        if self.params['eval']:
            self.val_test(x, dx, x_translate_stack)
        return loss, loss_dict
