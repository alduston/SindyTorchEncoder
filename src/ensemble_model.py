import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from scipy.stats import anderson
from sindy_utils import z_derivative, z_derivative_order2,\
    get_initialized_weights, sindy_library_torch, sindy_library_torch_order2
import warnings
from copy import copy, deepcopy
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from geom_median.torch import compute_geometric_median
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
    return binary_tensor


def avg_criterion(vec, zero_threshold = .1):
    return torch.abs(torch.mean(vec)) >  zero_threshold


def stability_criterion(vec, zero_threshold = .1, accept_threshold = .6):
    return (sum([abs(val) > zero_threshold for val in vec])/len(vec)) > accept_threshold


def anderson_criterion(vec, zero_threshold = .1):
    alpha = avg_criterion(vec,zero_threshold)
    res = anderson(vec.numpy())
    beta = res.statistic > res.critical_values[2]
    bool = not (not alpha and beta)
    return bool


def binarize_xavier(tensor):
    binary_tensor = (tensor >= 0).float() * 2
    return .5 * (binary_tensor - 1.0)


def gmean(tensor, dim = 0):
    min = torch.min(tensor)
    p_tensor = tensor + min + 1
    log_x = torch.log(p_tensor)
    return torch.exp(torch.mean(log_x, dim=dim)) - min - 1


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
        self.criterion_f = self.get_criterion_f(params)

        submodels = self.init_submodels()
        self.submodels = submodels
        self.coefficient_mask = torch.tensor(params['coefficient_mask'], dtype=torch.float32, device=self.device)
        self.sub_model_coeffs = torch.nn.Parameter(torch.stack([submodel['sindy_coeffs'] for submodel in self.submodels]),
                                                   requires_grad= True)
        self.sindy_coeffs = self.get_sindy_coefficients()


        decoder, decoder_layers = self.Decoder(self.params)
        self.decoder = decoder
        self.decoder_layers = decoder_layers

        self.iter_count = torch.tensor(0, device = device)
        self.epoch = torch.tensor(0, device = device)


        self.num_active_coeffs = torch.sum(copy(self.coefficient_mask)).cpu().detach().numpy()
        self.exp_label = params['exp_label']
        self.true_coeffs = torch.tensor(params['true_coeffs'], dtype=torch.float32, device=self.device)
        self.torch_params =  self.get_params()



    def get_params(self):
        params = list(self.parameters())
        for i,model in enumerate(self.submodels):
            params += model['encoder'].parameters()
        for i,tensor in enumerate(params):
            self.register_parameter(name=f'param{i}', param = tensor)
        return params


    def init_submodel(self, idx):
        encoder, encoder_layers = self.Encoder(self.params)
        sindy_coeffs =  torch.nn.Parameter(self.init_sindy_coefficients(), requires_grad=True)
        submodel = {'encoder' : encoder, 'encoder_layers': encoder_layers,
                    'sindy_coeffs': sindy_coeffs}
        return submodel


    def init_submodels(self):
        submodels = []
        for bag_idx in range(self.params['nbags']):
            submodel = self.init_submodel(bag_idx)
            submodels.append(submodel)
        return submodels


    def get_network(self, input_dim, widths, final_dim, activation):
        layers = []
        for output_dim in widths:
            layer = nn.Linear(input_dim, output_dim)
            nn.init.xavier_uniform(layer.weight)
            nn.init.constant_(layer.bias.data, 0)

            input_dim = output_dim
            layers.append(layer)
            layers.append(activation)

        layer = nn.Linear(input_dim,  final_dim)
        nn.init.xavier_uniform(layer.weight)
        nn.init.constant_(layer.bias.data, 0)
        layers.append(layer)
        net = nn.Sequential(*layers)
        return net, layers


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


    def Translator(self, params):
        activation_function = self.get_activation_f(params)
        input_dim = params['latent_dim']
        widths = [2 * input_dim]
        final_dim = params['input_dim']
        return self.get_network(input_dim, widths, final_dim, activation_function)


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


    def get_criterion_f(self, params):
        criterion = params['criterion']
        if criterion == 'stability':
            zero_threshold = params['zero_threshold']
            accept_threshold = params['accept_threshold']
            criterion_function = lambda vec: stability_criterion(vec, zero_threshold, accept_threshold)
        elif criterion == 'avg':
            zero_threshold = params['zero_threshold']
            criterion_function = lambda vec: avg_criterion(vec, zero_threshold)
        elif criterion == 'anderson':
            zero_threshold = params['zero_threshold']
            criterion_function = lambda vec: anderson_criterion(vec, zero_threshold)
        return criterion_function


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


    def Encoder_weights(self):
        encoder_weights = []
        encoder_biases = []
        for layer in self.encoder_layers:
            try:
                encoder_weights.append(layer.weight)
                encoder_biases.append(layer.bias)
            except AttributeError:
                pass
        return encoder_weights, encoder_biases


    def sub_dz(self, x, dx, sub_idx):
        encoder_weights, encoder_biases = self.encoder_weights(sub_idx)
        activation = self.params['activation']
        sub_dz = z_derivative(x, dx, encoder_weights, encoder_biases, activation)
        return sub_dz


    def agr_dz(self, x, dx):
        activation = self.params['activation']
        dzs = []
        for sub_idx in range(self.params['nbags']):
            encoder_weights, encoder_biases = self.encoder_weights(sub_idx)
            dz_sub = z_derivative(x, dx, encoder_weights, encoder_biases, activation)
            dzs.append(dz_sub)
        dz = self.aggregate(torch.stack(dzs))
        return dz


    def masked_dz(self, x, dx):
        sub_models = self.submodels
        masks = [deepcopy(submodel['output_mask']) for submodel in self.submodels]
        dz_stack = torch.stack([self.sub_dz(x, dx, bag_idx) for bag_idx in range(len(sub_models))])
        dz_masks = torch.stack([self.reshape_mask(mask, dz.T.shape, first=False).T for mask, dz in zip(masks, dz_stack)])
        dz = torch.sum(dz_stack*dz_masks, 0)
        return dz


    def dz(self, x, dx):
        if self.params['eval']:
            dz = self.agr_dz(x,dx)
        else:
            dz = self.masked_dz(x, dx)
        return dz


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
        weights =  get_initialized_weights([library_dim, latent_dim], initializer,
                                       init_param = init_param, device = self.device)
        if self.params['coefficient_initialization']=='binary_xavier':
            weights = binarize_xavier(weights)
        return weights


    def get_sindy_coefficients(self):
        sindy_coeffs = self.aggregate(self.sub_model_coeffs)
        return sindy_coeffs


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
        masks = [deepcopy(submodel['output_mask']) for submodel in self.submodels]
        coeffs = self.sub_model_coeffs
        coeff_mask = self.coefficient_mask
        pred_stack = torch.stack([torch.matmul(Theta, coeff_mask * coeff_m) for coeff_m in coeffs])
        pred_masks = torch.stack([self.reshape_mask(mask, pred.shape, first=False) for mask, pred in zip(masks, pred_stack)])
        sindy_predict = torch.sum(pred_stack*pred_masks, 0)
        return sindy_predict


    def sindy_predict(self, z, x = None, dx = None):
        Theta = self.Theta(z, x, dx)
        if self.params['eval']:
            sindy_coefficients = self.get_sindy_coefficients()
            return torch.matmul(Theta, self.coefficient_mask * sindy_coefficients)
        else:
            return self.masked_predict(Theta)


    def calc_coefficient_mask(self):
        sindy_coefficients = self.get_sindy_coefficients()
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
        sindy_coefficients = self.get_sindy_coefficients()
        coefficient_mask = self.coefficient_mask
        return sindy_coefficients * coefficient_mask


    def sub_dx(self, bag_idx, x, dx, coeff_m_p = []):
        sub_model = self.submodels[bag_idx]
        z_p = sub_model['encoder'](x)
        Theta = self.Theta(z_p, x, dx)
        if not len(coeff_m_p):
            coeff_m_p = self.sub_model_coeffs[bag_idx]
        sindy_predict = torch.matmul(Theta, self.coefficient_mask * coeff_m_p)
        decoder_weights, decoder_biases = self.decoder_weights()
        activation = self.params['activation']
        sub_dx = z_derivative(z_p, sindy_predict, decoder_weights, decoder_biases, activation=activation)
        return sub_dx


    def agr_dx(self, x, dx, coeff = []):
        dx_decodes = torch.stack([self.sub_dx(bag_idx, x, dx, coeff) for bag_idx in range(self.params['nbags'])])
        agr_dx_decode = self.aggregate(dx_decodes)
        return agr_dx_decode


    def get_dx_errors(self, x, dx):
        sub_models = self.submodels
        for bag_idx, sub_model in enumerate(sub_models):
            dx_p = self.sub_dx(bag_idx, x, dx)
            error = deepcopy(torch.mean((((dx_p.T - dx) ** 2))))
            self.params['dx_errors'][bag_idx] += float(error.detach())
            if bag_idx == 0:
                coeff = self.aggregate(self.sub_model_coeffs)
                dx_agr = self.agr_dx(x, dx, coeff)
                error = deepcopy(torch.mean((((dx_agr.T - dx) ** 2))))
                self.params['dx_errors'][-1] += float(error.detach())
        return True


    def get_decode_errors(self, x):
        sub_models = self.submodels
        for bag_idx, sub_model in enumerate(sub_models):
            z_p = sub_model['encoder'](x)
            x_p = self.decoder(z_p)
            error = deepcopy(torch.mean((((x_p - x) ** 2))))
            self.params['decode_errors'][bag_idx] += float(error.detach())
            if bag_idx == 0:
                x_agr = self.decoder(self.agr_forward(x)[0])
                error = deepcopy(torch.mean((((x_agr - x) ** 2))))
                self.params['decode_errors'][-1] += float(error.detach())
        return True

    def mask_indexes(self, mask, N = torch.inf):
        idx = torch.where(torch.sum(mask, 1) != 0)[0].long()
        idx = idx[idx < N]
        return idx

    def dx_decode(self,z, x, dx = None):
        if self.params['eval']:
            dx_decode = self.agr_dx(x, dx)
            self.get_dx_errors(x, dx)
            self.get_decode_errors(x)
        else:
            sindy_predict = self.sindy_predict(z, x, dx)
            decoder_weights, decoder_biases = self.decoder_weights()
            activation = self.params['activation']
            dx_decode = z_derivative(z, sindy_predict, decoder_weights,
                                 decoder_biases, activation=activation)
        return dx_decode


    def ddx_decode(self,z, x, dx):
        sindy_predict = self.sindy_predict(z, x, dx)
        decoder_weights, decoder_biases = self.decoder_weights()
        activation = self.params['activation']
        dz = self.dz(x,dx)
        dx_decode, ddx_decode = z_derivative_order2(z, dz, sindy_predict, decoder_weights, decoder_biases, activation=activation)
        return dx_decode, ddx_decode


    def get_sub_model_coeffs(self):
        return self.sub_model_coeffs


    def sub_model_masks(self):
        return torch.stack([submodel['output_mask'] for submodel in self.submodels])


    def aggregate(self, tensors, agr_key = 'median'):
        if agr_key == 'median':
            return torch.median(tensors,0)[0]
        if agr_key == 'mean':
            return torch.mean(tensors,0)


    def agr_forward(self, x):
        submodels = self.submodels
        z_stack = torch.stack([submodel['encoder'](x) for submodel in submodels])
        z = self.aggregate(z_stack)
        return z, z_stack


    def agr_decode(self, z):
        submodels = self.submodels
        x_stack = torch.stack([submodel['decoder'](z) for submodel in submodels])
        x_pred = self.aggregate(x_stack)
        return x_pred, x_stack


    def stack_decode(self, z_stack):
        x_stack = torch.stack([self.decoder(z) for z in z_stack])
        return self.aggregate(x_stack)


    def masked_forward(self, x):
        submodels = self.submodels
        masks = self.sub_model_masks()
        z_stack = torch.stack([submodel['encoder'](x) for submodel in submodels])
        z_masks = torch.stack([self.reshape_mask(mask, z.shape, first = False) for mask, z in zip(masks, z_stack)])
        z = torch.sum(z_masks * z_stack, 0)
        return z, z_stack


    def masked_decode(self, z):
        submodels = self.submodels
        masks = self.sub_model_masks()
        x_stack = torch.stack([submodel['decoder'](z) for submodel in submodels])
        x_masks = torch.stack([self.reshape_mask(mask, x.shape, first=False) for mask,x in zip(masks, x_stack)])
        x_pred = torch.sum(x_masks * x_stack, 0)
        return x_pred, x_stack


    def forward(self, x):
        if self.params['eval']:
            z, z_stack = self.agr_forward(x)
            x_decode = self.decoder(z)
        else:
            z, z_stack = self.masked_forward(x)
            x_decode = self.decoder(z)
        return x_decode, z, z_stack


    def decoder_loss(self, x, x_pred):
        criterion = nn.MSELoss()
        loss =  self.params['loss_weight_decoder'] *  self.params['bagn_factor'] * criterion(x, x_pred)
        return loss


    def sindy_reg_loss(self, alt = False):
        rescale = 1
        if self.params['eval']:
            reg_loss = self.params['loss_weight_sindy_regularization'] * torch.mean(torch.abs(self.sindy_coeffs))
        else:
            sub_coeffs = self.sub_model_coeffs
            if alt:
                sub_coeffs = torch.sum(sub_coeffs, dim = 0)
                rescale = (1 / self.params['nbags'])
            reg_loss = self.params['loss_weight_sindy_regularization'] * torch.mean(torch.abs(sub_coeffs))
        return reg_loss * rescale


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


    def latent_loss(self, z_stack):
        z_med = compute_geometric_median([z for z in z_stack], weights=None, per_component=True).median
        stack_var = torch.mean((z_stack - z_med)**2)
        return self.params['loss_weight_latent'] * stack_var


    def alt_forward(self, x, dx):
        z_stack = []
        decode_stack = []
        x_stack = []
        dx_stack = []
        N = len(x)
        for bag_idx, submodel in enumerate(self.submodels):
            mask = self.sub_model_masks()[bag_idx]
            mask_idx = self.mask_indexes(mask, N)
            sub_x = x[mask_idx]
            sub_dx = dx[mask_idx]
            if len(sub_x):
                z = self.submodels[bag_idx]['encoder'](sub_x)
                x_decode = self.decoder(z)
                z_stack.append(z)
                decode_stack.append(self.decoder(x_decode))
                x_stack.append(sub_x)
                dx_stack.append(sub_dx)

        x_stack = torch.stack(x_stack)
        dx_stack = torch.stack(dx_stack)
        z_stack = torch.stack(z_stack)
        decode_stack = torch.stack(decode_stack)

        return x_stack, dx_stack, z_stack, decode_stack


    def alt_sindy_x_loss(self, x_stack, z_stack):
        dx_loss = 0
        bag_idx = 0
        for x,dx in zip(x_stack, z_stack):
            dx_pred = self.sub_dx(bag_idx, x, dx)
            dx_loss += nn.MSELoss()(dx, dx_pred)
            bag_idx += 1
        return self.params['loss_weight_sindy_x'] * dx_loss


    def alt_decoder_loss(self, x_stack, decode_stack):
        decoder_loss = 0
        for x, x_decode in zip(x_stack, decode_stack):
            decoder_loss += nn.MSELoss()(x, x_decode)
        return self.params['loss_weight_decoder'] * decoder_loss


    def Loss(self, x, dx, ddx = None):
        x_decode, z, z_stack = self.forward(x)
        x_stack, dx_stack, z_stack2, decode_stack = self.alt_forward(x, dx)

        #decoder_loss = self.alt_decoder_loss(x_stack, decode_stack)
        #sindy_x_loss = self.alt_sindy_x_loss(x_stack, dx_stack)

        decoder_loss = self.decoder_loss(x, x_decode)
        sindy_z_loss = self.sindy_z_loss(z, x, dx, ddx)
        sindy_x_loss = self.sindy_x_loss(z, x, dx, ddx)
        reg_loss = self.sindy_reg_loss(alt = False)
        latent_loss = self.latent_loss(z_stack)

        loss_refinement = decoder_loss + sindy_z_loss + sindy_x_loss + latent_loss
        loss = loss_refinement + reg_loss
        losses = {'decoder': decoder_loss, 'sindy_z': sindy_z_loss, 'sindy_x': sindy_x_loss,
                  'latent': latent_loss, 'reg': reg_loss}
        losses = {key: self.params['print_factor'] * val for (key,val) in losses.items()}
        return loss, loss_refinement, losses


    def loss(self, x, x_decode, z, dx, ddx=None):
        return self.Loss(x, x_decode, z, dx, ddx)[0]


    def auto_Loss(self, x, dx, ddx=None):
        return self.Loss(x, dx, ddx)
