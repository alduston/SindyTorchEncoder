import torch
import torch.nn as nn
from sindy_utils import z_derivative, get_initialized_weights, sindy_library_torch
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
from itertools import permutations


def col_permutations(M):
    M_permutes = []
    columns = list(range(M.shape[1]))
    for perm in permutations(columns):
        M_permuted = np.stack([M[:, idx] for idx in perm])
        M_permutes.append(np.transpose(M_permuted))
    return M_permutes


def coeff_pattern_loss(pred_coeffs, true_coeffs, binary = True):
    pred_coeffs = deepcopy(pred_coeffs).detach().cpu().numpy()
    true_coeffs = deepcopy(true_coeffs).detach().cpu().numpy()
    if binary:
        pred_coeffs[np.where(np.abs(pred_coeffs) <.1)] = 0
        pred_coeffs[np.where(np.abs(pred_coeffs) >= .1)] = 1
        pred_coeffs = np.asarray((np.abs(pred_coeffs) ** 0.000000000001),int)
        true_coeffs = np.asarray(np.abs(true_coeffs) ** 0.00000000001,int)

    losses = []
    for permutation in col_permutations(true_coeffs):
        leq_M = pred_coeffs[np.where(pred_coeffs + .1 < permutation)]
        L_minus = len(leq_M)
        losses.append(L_minus)
    return min(losses)/np.sum(true_coeffs > 0)


def get_coeff_loss(model,true_coeffs, mask = []):
    if len(mask):
        return coeff_pattern_loss(mask, true_coeffs)
    else:
        try:
            coeff_loss_val = coeff_pattern_loss(model.coefficient_mask, true_coeffs)
        except BaseException:
            coeff_loss_val = 0
            for mask in model.coefficient_masks:
                pred_coeffs = mask
                coeff_loss_val += coeff_pattern_loss(pred_coeffs, true_coeffs)
            coeff_loss_val /= len(model.coefficient_masks)
    return coeff_loss_val


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


def binarize(tensor):
    binary_tensor = (tensor != 0).float() * 1
    return binary_tensor


def avg_criterion(vec, zero_threshold = .1):
    return torch.abs(torch.mean(vec)) >  zero_threshold


def stability_criterion(vec, zero_threshold = .1, accept_threshold = .6):
    return (sum([abs(val) > zero_threshold for val in vec])/len(vec)) > accept_threshold


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
        self.dtype = torch.float32
        params['device'] = self.device
        self.params = params
        self.activation_f = self.get_activation_f(params)
        self.params['activation_f'] =  self.activation_f
        self.criterion_f = self.criterion_function

        self.Encoders, self.Decoders, self.Sindy_coeffs =  self.init_sub_components()
        self.Encode_indexes, self.Decode_indexes = self.init_ED_indexes()
        self.coefficient_masks = self.get_indep_coefficient_masks()
        self.torch_params = self.get_params()

        sindy_shape = [self.params['n_encoders']] + list(self.Sindy_coeffs[0].shape)
        self.Sindy_coeffs = torch.nn.Parameter(torch.stack(self.Sindy_coeffs).reshape(sindy_shape))

        self.iter_count = torch.tensor(0, device = device)
        self.epoch = torch.tensor(0, device = device)
        self.stage = 1
        self.refresh_val_dict = True

        self.exp_label = params['exp_label']
        try:
            self.true_coeffs = torch.tensor(params['true_coeffs'], dtype=self.dtype, device=self.device)
        except TypeError:
            self.true_coeffs = None

        self.item_loss_dict = self.get_item_loss_dict()


    def get_indep_coefficient_masks(self):
        n = self.params['n_encoders']
        coefficient_masks = [torch.tensor(deepcopy(self.params['coefficient_mask']),
                                          dtype=self.dtype, device=self.device) for i in range(n)]
        return torch.stack(coefficient_masks)


    def get_item_loss_dict(self):
        loss_keys = ['decoder', 'sindy_x', 'sindy_z', 'active_coeffs', 'coeff']
        item_loss_dict = {}
        for loss_key in loss_keys:
            item_loss_dict[f'{loss_key}_agg'] = []
            for encode_idx in range(self.params['n_encoders']):
                for decode_idx in range(self.params['n_decoders']):
                        item_loss_dict[f'{loss_key}_{encode_idx}_{decode_idx}'] = []
        return item_loss_dict


    def get_params(self):
        params = []
        for i, model in enumerate(self.Encoders):
            params += model['encoder'].parameters()
            for j, tensor in enumerate(model['encoder'].parameters()):
                self.register_parameter(name=f'{"encoder"}{i}{j}', param=tensor)
        for i, model in enumerate(self.Decoders):
            params += model['decoder'].parameters()
            for j, tensor in enumerate(model['decoder'].parameters()):
                self.register_parameter(name=f'{"decoder"}{i}{j}', param=tensor)
        return params


    def s2_param_update(self):
        for (name, param) in self.named_parameters():
            if name.startswith('decoder') or name.startswith('encoder') or name.startswith('s1'):
                param.requires_grad = False
        return True


    def init_sub_components(self):
        Encoders = []
        for i in range(self.params['n_encoders']):
            encoder, encoder_layers = self.Encoder(self.params)
            Encoders.append({'encoder': encoder, 'encoder_layers': encoder_layers})
        Decoders = []
        for i in range(self.params['n_decoders']):
            decoder, decoder_layers = self.Decoder(self.params)
            Decoders.append({'decoder': decoder, 'decoder_layers': decoder_layers})
        Sindy_coeffs = []
        for i,encoder in enumerate(Encoders):
            #for j,decoder in enumerate(Decoders):
            sindy_coeffs = torch.nn.Parameter(self.init_sindy_coefficients(), requires_grad=True)
            Sindy_coeffs.append(sindy_coeffs)
        return Encoders, Decoders, Sindy_coeffs


    def init_ED_indexes(self):
        Encode_indexes = []
        Decode_indexes = []
        n = self.params['batch_size']
        base_indexes = np.asarray(range(self.params['batch_size']))

        for encoder in self.Encoders:
            encoder_idxs = np.random.choice(base_indexes, size  = n, replace = True) #self.params['replacement'])
            encoder_idxs = torch.tensor(encoder_idxs, device = self.device, dtype = self.dtype).long()
            Encode_indexes.append(encoder_idxs)

        ne = len(self.Decoders)
        for i,decoder in enumerate(self.Decoders):
            decoder_idxs = base_indexes[(n//ne) * i: (n//ne) * (i+1)]
            decoder_idxs = torch.tensor(decoder_idxs, device=self.device, dtype=self.dtype).long()
            Decode_indexes.append(decoder_idxs)
        return Encode_indexes,Decode_indexes


    def ed_sample(self, x, dx, encode_idx, decode_idx):
        e_indexes = self.Encode_indexes[encode_idx]%len(x)
        ed_indexes = e_indexes[self.Decode_indexes[decode_idx]]%len(e_indexes)
        xed = x[ed_indexes]
        dxed = dx[ed_indexes]
        return xed, dxed


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


    def criterion_function(self, vec):
        criterion = self.params['criterion']
        if criterion == 'stability':
            zero_threshold = self.params['zero_threshold']
            accept_threshold = self.params['accept_threshold']
            return stability_criterion(vec, zero_threshold, accept_threshold)
        elif criterion == 'avg':
            zero_threshold = self.params['zero_threshold']
            return avg_criterion(vec, zero_threshold)


    def decoder_weights(self, decoder_idx):
        decoder_weights = []
        decoder_biases = []
        decoder_layers = self.Decoders[decoder_idx]['decoder_layers']
        for layer in  decoder_layers[0::2]:
            decoder_weights.append(layer.weight)
            decoder_biases.append(layer.bias)
        return decoder_weights, decoder_biases


    def encoder_weights(self, encoder_idx):
        encoder_weights = []
        encoder_biases = []
        encoder_layers = self.Encoders[encoder_idx]['encoder_layers']
        for layer in encoder_layers[0::2]:
            encoder_weights.append(layer.weight)
            encoder_biases.append(layer.bias)
        return encoder_weights, encoder_biases


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


    def aggregate(self, tensors, agr_key = 'mean', agr_dim = 0):
        try:
            tensors = torch.stack(tensors)
        except BaseException:
            pass
        if agr_key == 'median':
            return torch.median(tensors,agr_dim)[0]
        if agr_key == 'mean':
            return torch.mean(tensors,agr_dim)


    def Theta(self, z):
        poly_order = self.params['poly_order']
        include_sine = self.params['include_sine']
        latent_dim = self.params['latent_dim']
        Theta = sindy_library_torch(z, latent_dim, poly_order, include_sine, device = self.device)
        return Theta


    def sub_forward_s1(self,  x, encode_idx, decode_idx):
        z = self.Encoders[encode_idx]['encoder'](x)
        x_decode = self.Decoders[decode_idx]['decoder'](z)
        return z, x_decode


    def agr_forward_s1(self,  x):
        z = self.aggregate([submodel['encoder'](x) for submodel in self.Encoders])
        x_decode = self.aggregate([submodel['decoder'](z) for submodel in self.Decoders])
        return z, x_decode


    def agr_forward_s1_indep(self, x):
        zs = [submodel['encoder'](x) for submodel in self.Encoders]
        z = self.aggregate(zs)
        x_decodes = [submodel['decoder'](zi) for zi, submodel in zip(zs, self.Decoders)]
        x_decode = self.aggregate(x_decodes, agr_key='mean')
        return z, x_decode


    def decoder_loss(self, x, x_pred):
        criterion = nn.MSELoss()
        loss =  self.params['loss_weight_decoder'] * criterion(x, x_pred)
        return loss


    def num_active_coeffs(self):
        return torch.mean(torch.stack([torch.sum(mask) for mask in self.coefficient_masks]))


    def sub_dz(self, x, dx, encode_idx):
        weights, biases = self.encoder_weights(encode_idx)
        activation = self.params['activation']

        sub_dz = z_derivative(x, dx, weights, biases, activation)
        return sub_dz


    def agr_dz(self, x, dx):
        sub_dzs = [self.sub_dz(x, dx, encode_idx) for encode_idx, subm in enumerate(self.Encoders)]
        return self.aggregate(sub_dzs)


    def z_loss(self, dz, dz_predict):
        criterion = nn.MSELoss()
        return self.params['loss_weight_sindy_z'] * criterion(dz, dz_predict.T)


    def sub_sindy_predict(self, z, coeffs, mask):
        Theta = self.Theta(z)
        return torch.matmul(Theta, mask * coeffs)


    def sub_dx_decode(self, z, dz_predict, decode_idx):
        weights, biases = self.decoder_weights(decode_idx)
        activation = self.params['activation']
        dx_decode = z_derivative(z,  dz_predict, weights,
                                 biases, activation=activation)
        return dx_decode


    def agr_dx_decode(self, z, dz_predict):
        return self.aggregate([self.sub_dx_decode(z, dz_predict,  decode_idx) for decode_idx in
                               range(self.params['n_decoders'])])


    def x_loss(self, dx, dx_decode):
        criterion = nn.MSELoss()
        return self.params['loss_weight_sindy_x'] * criterion(dx, dx_decode)


    def reg_loss(self, coeffs):
        reg_vals = coeffs
        reg_loss = self.params['loss_weight_sindy_regularization'] * torch.mean(torch.abs(reg_vals))
        return  reg_loss


    def latent_loss(self, x):
        zs = [submodel['encoder'](x) for submodel in self.Encoders]
        z_mean = torch.mean(torch.stack(zs), dim = 0)
        latent_variance = torch.mean((torch.stack(zs) - z_mean)**2)
        return self.params['loss_weight_latent'] * latent_variance


    def s1_sub_items(self, x, dx, encode_idx, decode_idx):
        xed, dxed = self.ed_sample(x, dx, encode_idx, decode_idx)
        coeffs = self.Sindy_coeffs[encode_idx]
        z, x_decode = self.sub_forward_s1(xed, encode_idx, decode_idx)

        dz = self.sub_dz(xed, dxed, encode_idx)
        mask = self.coefficient_masks[encode_idx]
        dz_predict = self.sub_sindy_predict(z, coeffs, mask)
        dx_decode = torch.transpose(self.sub_dx_decode(z, dz_predict, decode_idx), 0, 1)

        sub_items = {'x': xed, 'dx': dxed, 'z': z, 'x_decode': x_decode, 'dz': dz, 'mask': mask,
                     'dz_predict':dz_predict, 'dx_decode':  dx_decode, 'coeffs': coeffs}
        return sub_items


    def s1_agr_items(self, x, dx):
        agr_shape = [self.params['n_encoders']*self.params['n_decoders']] + list(self.Sindy_coeffs.shape[-2:])
        agr_coeffs = self.aggregate(self.Sindy_coeffs.reshape(agr_shape))
        z, x_decode = self.agr_forward_s1(x)
        dz = self.agr_dz(x, dx)
        dz_predict = self.sub_sindy_predict(z, agr_coeffs)
        dx_decode = torch.transpose(self.agr_dx_decode(z, dz_predict), 0, 1)
        items = {'x': x, 'dx': dx, 'z': z, 'x_decode': x_decode, 'dz': dz,
                     'dz_predict': dz_predict, 'dx_decode': dx_decode, 'coeffs': agr_coeffs}
        return items


    def s1_sub_loss(self, items):
        decoder_loss = self.decoder_loss(items['x'], items['x_decode'])
        sindy_z_loss = self.z_loss(items['dz'], items['dz_predict'])
        sindy_x_loss = self.x_loss(items['dx'],items['dx_decode'])
        reg_loss = self.reg_loss(items['coeffs'])

        sub_loss = decoder_loss + sindy_z_loss + sindy_x_loss + reg_loss
        sub_losses = {'decoder': decoder_loss, 'sindy_z': sindy_z_loss, 'sindy_x': sindy_x_loss,  'reg': reg_loss}
        return sub_loss, sub_losses


    def update_item_losses(self, loss_dict, encode_idx, decode_idx, agg = False):
        for loss_key in ['decoder', 'sindy_x', 'active_coeffs', 'coeff']:
            if agg:
                key = f'{loss_key}_agg'
            else:
                key = f'{loss_key}_{encode_idx}_{decode_idx}'
            val = loss_dict[loss_key].detach().cpu().numpy()
            self.item_loss_dict[key].append(val)

        return True


    def error_plot(self, x, dx):
        x_errors = [0.0]
        y_errors = [0.0]
        for encode_idx in range(self.params['n_encoders']):
            for decode_idx in range(self.params['n_decoders']):
                z = self.Encoders[encode_idx]['encoder'](x)
                x_p = self.Decoders[decode_idx]['decoder'](z)
                (x_error,y_error) = ((x_p  - x)[0,:2].detach().cpu())
                x_errors.append(float(x_error))
                y_errors.append(float(y_error))

        plt.scatter(x_errors,  y_errors)
        plt.savefig('errors.png')
        clear_plt()
        return True


    def s1_Loss_agr(self, x, dx):
        s1_items = self.s1_agr_items(x, dx)
        loss, loss_dict = self.s1_sub_loss(s1_items)
        latent_loss = self.latent_loss(x)
        loss_dict['latent'] = latent_loss
        loss += latent_loss
        return loss, loss_dict


    def agr_dx_loss(self, x, dx):
        dx_decodes = []
        for encode_idx in range(self.params['n_encoders']):
            z, x_decode =  self.sub_forward_s1(x, encode_idx, encode_idx)
            coeffs = self.Sindy_coeffs[encode_idx]
            mask = self.coefficient_masks[encode_idx]

            dz_predict = self.sub_sindy_predict(z, coeffs, mask)
            dx_decodes.append(self.sub_dx_decode(z, dz_predict, encode_idx))
        agr_dx_decode = self.aggregate(dx_decodes, agr_key='mean')
        criterion = nn.MSELoss()
        return self.params['loss_weight_sindy_x'] * criterion(agr_dx_decode.T, dx)

    def agr_decode_loss(self, x):
        x_decodes = []
        for encode_idx in range(self.params['n_encoders']):
            z,x_decode =  self.sub_forward_s1(x, encode_idx, encode_idx)
            x_decodes.append(x_decode)
        agr_x_decode = self.aggregate(x_decodes, agr_key='mean')
        criterion = nn.MSELoss()
        return self.params['loss_weight_decoder'] * criterion(agr_x_decode, x)


    def val_test(self, x, dx):
        if self.refresh_val_dict:
            self.val_dict = {'E_Decoder': [],  'E_Sindy_x': []}

        agr_decoder_loss = self.agr_decode_loss(x).detach().cpu()
        agr_dx_loss = self.agr_dx_loss(x, dx).detach().cpu()
        self.val_dict['E_Decoder'].append(agr_decoder_loss)
        self.val_dict['E_Sindy_x'].append(agr_dx_loss)
        self.refresh_val_dict = False
        return True

    def s1_Loss(self, x, dx):
        losses = []
        loss_dicts = []

        for encode_idx in range( self.params['n_encoders']):
            decode_idx = encode_idx
            s1_items = self.s1_sub_items(x, dx, encode_idx, decode_idx)

            loss, loss_dict = self.s1_sub_loss(s1_items)
            losses.append(loss)
            loss_dicts.append(loss_dict)
            if self.params['cp_batch']:
                loss_dict['active_coeffs'] = torch.sum(torch.abs(s1_items['mask']))
                loss_dict['coeff'] = torch.tensor(get_coeff_loss(self, self.true_coeffs, s1_items['mask']),
                                                  device = self.device, dtype = self.dtype)
                self.update_item_losses(loss_dict, encode_idx, decode_idx)

        if self.params['cp_batch']:
            self.val_test(x, dx)
            agr_loss_vals = {'decoder': self.val_dict['E_Decoder'][-1],
                             'sindy_x':  self.val_dict['E_Sindy_x'][-1],
                             'active_coeffs': self.num_active_coeffs(),
                             'coeff': torch.tensor(get_coeff_loss(self, self.true_coeffs),
                                                  device = self.device, dtype = self.dtype)}
            self.update_item_losses(agr_loss_vals, 0, 0, agg = True)


        loss_dict = dict_mean(loss_dicts)
        loss = torch.mean(torch.stack(losses))
        return loss, loss_dict


    def Loss(self, x, dx):
        loss, loss_dict = self.s1_Loss(x, dx)
        if self.params['cp_batch']:
            self.params['cp_batch'] = False
        return  loss, loss_dict




#0.00139449, Sindy_x: 0.000346457