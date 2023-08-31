import torch
import torch.nn as nn
#from sindy_utils import z_derivative, get_initialized_weights, sindy_library_torch
#from copy import copy, deepcopy
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R


def rand_rotation_matrix(deflection=1.0):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random rotation. Small
    deflection => small perturbation.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    theta = np.random.uniform(0, 2.0 * deflection * np.pi)  # Rotation about the pole (Z).
    phi = np.random.uniform(0, 2.0 * np.pi)  # For direction of pole deflection.
    z = np.random.uniform(0, 2.0 * deflection)  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    # Compute the row vector S = Transpose(V) * R, where R is a simple
    # rotation by theta about the z-axis.  No need to compute Sz since
    # it's just Vz.

    st = np.sin(theta)
    ct = np.cos(theta)
    Sx = Vx * ct - Vy * st
    Sy = Vx * st + Vy * ct

    # Construct the rotation matrix  ( V Transpose(V) - I ) R, which
    # is equivalent to V S - R.

    M = np.array((
        (
            Vx * Sx - ct,
            Vx * Sy - st,
            Vx * Vz
        ),
        (
            Vy * Sx + st,
            Vy * Sy - ct,
            Vy * Vz
        ),
        (
            Vz * Sx,
            Vz * Sy,
            1.0 - z  # This equals Vz * Vz - 1.0
        )
    )
    )
    return M



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
    while widths[-1] > 2 * out_width:
        widths.append(widths[-1]//2)
    return widths[0:]


def diagnolize_weights(w_list):
    n = len(w_list)
    l,w =  w_list[0].shape
    device = w_list[0].device
    dtype = w_list[0].dtype
    diag_tensor = torch.zeros((n * l, n * w), device = device, dtype=dtype)
    for i in range(n):
        diag_tensor[(i) * l:(i+1) * l, (i) * w:(i+1) * w] += w_list[i]
    return diag_tensor


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

    def forward(self, x):
        n = self.n
        activation_f = self.activation_f
        x_linear = x[:, :n]
        return torch.concat([activation_f(self.lin_layer(x)), x_linear], dim=1)


class SindyNetCompEnsemble(nn.Module):
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
        self.params['indep_models'] = indep_models

        self.activation_f = indep_models.activation_f
        self.compressor, self.compressor_layers = self.Residual_Compressor(self.params)
        self.decompressor, self.decompressor_layers = self.Residual_Decompressor(self.params)
        self.decoder, self.decoder_layers = indep_models.Decoder(params)
        self.params['stacked_encoder'],self.params['stacked_encoder_layers'] = self.Stacked_encoder(self.params)
        self.params['stacked_decoder'], self.params['stacked_decoder_layers'] = self.Stacked_decoder(self.params)
        self.epoch = 0


    def get_residual_lat_activation(self, activation_f, input_dim):
        return InputActivation(activation_f, input_dim)


    def Stacked_encoder(self, params):
        indep_models = self.params['indep_models']
        n_encoders = params['n_encoders']
        weights = []
        biases = []
        for j in range(len(indep_models.encoder_weights(0)[0])):
            w_list = [indep_models.encoder_weights(0)[0][j] for i in range(n_encoders)]
            #w_list = [indep_models.encoder_weights(i)[0][j] for i in range(n_encoders)]
            w_stack = diagnolize_weights(w_list)
            weights.append(w_stack)

            b_stack = torch.concat([indep_models.encoder_weights(0)[1][j] for i in range(n_encoders)], dim=0)
            #b_stack = torch.concat([indep_models.encoder_weights(i)[1][j] for i in range(n_encoders)], dim=0)
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

        rotation_matrixes = [torch.tensor(rand_rotation_matrix(1),device=self.device,
                             dtype=self.dtype) for i in range(n_encoders)]
        rotations_tensor = diagnolize_weights(rotation_matrixes)
        rotation_layer = nn.Linear(latent_dim, latent_dim,  device=self.device)
        rotation_layer.weight.data = rotations_tensor
        rotation_layer.bias.data = torch.zeros(latent_dim, device = self.device, dtype=self.dtype)
        layers.append(rotation_layer)

        Stacked_encoder = nn.Sequential(*layers)

        return Stacked_encoder, layers


    def Stacked_decoder(self, params):
        indep_models = self.params['indep_models']
        n_decoders = params['n_decoders']
        weights = []
        biases = []



        for j in range(len(indep_models.decoder_weights(0)[0])):
            #w_list = [indep_models.decoder_weights(i)[0][j] for i in range(n_decoders)]
            w_list = [indep_models.decoder_weights(0)[0][j] for i in range(n_decoders)]
            w_stack = diagnolize_weights(w_list)
            weights.append(w_stack)

            #b_stack = torch.concat([indep_models.decoder_weights(i)[1][j] for i in range(n_decoders)],
                                   #dim=0)

            b_stack = torch.concat([indep_models.decoder_weights(0)[1][j] for i in range(n_decoders)],dim=0)

            biases.append(b_stack)

        activation_function = self.activation_f #self.get_activation_f(params)
        input_dim = params['latent_dim'] * n_decoders
        final_dim =  params['input_dim'] * n_decoders
        widths = np.asarray(params['widths']) * n_decoders

        layers = []
        rotation_matrixes = [torch.tensor(rand_rotation_matrix(1), device=self.device,
                                          dtype=self.dtype) for i in range(n_decoders)]
        rotations_tensor = diagnolize_weights(rotation_matrixes)
        rotation_layer = nn.Linear(input_dim, input_dim, device=self.device)
        rotation_layer.weight.data = rotations_tensor
        rotation_layer.bias.data = torch.zeros(input_dim, device=self.device, dtype=self.dtype)
        layers.append(rotation_layer)

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

        return Stacked_decoder, layers


    def Residual_Compressor(self, params):
        activation_function = self.activation_f #self.get_activation_f(params)
        init_dim = deepcopy(params['latent_dim'] * params['n_encoders'])
        input_dim = params['latent_dim'] * params['n_encoders']
        latent_dim = params['latent_dim']
        widths = halving_widths(input_dim, latent_dim)
        layers = []

        for output_dim in widths:
            layer = ResidualBlock(init_dim, input_dim, output_dim, activation_function, self.device)
            input_dim = output_dim
            layers.append(layer)


        final_layer = nn.Linear(init_dim + input_dim, latent_dim)
        nn.init.xavier_uniform(final_layer.weight)
        nn.init.constant_(final_layer.bias.data, 0)
        layers.append(final_layer)
        Compressor = nn.Sequential(*layers)
        return Compressor, layers


    def Residual_Decompressor(self, params):
        activation_function = self.activation_f #self.get_activation_f(params)
        input_dim = params['latent_dim']
        init_dim = deepcopy(input_dim)
        final_dim = params['latent_dim'] * params['n_encoders']
        widths = halving_widths(final_dim, input_dim)[::-1]
        layers = []

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


    def expand(self, x):
        n = self.params['n_encoders']
        x_stack = torch.concat([x for i in range(n)], dim = 1)
        return x_stack


    def collapse(self,x_stack):

        N,m = x_stack.shape
        n = self.params['n_encoders']
        x_stack = x_stack.reshape((N,n,m//n))
        x = torch.mean(x_stack, dim = 1)
        return x


    def forward(self, x):
        x_stack = self.expand(x)
        stacked_encoder =  self.params['stacked_encoder']
        stacked_decoder = self.params['stacked_decoder']

        x_encode = stacked_encoder(x_stack)
        x_comp = self.compressor(double(x_encode))

        #x_comp_decode = self.decoder(x_comp)
        x_decomp = self.decompressor(double(x_comp))
        x_decomp_decode = stacked_decoder(x_decomp)
        x_decode = stacked_decoder(x_encode)

        return x_decomp, x_encode, x_decode, x_decomp_decode, x_stack


    def decode_loss(self, x, x_pred):
        criterion = nn.MSELoss()
        loss = self.params['loss_weight_decoder'] * criterion(x, x_pred)
        return loss


    def Loss(self, x, dx):
        x_decomp, x_encode, x_decode, x_decomp_decode, x_stack = self.forward(x)
        #decompressor_loss = self.decode_loss(x_encode, x_decomp)

        decompressor_loss = self.decode_loss(x_decomp,  x_encode)
        s2_decoder_loss = self.decode_loss(x_decomp_decode,  x_stack)
        s1_decoder_loss = self.decode_loss(self.collapse(x_decomp_decode), x)
        sindy_z_loss = 0 * decompressor_loss
        sindy_x_loss = 0 * decompressor_loss
        reg_loss = 0 * decompressor_loss
        loss = 0 * decompressor_loss + s2_decoder_loss  + s1_decoder_loss #+ sindy_z_loss + sindy_x_loss + reg_loss

        loss_dict = {'decoder': decompressor_loss , 'sindy_x': s2_decoder_loss, 'sindy_z': s1_decoder_loss, 'reg': reg_loss}
        return loss, loss_dict





#0.00139449, Sindy_x: 0.000346457