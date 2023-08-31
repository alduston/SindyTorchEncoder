import torch
import torch.nn as nn
import numpy as np


def get_network(input_dim, widths, final_dim, activation):
    layers = []
    for output_dim in widths:
        layer = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform(layer.weight)
        nn.init.constant_(layer.bias.data, 0)

        input_dim = output_dim
        layers.append(layer)
        layers.append(activation)

    layer = nn.Linear(input_dim, final_dim)
    nn.init.xavier_uniform(layer.weight)
    nn.init.constant_(layer.bias.data, 0)
    layers.append(layer)
    net = nn.Sequential(*layers)
    return net, layers


class Additive_flow(nn.Module):
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
        self.activation_f = params['activation_f']
        self.idx_range = list(range(self.params['latent_dim'])) + [0]

        self.permute_matrixes = self.get_permute_matrixes()
        self.t_blocks = self.init_t_blocks()
        self.s = nn.Parameter(self.init_s(), requires_grad=True)



    def init_s(self):
        s = torch.randn(self.params['latent_dim'], device=self.device, dtype=self.dtype)
        return s


    def t_block(self):
        activation = self.activation_f
        n = self.params['latent_dim']
        input_dim = n-1
        widths = self.params['flow_widths']
        final_dim = 1
        return get_network(input_dim, widths, final_dim, activation)


    def init_t_blocks(self):
        t_blocks = []
        index_range = self.idx_range
        n = self.params['latent_dim']
        for i,idx in enumerate(index_range):
            t_block, t_block_layers = self.t_block()
            for j,block_param in enumerate(t_block.parameters()):
                self.register_parameter(name=f't_{i//n}_{idx}_{j}', param=block_param)
            t_blocks.append(t_block)
        return t_blocks


    def get_permute_matrixes(self):
        index_range = self.idx_range
        permute_matrixes = []
        for idx in index_range:
            permute_m = torch.eye(self.params['latent_dim'], device = self.device)
            permute_m[[-1,idx]] = permute_m[[idx,-1]]
            permute_matrixes.append(permute_m)
        return permute_matrixes


    def forward_flow(self,x):
        z = x
        permute_matrixes = self.permute_matrixes
        t_blocks = self.t_blocks
        for Pi, ti in zip(permute_matrixes, t_blocks):
            z = (Pi @ z.T).T
            z_1 = z[:, :-1]
            z_2 = z[:, -1]


            z_1p = z_1
            z_2p = z_2.reshape(ti(z_1).shape) + ti(z_1)

            zp = torch.concat((z_1p, z_2p), dim = 1).reshape(z.shape)
            z = (Pi.T @ zp.T).T
        return torch.exp(self.s) * z


    def backward_flow(self, z):
        x = torch.exp(-self.s) * z
        permute_matrixes = self.permute_matrixes[::-1]
        t_blocks = self.t_blocks[::-1]
        for Pi, ti in zip(permute_matrixes, t_blocks):
            x = (Pi.T @ x.T).T
            x_1 = x[:, :-1]
            x_2 = x[:, -1]

            x_1p = x_1
            x_2p = x_2.reshape(ti(x_1).shape) - ti(x_1)

            xp = torch.concat((x_1p, x_2p), dim = 1).reshape(x.shape)
            x = (Pi @ xp.T).T
        return x


def run():
    d = 3
    params = {'activation_f': torch.nn.Sigmoid(), 'latent_dim': d, 'flow_widths': [3,2] }
    x = torch.randn((1000,3), device = 'cpu', dtype = torch.float32)
    flow = Additive_flow(params)
    z = flow.forward_flow(x)
    x_p = flow.backward_flow(z)

if __name__=='__main__':
    run()




















