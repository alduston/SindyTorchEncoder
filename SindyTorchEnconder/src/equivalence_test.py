from torch_autoencoder import SindyNet
import torch
import sys
sys.path.append("../src")
sys.path.append("../tf_model/src/")
sys.path.append("../examples/lorenz/")
import tf_autoencoder
from tf_autoencoder import full_network, define_loss
from example_lorenz import get_lorenz_data
from test import get_params
import tensorflow as tf
import numpy as np

def run():
    params, training_data, validation_data = get_params()
    torch_net = SindyNet(params)
    batch_size = params['batch_size']

    x_batch = training_data['x'][:batch_size]
    dx_batch = training_data['dx'][:batch_size]
    ddx_batch = training_data['dx'][:batch_size]

    x_batch_torch = torch.tensor(x_batch, dtype=torch.float32)
    dx_batch_torch = torch.tensor(dx_batch, dtype=torch.float32)

    x_batch_flow = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    dx_batch_flow = tf.convert_to_tensor(dx_batch, dtype=tf.float32)


    encoder_weights, encoder_biases = torch_net.encoder_weights()
    encoder_weights = [tf.transpose(tf.convert_to_tensor(weight.detach().numpy())) for weight in encoder_weights]

    decoder_weights, decoder_biases = torch_net.decoder_weights()
    decoder_weights = [tf.transpose(tf.convert_to_tensor(weight.detach().numpy())) for weight in decoder_weights]

    tf_net = full_network(params, n = batch_size, encode_w = encoder_weights, decode_w = decoder_weights,
                          x = x_batch_flow, dx = dx_batch_flow)

    xp_torch,z_torch = torch_net.forward(x_batch_torch)
    dz_torch = torch_net.dz(x_batch_torch, dx_batch_torch)
    dzp_torch = torch.transpose(torch_net.sindy_predict(z_torch,x_batch_torch, dx_batch_torch),0,1)
    #torch_loss = torch.mean((dz_torch - dzp_torch)**2)

    loss_torch = torch_net.Loss(x_batch_torch, xp_torch, z_torch, dx_batch_torch)[0]
    #print(losses_torch)

    loss_tf = define_loss(tf_net, params)[0]
    #print('\n\n')
    #dzp_flow = tf_net['dz_predict']
    #dz_flow  = tf_net['dz']
    #tf_loss = tf.reduce_mean((dzp_flow - dz_flow)**2)

    print('Pytorch vals:')
    print(loss_torch)
    print('\n')
    #print(dz_torch.detach().numpy()[:3,:3])
    #print('\n\n')
    #print(dzp_torch.detach().numpy()[:3,:3])
    #print('\n\n')
    #print(torch_loss.detach().numpy())
    #print('\n\n')
    #print(dxd_torch.detach().numpy()[:3,:3])
    #print('\n\n')
    #print(dx_batch_torch.detach().numpy()[:3,:3])

    with tf.Session() as sess:
        print('Tensorflow vals:')
        sess.run(tf.global_variables_initializer())

        #print(dz_flow.eval()[:3,:3])
        #print('\n\n')
        #print(dzp_flow.eval()[:3, :3])
        #print('\n\n')
        #print(tf_loss.eval())
        #print('\n\n')
        #print(dxd_flow.eval()[:3,:3])
        #print('\n\n')
        #print(dx_flow.eval()[:3,:3])

        #losses_tf = {key:val.eval() for (key,val) in losses_tf.items()}
        #print(losses_tf)

        #print(params['loss_weight_sindy_z'])





if __name__ == '__main__':
    run()

