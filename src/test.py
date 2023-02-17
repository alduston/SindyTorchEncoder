import sys
sys.path.append("../src")
sys.path.append("../tf_model/src")
sys.path.append("../examples/lorenz")

import os
import datetime
#import pandas as pd
import numpy as np
from example_lorenz import get_lorenz_data
import torch
from sindy_utils import library_size
#from tf_training import train_network
import torch_training
from torch_autoencoder import SindyNet
#import tensorflow as tf
import pickle
import warnings
from data_utils import get_test_params,get_loader
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

#data_path = '../examples/lorenz/'
#data_path = '../examples/pendulum/'
#data_path = '../examples/rd/'

#save_name = 'model1'
#save_name = 'model2'

#params = pickle.load(open(data_path + save_name + '_params.pkl', 'rb'))
#params['save_name'] = data_path + save_name


#x = torch.rand(params['input_dim'])
#dx = torch.rand(params['input_dim'])
print_freq = 10


def run():
    model_params,training_data, validation_data = get_test_params(max_data = 500)
    train_params = {'bag_epochs': 100, 'pretrain_epochs': 1400, 'nbags': 50, 'bag_size':7,
                    'subtrain_epochs': 40, 'bag_sub_epochs':20, 'bag_learning_rate':.01}
    model_params['batch_size'] = 7
    model_params['threshold_frequency'] = 25
    model_params['sequential_thresholding'] = True
    if torch.cuda.is_available():
        l = len(training_data['x'])
        model_params, training_data, validation_data = get_test_params(max_data =5000)
        model_params['sequential_thresholding'] = False
        train_params = {'bag_epochs': 100, 'pretrain_epochs': 200, 'nbags': l//100, 'bag_size': 100,
                        'subtrain_epochs': 60, 'bag_sub_epochs':30, 'bag_learning_rate':.01}
        model_params['batch_size'] = 2000
        model_params['threshold_frequency'] = 25
    torch_training.train_sindy(model_params, train_params, training_data, validation_data)


    #train_loader = get_loader(training_data, params, device = device)
    #test_loader = get_loader(validation_data, params, device = device)

    #net = SindyNet(params).to(device)
    #optimizer = torch.optim.Adam(net.parameters(), lr = params['learning_rate'])

    #for epoch in range(params['max_epochs']):
        #total_loss, total_loss_dict = torch_training.train_one_epoch(net, train_loader, optimizer)
        #if not epoch % print_freq:
            #pass
            #print([f'Epoch: {epoch}, Active coeffs: {net.num_active_coeffs}'] + [f'{key}: {val.cpu().detach().numpy()} \n' for (key,val) in total_loss_dict.items()])

    #x = training_data['x'][:2]
    #z = net.forward(torch.tensor(x,dtype = torch.float32, device = device))[1]
    #Z_sim = [z]
    #for i in range(2, len(training_data['x'])//50):
        #dz_predict = net.sindy_predict(Z_sim[-1])
        #v = Z_sim[-1] + dz_predict * delta_t
        #Z_sim.append(v)

    #Z_real = []
    #for x in training_data['x'][:len(Z_sim)]:
        #x_decode, z = net.forward(torch.tensor(x,dtype = torch.float32, device = device))
        #Z_real.append(z)

    #Z_sim_cords = [z.cpu().detach().numpy()[0][0] for z in Z_sim]
    #Z_cords = [z.cpu().detach().numpy()[0] for z in Z_real]


    #print(Z_sim_cords[:5])
    #print(Z_cords[:5])
    #plt.plot(Z_sim_cords, color='blue')
    #plt.plot(Z_cords, color='red')

    #plt.savefig('fig.png')
   # plt.show()



if __name__=='__main__':
    run()
