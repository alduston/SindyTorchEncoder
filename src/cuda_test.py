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

def run():
    params,training_data, validation_data = get_test_params(train_size = 100)
    if torch.cuda.is_available():
        pass
    else:
        params['batch_size'] = 5
        params['threshold_frequency'] = 25
        params['max_epochs'] = 3000
    train_loader = get_loader(training_data, params)
    test_loader = get_loader(validation_data, params)

    net = SindyNet(params)
    coeffs = net.sindy_coeffs()
    coeffs.get_device()

    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)


if __name__=='__main__':
    run()