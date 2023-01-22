import torch
import torch.nn as nn
import warnings
from sindy_utils import z_derivative, z_derivative_order2,\
    get_initialized_weights, sindy_library_torch, sindy_library_torch_order2
import warnings
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

