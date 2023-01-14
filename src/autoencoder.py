import sys
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")



def full_network_torch(params):
    """
    Define the full network architecture.

    Arguments:
        params - Dictionary object containing the parameters that specify the training.
        See README file for a description of the parameters.

    Returns:
        network - Dictionary containing the tensorflow objects that make up the network.
    """
    input_dim = params['input_dim']
    latent_dim = params['latent_dim']
    activation = params['activation']
    poly_order = params['poly_order']
    if 'include_sine' in params.keys():
        include_sine = params['include_sine']
    else:
        include_sine = False
    library_dim = params['library_dim']
    model_order = params['model_order']

    network = {}

    x = torch.empty((0, input_dim),  dtype=torch.float32)
    dx = torch.empty((0, input_dim), dtype=torch.float32)

    if model_order == 2:
        ddx = torch.empty((0, input_dim), dtype=torch.float32)

    if activation == 'linear':
        z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases = linear_autoencoder(x, input_dim, latent_dim)
    else:
        z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases = nonlinear_autoencoder(x, input_dim, latent_dim, params['widths'], activation=activation)
    if model_order == 1:
        dz = z_derivative(x, dx, encoder_weights, encoder_biases, activation=activation)
        Theta = sindy_library_torch(z, latent_dim, poly_order, include_sine)
    else:
        dz, ddz = z_derivative_order2(x, dx, ddx, encoder_weights, encoder_biases, activation=activation)
        Theta = sindy_library_torch_order2(z, dz, latent_dim, poly_order, include_sine)

    if params['coefficient_initialization'] == 'xavier':
        intializer = torch.nn.init.xavier_uniform
        sindy_coefficients = get_initialized_weights([library_dim, latent_dim], intializer)

    elif params['coefficient_initialization'] == 'specified':
        pass
        #sindy_coefficients = tf.get_variable('sindy_coefficients', initializer=params['init_coefficients'])

    elif params['coefficient_initialization'] == 'constant':
        intializer = torch.nn.init.constant_
        init_param = [1.0]
        sindy_coefficients = get_initialized_weights([library_dim, latent_dim], intializer, init_param)

    elif params['coefficient_initialization'] == 'normal':
        intializer = torch.nn.init.normal_
        sindy_coefficients = get_initialized_weights([library_dim, latent_dim], intializer)

    if params['sequential_thresholding']:
        coefficient_mask = torch.empty((library_dim, latent_dim), dtype=torch.float32)
        sindy_predict = torch.matmul(Theta, coefficient_mask * sindy_coefficients)
        network['coefficient_mask'] = coefficient_mask

    else:
        sindy_predict = torch.matmul(Theta, sindy_coefficients)

    if model_order == 1:
        dx_decode = z_derivative(z, sindy_predict, decoder_weights, decoder_biases, activation=activation)
    else:
        dx_decode, ddx_decode = z_derivative_order2(z, dz, sindy_predict, decoder_weights, decoder_biases,
                                                    activation=activation)

    network['x'] = x
    network['dx'] = dx
    network['z'] = z
    network['dz'] = dz
    network['x_decode'] = x_decode
    network['dx_decode'] = dx_decode
    network['encoder_weights'] = encoder_weights
    network['encoder_biases'] = encoder_biases
    network['decoder_weights'] = decoder_weights
    network['decoder_biases'] = decoder_biases
    network['Theta'] = Theta
    network['sindy_coefficients'] = sindy_coefficients

    #print(f'x has shape {x.shape}')
    #print(f'dx has shape {dx.shape}')
    #print(f'z has shape {dx.shape}')
    #print(f'dz has shape {dx.shape}')

    #print(f'x_decode has shape {dx.shape}')
    #print(f'dx_decode has shape {dx_decode}')
    #print(f'encoder_weights has shape {len(encoder_weights)}x{encoder_weights[0].shape}')
    #print(f'decoder_weights has shape {len(decoder_weights)}x{decoder_weights[0].shape}')
    #print(f'decoder_biases has shape {len(decoder_biases)}x{decoder_biases[0].shape}')
    #print(f'decoder_biases has shape {len(decoder_biases)}x{decoder_biases[0].shape}')
    #print(f'Theta has shape {Theta.shape}')
    #print(f'Sindy_coefficients have shape {sindy_coefficients.shape}')


    if model_order == 1:
        network['dz_predict'] = sindy_predict
    else:
        network['ddz'] = ddz
        network['ddz_predict'] = sindy_predict
        network['ddx'] = ddx
        network['ddx_decode'] = ddx_decode

    return network



def get_initialized_weights(shape, initializer, init_param=None):
    W = torch.nn.Linear(shape[0], shape[1])
    if init_param:
        initializer(W.weight, *init_param)
    else:
        initializer(W.weight)
    return torch.transpose(W.state_dict()['weight'],0,1)


def define_loss(network, params):
    """
    Create the loss functions.

    Arguments:
        network - Dictionary object containing the elements of the network architecture.
        This will be the output of the full_network() function.
    """
    x = network['x']
    x_decode = network['x_decode']
    if params['model_order'] == 1:
        dz = network['dz']
        dz_predict = network['dz_predict']
        dx = network['dx']
        dx_decode = network['dx_decode']
    else:
        ddz = network['ddz']
        ddz_predict = network['ddz_predict']
        ddx = network['ddx']
        ddx_decode = network['ddx_decode']
    sindy_coefficients = torch.tensor(params['coefficient_mask'])*network['sindy_coefficients']

    losses = {}
    losses['decoder'] = torch.mean((x - x_decode)**2)
    if params['model_order'] == 1:
        losses['sindy_z'] = torch.mean((dz - dz_predict)**2)
        losses['sindy_x'] = torch.mean((dx - dx_decode)**2)
    else:
        losses['sindy_z'] = torch.mean((ddz - ddz_predict)**2)
        losses['sindy_x'] = torch.mean((ddx - ddx_decode)**2)
    losses['sindy_regularization'] = torch.mean(torch.abs(sindy_coefficients))
    loss = params['loss_weight_decoder'] * losses['decoder'] \
           + params['loss_weight_sindy_z'] * losses['sindy_z'] \
           + params['loss_weight_sindy_x'] * losses['sindy_x'] \
           + params['loss_weight_sindy_regularization'] * losses['sindy_regularization']

    loss_refinement = params['loss_weight_decoder'] * losses['decoder'] \
                      + params['loss_weight_sindy_z'] * losses['sindy_z'] \
                      + params['loss_weight_sindy_x'] * losses['sindy_x']

    return loss, losses, loss_refinement


def linear_autoencoder(x, input_dim, latent_dim):
    z,encoder_weights,encoder_biases = build_network_layers(x, input_dim, latent_dim, [], None, 'encoder')
    x_decode,decoder_weights,decoder_biases = build_network_layers(z, latent_dim, input_dim, [], None, 'decoder')

    return z, x_decode, encoder_weights, encoder_biases,decoder_weights,decoder_biases


def nonlinear_autoencoder(x, input_dim, latent_dim, widths, activation='elu'):
    """
    Construct a nonlinear autoencoder.

    Arguments:

    Returns:
        z -
        x_decode -
        encoder_weights - List of tensorflow arrays containing the encoder weights
        encoder_biases - List of tensorflow arrays containing the encoder biases
        decoder_weights - List of tensorflow arrays containing the decoder weights
        decoder_biases - List of tensorflow arrays containing the decoder biases
    """

    if activation == 'relu':
        activation_function = torch.nn.ReLU()
    elif activation == 'elu':
        activation_function = torch.nn.ELU()
    elif activation == 'sigmoid':
        activation_function = torch.nn.Sigmoid()

    z,encoder_weights,encoder_biases = build_network_layers(x, input_dim, latent_dim, widths, activation_function, 'encoder')
    x_decode,decoder_weights,decoder_biases = build_network_layers(z, latent_dim, input_dim, widths[::-1], activation_function, 'decoder')

    return z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases


def build_network_layers(input, input_dim, output_dim, widths, activation, name):
    """
    Construct one portion of the network (either encoder or decoder).

    Arguments:
        input - 2D tensorflow array, input to the network (shape is [?,input_dim])
        input_dim - Integer, number of state variables in the input to the first layer
        output_dim - Integer, number of state variables to output from the final layer
        widths - List of integers representing how many units are in each network layer
        activation - Tensorflow function to be used as the activation function at each layer
        name - String, prefix to be used in naming the tensorflow variables

    Returns:
        input - Tensorflow array, output of the network layers (shape is [?,output_dim])
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
    """
    weights = []
    biases = []
    last_width=input_dim
    for i,n_units in enumerate(widths):
        W = torch.nn.Linear(last_width, n_units)
        torch.nn.init.xavier_uniform(W.weight)
        torch.nn.init.constant_(W.bias.data, 0)
        input = W(input)

        w = torch.transpose(W.state_dict()['weight'],0,1)
        b = W.state_dict()['bias']

        if activation is not None:
            input = activation(input)
        last_width = n_units
        weights.append(w)
        biases.append(b)

    W = torch.nn.Linear(last_width, output_dim)
    torch.nn.init.xavier_uniform(W.weight)
    torch.nn.init.constant_(W.bias.data, 0)
    input = W(input)

    w = torch.transpose(W.state_dict()['weight'], 0, 1)
    b = W.state_dict()['bias']
    weights.append(w)
    biases.append(b)

    return input, weights, biases


def sindy_library_torch(z, latent_dim, poly_order, include_sine=False):
    if len(z.shape) == 1:
        z = z.view(len(z),1)
    """
    Build the SINDy library.

    Arguments:
        z - 2D tensorflow array of the snapshots on which to build the library. Shape is number of
        time points by the number of state variables.
        latent_dim - Integer, number of state variable in z.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.

    Returns:
        2D tensorflow array containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """
    library = [torch.ones(z.shape[0])]

    for i in range(latent_dim):
        library.append(z[:,i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(torch.multiply(z[:,i], z[:,j]))

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i]*z[:,j]*z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q])

    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:,i]))

    return torch.stack(library, axis=1)


def sindy_library_torch_order2(z, dz, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library for a second order system. This is essentially the same as for a first
    order system, but library terms are also built for the derivatives.
    """
    library = [torch.ones(z.shape[0])]

    z_combined = torch.concat([z, dz], 1)

    for i in range(2*latent_dim):
        library.append(z_combined[:,i])

    if poly_order > 1:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                library.append(torch.multiply(z_combined[:,i], z_combined[:,j]))

    if poly_order > 2:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k])

    if poly_order > 3:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    for p in range(k,2*latent_dim):
                        library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p])

    if poly_order > 4:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    for p in range(k,2*latent_dim):
                        for q in range(p,2*latent_dim):
                            library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p]*z_combined[:,q])

    if include_sine:
        for i in range(2*latent_dim):
            library.append(torch.sin(z_combined[:,i]))

    return torch.stack(library, axis=1)


def z_derivative(input, dx, weights, biases, activation='elu'):
    input = torch.transpose(input, 0, 1)
    dx = torch.transpose(dx, 0, 1)
    """
    Compute the first order time derivatives by propagating through the network.

    Arguments:
        input - 2D tensorflow array, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.

    Returns:
        dz - Tensorflow array, first order time derivatives of the network output.
    """
    dz = dx
    biases = [bias.view(len(bias),1) for bias in biases]
    if activation == 'elu':
        for i in range(len(weights)-1):
            input = torch.add(torch.matmul(weights[i], input), biases[i])
            dz = torch.multiply(torch.minimum(torch.exp(input),torch.full(input.shape,1.0)),
                                  torch.matmul(dz, weights[i]))
            input = torch.nn.ELU()(input)

        dz = torch.matmul(dz, weights[-1])
    elif activation == 'relu':
        for i in range(len(weights)-1):
            input = torch.add(torch.matmul(weights[i], input), biases[i])
            bool_tensor = (input>0)
            bool_tensor.float()
            dz = torch.multiply(bool_tensor, torch.matmul(dz, weights[i]))
            input = torch.nn.ReLU()(input)
        dz = torch.matmul(dz, weights[-1])
    elif activation == 'sigmoid':
        for i in range(len(weights)-1):
            input = torch.add(torch.matmul(weights[i], input), biases[i])
            input = torch.nn.Sigmoid()(input)
            dz = torch.multiply(torch.multiply(input, 1-input), torch.matmul( weights[i], dz))
        dz = torch.matmul( weights[-1], dz)
    else:
        for i in range(len(weights)-1):
            dz = torch.matmul(weights[-1], dz)
        dz = torch.matmul(weights[-1], dz)
    return torch.mean(dz)


def z_derivative_order2(input, dx, ddx, weights, biases, activation='elu'):
    """
    Compute the first and second order time derivatives by propagating through the network.

    Arguments:
        input - 2D tensorflow array, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        ddx - Second order time derivatives of the input to the network.
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.

    Returns:
        dz - Tensorflow array, first order time derivatives of the network output.
        ddz - Tensorflow array, second order time derivatives of the network output.
    """
    dz = dx
    ddz = ddx
    if activation == 'elu':
        for i in range(len(weights)-1):
            input = torch.matmul(input, weights[i]) + biases[i]
            dz_prev = torch.matmul(dz, weights[i])
            elu_derivative = torch.minimum(torch.exp(input),torch.full(input.shape,1.0))

            bool_tensor = (input > 0)
            bool_tensor.float()
            elu_derivative2 = torch.multiply(torch.exp(input), bool_tensor)
            dz = torch.multiply(elu_derivative, dz_prev)
            ddz = torch.multiply(elu_derivative2, torch.square(dz_prev)) \
                  + torch.multiply(elu_derivative, torch.matmul(ddz, weights[i]))
            input = torch.nn.ELU()(input)
        dz = torch.matmul(dz, weights[-1])
        ddz = torch.matmul(ddz, weights[-1])
    elif activation == 'relu':
        # NOTE: currently having trouble assessing accuracy of 2nd derivative due to discontinuity
        for i in range(len(weights)-1):
            input = torch.matmul(input, weights[i]) + biases[i]
            bool_tensor = (input > 0)
            bool_tensor.float()
            relu_derivative = bool_tensor
            dz = torch.multiply(relu_derivative, torch.matmul(dz, weights[i]))
            ddz = torch.multiply(relu_derivative, torch.matmul(ddz, weights[i]))
            input = torch.nn.ReLU()(input)
        dz = torch.matmul(dz, weights[-1])
        ddz = torch.matmul(ddz, weights[-1])
    elif activation == 'sigmoid':
        for i in range(len(weights)-1):
            input = torch.matmul(input, weights[i]) + biases[i]
            input = torch.nn.Sigmoid()(input)
            dz_prev = torch.matmul(dz, weights[i])
            sigmoid_derivative = torch.multiply(input, 1-input)
            sigmoid_derivative2 = torch.multiply(sigmoid_derivative, 1 - 2*input)
            dz = torch.multiply(sigmoid_derivative, dz_prev)
            ddz = torch.multiply(sigmoid_derivative2, torch.square(dz_prev)) \
                  + torch.multiply(sigmoid_derivative, torch.matmul(ddz, weights[i]))
        dz = torch.matmul(dz, weights[-1])
        ddz = torch.matmul(ddz, weights[-1])
    else:
        for i in range(len(weights)-1):
            dz = torch.matmul(dz, weights[i])
            ddz = torch.matmul(ddz, weights[i])
        dz = torch.matmul(dz, weights[-1])
        ddz = torch.matmul(ddz, weights[-1])
    return dz,ddz

