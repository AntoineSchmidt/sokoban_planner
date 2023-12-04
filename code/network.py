import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Multiply, LeakyReLU, PReLU, Concatenate
from keras.utils.vis_utils import plot_model

from config import *


def create(model=None, shape_in=(19, 19, 4), shape_out=4, activation_out='softmax', loss=None, lr=1e-3, prelu=False, skip=False):
    if model is None:
        if shape_out == 1: model = Config.net_1
        if shape_out == 4: model = Config.net_4

    # input layer
    data_in = Input(shape=shape_in)
    layer = data_in
    layer_conv = [data_in]

    # conv layer
    for nodes in model[0]:
        layer = Conv2D(nodes[0], kernel_size=nodes[1], padding='same')(layer)

        # activation
        if prelu: layer = PReLU(shared_axes=[1, 2])(layer)
        else: layer = LeakyReLU(alpha=0.3)(layer)

        layer_conv.append(layer)

        # pooling
        if len(nodes) > 2 and nodes[2] > 1:
            for i in range(1, 3): layer_conv[-i] = MaxPooling2D(nodes[2], padding='same')(layer_conv[-i])

        # skip connection
        if skip: layer = Concatenate(axis=-1)(layer_conv[-2:])
        else: layer = layer_conv[-1]


    layer = Flatten()(layer_conv[-1])

    # dense layer
    for nodes in model[1]:
        layer = Dense(nodes)(layer)
        
        # activation
        if prelu: layer = PReLU()(layer)
        else: layer = LeakyReLU(alpha=0.3)(layer)

    # ouput layer
    data_out = Dense(shape_out, activation=activation_out)(layer)

    # mask for reinforcement learning loss
    mask = Input(shape=(shape_out,))
    data_out_masked = Multiply()([data_out, mask])

    # compile model
    model = Model(data_in, data_out)
    model_masked = Model([data_in, mask], [data_out_masked])

    optimizer = Adam(lr=lr)
    if activation_out == 'softmax':
        if loss is None:
            loss = 'categorical_crossentropy'
            metrics = ['categorical_accuracy']
        else: # ppo
            optimizer = SGD(lr=lr)
            loss = loss(mask)
            metrics = None

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model_masked.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    else:
        if loss is None: loss = 'mse'
        elif loss == 'huber': loss = tf.keras.losses.Huber()
        elif loss == 'custom':
            # loss function enforcing admissible length estimates
            def loss(y_true, y_pred):
                error = y_pred - y_true
                return K.mean(K.abs(error) + 0.3 * error)

        model.compile(optimizer=optimizer, loss=loss)
        model_masked.compile(optimizer=optimizer, loss=loss)

    return model, model_masked


# install GraphViz (https://www.graphviz.org/) and ensure that its executables are in the $PATH
def plot(model, file):
    plot_model(model, to_file=file, show_shapes=True, show_layer_names=False)


def approximate(src, tar, tau=0.01):
    tar.set_weights(tau * np.array(src.get_weights()) + (1. - tau) * np.array(tar.get_weights()))