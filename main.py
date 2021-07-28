import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

from modeling import network
from preprocessing import build_data

# Mipego
from mipego import mipego
from mipego.Surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

from objective import obj_function


def main():

    # hyperparameter configuration
    max_time = OrdinalSpace([10, 100], 'max_time')  # maximum lookback
    lr_rate = ContinuousSpace([1e-4, 1.0e-0], 'lr')  # learning rate
    num_rec = OrdinalSpace([2, 10], 'num_rec')  # maximum number of recurrent layers

    activations = ["tanh", "sigmoid"]  # activations of recurrent layers
    final_activations = ['softplus', 'exp']  # output activations
    neurons = OrdinalSpace([50, 200], 'neuron') * num_rec._ub[0]  # number of neurons
    acts = NominalSpace(activations, 'activation') * num_rec._ub[0]  # activations of recurrent layers
    dropout = ContinuousSpace([1e-5, .9], 'dropout') * num_rec._ub[0]  # normal dropout
    rec_dropout = ContinuousSpace([1e-5, .9], 'recurrent_dropout') * num_rec._ub[0]  # recurrent dropout
    f_acts = NominalSpace(final_activations, 'final_activation') * 2  # final activations. The "2" because we have 2
                                                                      # outputs

    search_space = num_rec * max_time * neurons * acts * dropout * rec_dropout * f_acts * lr_rate

    values = search_space.sampling(1)
    names = search_space.var_name
    cfg = {}
    for i in range(len(names)):
        cfg[names[i]] = values[0][i]

    net_cfg = cfg

    return net_cfg


if __name__ == '__main__':
    net_cfg = main()
    obj_function(net_cfg)
