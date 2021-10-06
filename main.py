# import numpy as np
# import pandas as pd
# from sklearn import pipeline
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.preprocessing import MinMaxScaler

# from modeling import network
# from preprocessing import build_data

# Mipego
from mipego import mipego
from mipego.Surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

from objective import obj_function


def main():

    # hyperparameter configuration
    max_time = OrdinalSpace([50, 100], 'max_time')  # maximum lookback
    lr_rate = ContinuousSpace([1e-4, 1.0e-1], 'lr')  # learning rate
    num_rec = OrdinalSpace([2, 4], 'num_rec')  # maximum number of recurrent layers

    activations = ["tanh", "sigmoid"]  # activations of recurrent layers
    final_activations = ['softplus', 'exp']  # output activations
    neurons = OrdinalSpace([50, 100], 'neuron') * num_rec._ub[0]  # number of neurons
    acts = NominalSpace(activations, 'activation') * num_rec._ub[0]  # activations of recurrent layers
    dropout = ContinuousSpace([1e-5, .9], 'dropout') * num_rec._ub[0]  # normal dropout
    rec_dropout = ContinuousSpace([1e-5, .9], 'recurrent_dropout') * num_rec._ub[0]  # recurrent dropout
    f_acts = NominalSpace(final_activations, 'final_activation') * 2  # final activations. The "2" because we have 2
                                                                      # outputs
    percentage = OrdinalSpace([20, 75], 'percentage')
    rul = OrdinalSpace([110, 135], 'rul')

    rul_style = NominalSpace(['linear', 'nonlinear'], 'rul_style')

    search_space = num_rec * max_time * neurons * acts * dropout * rec_dropout * f_acts * percentage * rul * rul_style * lr_rate

    #values = search_space.sampling(1)
    #names = search_space.var_name
    #net_cfg = {}
    #for i in range(len(names)):
    #    net_cfg[names[i]] = values[0][i]

    # Uncomment for debugging purposes.
    net_cfg={'max_time': 100, 'lr': 0.01, 'num_rec': 3, 'neuron_0': 100, 'activation_0': 'tanh', 'dropout_0': 0.25, 'recurrent_dropout_0': 0.25, 
    'neuron_1': 50, 'activation_1': 'tanh', 'dropout_1': 0.25, 'recurrent_dropout_1': 0.25, 
    'neuron_2': 20, 'activation_2': 'tanh', 'dropout_2': 0.25, 'recurrent_dropout_2': 0.25, 
    'final_activation_0': 'exp', 'final_activation_1': 'softplus', 'percentage': 70, 'rul': 115, 'rul_style': 'nonlinear'}
    
    """
    self, search_space, obj_func, surrogate, second_surrogate=None, ftarget=None,
                 minimize=True, noisy=False, max_eval=None, 
                 infill='MGFI', t0=2, tf=1e-1, schedule=None,
                 n_init_sample=None, n_point=1, n_job=1, backend='multiprocessing',
                 n_restart=None, max_infill_eval=None, wait_iter=3, optimizer='MIES', 
                 log_file=None, data_file=None, verbose=False, random_seed=None,
                 available_gpus=[],bi_objective=False,
                 ref_time=3000.0, ref_loss=3.0, hvi_alpha=0.1, ignore_gpu=[],
                 **obj_func_params
    """

    print(search_space.levels)

    # model1 = RandomForest(levels=search_space.levels)
    # model2 = RandomForest(levels=search_space.levels)

    # #now define the optimizer.
    # opt = mipego(search_space, obj_function, model1, second_surrogate=model2,
    #                 minimize=True, max_eval=50, 
    #                 infill='HVI', n_init_sample=10, 
    #                 n_point=1, n_job=1, optimizer='MIES', 
    #                 verbose=False, random_seed=None)


    #run
    # opt.run()
    # incumbent, stop_dict = opt.run()
    # print(incumbent)
    print(net_cfg)
    obj_function(net_cfg, cfg=None)

    # return incumbent


if __name__ == '__main__':
    # General hyperparameters
    # cfg = {'cv': 2, 'shuffle': True,
    #    'random_state': 21,
    #    'mask_value': -99,
    #    'reps': 30,
    #    'epochs': 2,
    #    'batches': 64}

    # incumbent = main()
    main()
    #rmse, std =  obj_function(net_cfg, cfg)
    # print(incumbent)
