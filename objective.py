# sklearn
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np

# Various
import pandas as pd
import json
import sys

# tensorflow
import tensorflow as tf
tf.random.set_seed(42)

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# sklearn
from sklearn import pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as k

# utilities
from data import load_data
from modeling import network
from preprocessing import build_data

def weibull_mean(alpha, beta):
    return alpha * math.gamma(1 + 1/beta)

model = None
def obj_function(net_cfg, cfg=None):

    if cfg==None:
        cfg = {'cv': 10, 'shuffle': True,
        'random_state': 21,
        'mask_value': -99,
        'reps': 30,
        'epochs': 50,
        'batches': 64}

    # deleting model if it exists
    try:
        del model
    except NameError:
        pass

    k.clear_session()

    train_x_orig, feature_cols, vld_trunc,  vld_x_orig, original_len, test_x_orig, test_y_orig = load_data()
    print(train_x_orig.shape)
    print(vld_trunc.shape)
    print(original_len.shape)
    # print(original_len)
    # print(vld_trunc.unit_number.unique())

    rmse_train = []
    r2_train = []
    mae_train = []
    std_train = []

    rmse_test = []
    r2_test = []
    mae_test = []
    std_test = []

    train_all = []
    test_all = []

    file = 'results_no_cv_HO_29_10'
    columns = ['fold', 'rmse_train', 'mae_train', 'r2_train','std_train', 'rmse_test', 'mae_test', 'r2_test', 'std_test', 'net_cfg']
    results = pd.DataFrame(columns=columns)
    start = time.time()

    # Pre-processing data
    scaler = pipeline.Pipeline(steps=[
        ('minmax', MinMaxScaler(feature_range=(-1, 1))),
        ('remove_constant', VarianceThreshold())])

    train = train_x_orig.copy()
    train = np.concatenate([train[['unit_number', 'time']], scaler.fit_transform(train[feature_cols])], axis=1)

    vld = vld_trunc.copy()
    vld = np.concatenate([vld[['unit_number', 'time']], scaler.transform(vld[feature_cols])], axis=1)

    # Preparing data for the RNN (numpy arrays)
    train_x, train_y = build_data(units=train[:, 0], time=train[:, 1], x=train[:, 2:], max_time=net_cfg['max_time'],
                                    is_test=False, mask_value=cfg['mask_value'],
                                    original_data=None, net_cfg = net_cfg, label=net_cfg['rul_style'])

    test_x, test_y = build_data(units=vld[:, 0], time=vld[:, 1], x=vld[:, 2:], max_time=net_cfg['max_time'],
                                is_test=True, mask_value=cfg['mask_value'],
                                original_data=original_len, net_cfg = net_cfg, label=net_cfg['rul_style'])

    # print(test_y)
    # only for debugging
    print('train_x', train_x.shape, 'train_y', train_y.shape, 'test_x', test_x.shape, 'test_y', test_y.shape)

    # training
    model, history = network(train_x, train_y, test_x, test_y, net_cfg, cfg)

    # For debugging
    # plt.plot(history.history['loss'], label='training')
    # plt.plot(history.history['val_loss'], label='validation')
    # plt.title('loss')
    # plt.legend()

    # predicting the rul on the train fold
    train_predict_1 = []
    train_predict_2 = []

    success = True
    try:
        for i in range(cfg['reps']):
            tf.random.set_seed(i)
            train_predict = model(train_x, training=True).numpy()
            train_predict_1.append(train_predict[:, 0].reshape(train_predict[:, 0].shape[0], 1))
            train_predict_2.append(train_predict[:, 1].reshape(train_predict[:, 1].shape[0], 1))

        train_predict_1_mean = np.mean(np.hstack(train_predict_1), axis=1)
        train_predict_2_mean = np.mean(np.hstack(train_predict_2), axis=1)
        train_predict_1_mean = train_predict_1_mean.reshape(train_predict_1_mean.shape[0], 1)
        train_predict_2_mean = train_predict_2_mean.reshape(train_predict_2_mean.shape[0], 1)
        train_predict_1_std = np.std(np.hstack(train_predict_1), axis=1)
        train_predict_2_std = np.std(np.hstack(train_predict_2), axis=1)
        train_predict_1_std = train_predict_1_std.reshape(train_predict_1_std.shape[0], 1)
        train_predict_2_std = train_predict_2_std.reshape(train_predict_2_std.shape[0], 1)

        train_predict = np.hstack([train_predict_1_mean, train_predict_2_mean,
                                    train_predict_1_std, train_predict_2_std])

        train_predict = np.resize(train_predict, (train_x.shape[0], 4))  # changed from 2 to 4
        train_result = np.concatenate((train_y, train_predict), axis=1)
        train_results_df = pd.DataFrame(train_result, columns=['T', 'mean_alpha', 'mean_beta', 'std_alpha',
                                                                'std_beta'])  # (add 'E' for event)
        train_results_df['unit_number'] = train_x_orig['unit_number'].to_numpy()
        train_results_df['time'] = train_x_orig['time'].to_numpy()

        train_results_df['predicted_mu'] = train_results_df[['mean_alpha', 'mean_beta']].apply(
            lambda row: weibull_mean(row[0], row[1]), axis=1)
        train_results_df['predicted_std+'] = train_results_df[['mean_alpha', 'mean_beta', 'std_alpha',
                                                                'std_beta']].apply(
            lambda row: weibull_mean(row[0] + 1.96 * row[2] / np.sqrt(cfg['reps']),
                                        row[1] + 1.96 * row[3] / np.sqrt(cfg['reps'])), axis=1)
        train_results_df['predicted_std-'] = train_results_df[['mean_alpha', 'mean_beta', 'std_alpha',
                                                                'std_beta']].apply(
            lambda row: weibull_mean(row[0] - 1.96 * row[2] / np.sqrt(cfg['reps']),
                                        row[1] - 1.96 * row[3] / np.sqrt(cfg['reps'])), axis=1)
        # predicting the rul on the test fold
        test_predict_1 = []
        test_predict_2 = []
        for i in range(cfg['reps']):
            tf.random.set_seed(i)
            test_predict = model(test_x, training=True).numpy()
            test_predict_1.append(test_predict[:, 0].reshape(test_predict[:, 0].shape[0], 1))
            test_predict_2.append(test_predict[:, 1].reshape(test_predict[:, 1].shape[0], 1))

        test_predict_1_mean = np.mean(np.hstack(test_predict_1), axis=1)
        test_predict_2_mean = np.mean(np.hstack(test_predict_2), axis=1)
        test_predict_1_mean = test_predict_1_mean.reshape(test_predict_1_mean.shape[0], 1)
        test_predict_2_mean = test_predict_2_mean.reshape(test_predict_2_mean.shape[0], 1)
        test_predict_1_std = np.std(np.hstack(test_predict_1), axis=1)
        test_predict_2_std = np.std(np.hstack(test_predict_2), axis=1)
        test_predict_1_std = test_predict_1_std.reshape(test_predict_1_std.shape[0], 1)
        test_predict_2_std = test_predict_2_std.reshape(test_predict_2_std.shape[0], 1)

        test_predict = np.hstack([test_predict_1_mean, test_predict_2_mean,
                                    test_predict_1_std, test_predict_2_std])

        test_predict = np.resize(test_predict, (test_x.shape[0], 4))  # changed from 2 to 4
        test_result = np.concatenate((test_y, test_predict), axis=1)
        test_results_df = pd.DataFrame(test_result, columns=['T', 'mean_alpha', 'mean_beta', 'std_alpha',
                                                                'std_beta'])  # (add 'E' for event)

        test_results_df['predicted_mu'] = test_results_df[['mean_alpha', 'mean_beta']].apply(
            lambda row: weibull_mean(row[0], row[1]), axis=1)
        test_results_df['predicted_std+'] = test_results_df[['mean_alpha', 'mean_beta', 'std_alpha',
                                                                'std_beta']].apply(
            lambda row: weibull_mean(row[0] + 1.96 * row[2] / np.sqrt(cfg['reps']),
                                        row[1] + 1.96 * row[3] / np.sqrt(cfg['reps'])), axis=1)
        test_results_df['predicted_std-'] = test_results_df[['mean_alpha', 'mean_beta', 'std_alpha',
                                                                'std_beta']].apply(
            lambda row: weibull_mean(row[0] - 1.96 * row[2] / np.sqrt(cfg['reps']),
                                        row[1] - 1.96 * row[3] / np.sqrt(cfg['reps'])), axis=1)
        # General administration

        train_all.append(train_results_df)
        test_all.append(test_results_df)

        # Performance evaluation
        # train:
        rmse_train.append(np.sqrt(mean_squared_error(train_results_df['predicted_mu'], train_results_df['T'])))
        mae_train.append((mean_absolute_error(train_results_df['predicted_mu'], train_results_df['T'])))
        r2_train.append(r2_score(train_results_df['predicted_mu'], train_results_df['T']))
        std_train.append((train_results_df['std_alpha'].mean() + train_results_df['std_beta'].mean())/2)

        # test:
        rmse_test.append(np.sqrt(mean_squared_error(test_results_df['predicted_mu'], test_results_df['T'])))
        mae_test.append((mean_absolute_error(test_results_df['predicted_mu'], test_results_df['T'])))
        r2_test.append(r2_score(test_results_df['predicted_mu'], test_results_df['T']))
        std_test.append((test_results_df['std_alpha'].mean() + test_results_df['std_beta'].mean())/2)
    except:
        success = False

    k.clear_session()
    # del model

    if (success == False):
        print('Failed')
        # return 1e10, 1e10, False  # not successful

    # registering results
    # results['fold'] = np.arange(cfg['cv'])
    results['rmse_train'] = rmse_train
    results['mae_train'] = mae_train
    results['r2_train'] = r2_train
    results['std_train'] = std_train
    results['rmse_test'] = rmse_test
    results['mae_test'] = mae_test
    results['r2_test'] = r2_test
    results['std_test'] = std_test
    results['net_cfg'] = json.dumps(net_cfg)

    # print(results)

    # return model, train_results_df, test_results_df, test_x_orig, test_y_orig, scaler, train_x, test_x
    
    if os.path.isfile(file):
        results.to_csv('./' + file, mode='a', index=False, header=False)
    else:
        results.to_csv('./' + file, mode='w', index=False, header=True)

    if (np.isfinite(results['rmse_test'].mean()) and np.isfinite(results['std_test'].mean())):
        return results['rmse_test'].mean(), results['std_test'].mean(), True
    else:
        return 1e10, 1e10, False #not successful
    # end = time.time()
    # print(f'Elapsed time: {(end - start) / 60} minutes')


#system arguments (configuration)
if len(sys.argv) > 2 and sys.argv[1] == '--cfg':
    cfg = eval(sys.argv[2])
    if len(sys.argv) > 3:
        gpu = sys.argv[3]
        
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    print(obj_function(cfg, None))
    k.clear_session()
