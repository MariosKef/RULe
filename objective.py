# sklearn
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
# Various
import pandas as pd
import json
# tensorflow
import tensorflow as tf
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

    if (cfg == None):
        cfg = {'cv': 10, 'shuffle': True,
       'random_state': 21,
       'mask_value': -99,
       'reps': 30,
       'epochs': 20,
       'batches': 64}

    # deleting model if it exists
    try:
        del model
    except NameError:
        pass

    k.clear_session()

    train_x_orig, feature_cols = load_data()
    print(train_x_orig.shape)

    kf = KFold(n_splits=cfg['cv'], shuffle=cfg['shuffle'], random_state=cfg['random_state'])

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

    file = 'results'
    columns = ['fold', 'rmse_train', 'mae_train', 'r2_train','std_train', 'rmse_test', 'mae_test', 'r2_test', 'std_test', 'net_cfg']
    results = pd.DataFrame(columns=columns)

    fold_count = 0

    start = time.time()

    for train_units, test_units in kf.split(train_x_orig.unit_number.unique()):

        fold_count += 1
        print(f'Fold: {fold_count}')
        tf.random.set_seed(fold_count)

        # Selecting data
        train_index = train_x_orig[train_x_orig.unit_number.isin(train_units)].index
        test_index = train_x_orig[train_x_orig.unit_number.isin(
            test_units)].index  # careful this was wrongly test_x_orig. It should be train_x_orig

        X_train = train_x_orig.iloc[train_index]
        X_test_or = train_x_orig.iloc[test_index]  # careful this was wrongly test_x_orig. It should be train_x_orig

        X_train.reset_index(drop=True, inplace=True)
        X_test = X_test_or.reset_index(drop=True, inplace=False)  # changed for debugging

        # Truncating test data randomly
        X_test_trunc = []
        test_y = []
        max_cycle = []
        test_index = []  # for debugging purposes
        temp_or_test_cycles = []
        for i in set(X_test.unit_number.unique()):
            np.random.seed(i)
            #         print(i)
            temp_df = X_test[X_test.unit_number == i]
            temp_df.reset_index(drop=True, inplace=True)  # important
            length = temp_df.shape[0]
            temp_or_test_cycles.append(length)
            level = np.random.choice(np.arange(5, 96), 1)[0]
            r = np.int(length * (1 - level / 100))
            test_index.append(X_test_or[X_test_or.unit_number == i].index.tolist()[
                              :r + 1])  # check this with train_x_orig instead of X_test_or (probably it's the same)
            temp_df = temp_df.truncate(after=r)
            max_cycle.append(temp_df.shape[0])
            X_test_trunc.append(temp_df)

        test_index = [item for sublist in test_index for item in sublist]

        X_test_trunc = pd.concat(X_test_trunc)
        X_test_trunc.reset_index(drop=True, inplace=True)

        # Pre-processing data
        scaler = pipeline.Pipeline(steps=[
            ('minmax', MinMaxScaler(feature_range=(-1, 1))),
            ('remove_constant', VarianceThreshold())])

        train = X_train.copy()
        train = np.concatenate([train[['unit_number', 'time']], scaler.fit_transform(train[feature_cols])], axis=1)

        test = X_test_trunc.copy()
        test = np.concatenate([test[['unit_number', 'time']], scaler.transform(test[feature_cols])], axis=1)

        # Preparing data for the RNN (numpy arrays)
        train_x, train_y = build_data(units=train[:, 0], time=train[:, 1], x=train[:, 2:], max_time=net_cfg['max_time'],
                                      is_test=False, mask_value=cfg['mask_value'],
                                      original_data=None, net_cfg = net_cfg, label=net_cfg['rul_style'])

        test_x, test_y = build_data(units=test[:, 0], time=test[:, 1], x=test[:, 2:], max_time=net_cfg['max_time'],
                                    is_test=True, mask_value=cfg['mask_value'],
                                    original_data=X_test_or, net_cfg = net_cfg, label=net_cfg['rul_style'])

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
        for i in range(cfg['reps']):
            tf.random.set_seed(i)
            train_predict = model(train_x, training=True).numpy()
            train_predict_1.append(train_predict[:, 0].reshape(train_predict[:, 0].shape[0], 1))
            train_predict_2.append(train_predict[:, 1].reshape(train_predict[:, 1].shape[0], 1))

        train_predict_1_mean = np.average(np.hstack(train_predict_1), axis=1)
        train_predict_2_mean = np.average(np.hstack(train_predict_2), axis=1)
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
        train_results_df['unit_number'] = train_x_orig.iloc[train_index]['unit_number'].to_numpy()
        train_results_df['time'] = train_x_orig.iloc[train_index]['time'].to_numpy()

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

        test_predict_1_mean = np.average(np.hstack(test_predict_1), axis=1)
        test_predict_2_mean = np.average(np.hstack(test_predict_2), axis=1)
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
        success = True
        try:
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
        del model

        if (success == False):
            return 0,0, False #not successful

        # registering results
    results['fold'] = np.arange(cfg['cv'])
    results['rmse_train'] = rmse_train
    results['mae_train'] = mae_train
    results['r2_train'] = r2_train
    results['std_train'] = std_train
    results['rmse_test'] = rmse_test
    results['mae_test'] = mae_test
    results['r2_test'] = r2_test
    results['std_test'] = std_test
    results['net_cfg'] = json.dumps(net_cfg)

    print(results)

    if os.path.isfile(file):
        results.to_csv('./' + file, mode='a', index=False, header=False)
    else:
        results.to_csv('./' + file, mode='w', index=False, header=True)

    if (np.isfinite(results['rmse_test'].mean()) and np.isfinite(results['std_test'].mean())):
        return results['rmse_test'].mean(), results['std_test'].mean(), True
    else:
        return 0,0, False #not successful
    # end = time.time()
    # print(f'Elapsed time: {(end - start) / 60} minutes')
