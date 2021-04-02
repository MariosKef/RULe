import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

from modeling import network
from preprocessing import build_data

"""
The below is specific to the CMAPSS data
"""


def main():
    # Loading and preparing data
    print('Loading and preparing Data')
    id_col = 'unit_number'
    time_col = 'time'
    feature_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3'] + ['sensor_measurement_{}'.format(x) for x
                                                                       in range(1, 22)]
    column_names = [id_col, time_col] + feature_cols

    train_x_orig = pd.read_csv('https://raw.githubusercontent.com/daynebatten/keras-wtte-rnn/master/train.csv',
                               header=None, names=column_names)
    test_x_orig = pd.read_csv('https://raw.githubusercontent.com/daynebatten/keras-wtte-rnn/master/test_x.csv',
                              header=None, names=column_names)
    test_y_orig = pd.read_csv('https://raw.githubusercontent.com/daynebatten/keras-wtte-rnn/master/test_y.csv',
                              header=None, names=['T'])

    # Pre-processing data
    print('Data Preprocessing')
    scaler = pipeline.Pipeline(steps=[('minmax', MinMaxScaler(feature_range=(-1, 1))),
                                      ('remove_constant', VarianceThreshold())])

    train = train_x_orig.copy()
    train = np.concatenate([train[['unit_number', 'time']], scaler.fit_transform(train[feature_cols])], axis=1)

    test = test_x_orig.copy()
    test = np.concatenate([test[['unit_number', 'time']], scaler.transform(test[feature_cols])], axis=1)

    # Make engine numbers and days zero-indexed
    train[:, 0:2] -= 1
    test[:, 0:2] -= 1

    # Configurable observation look-back period for each engine/day
    print('Data Transformation')
    max_time = 100
    mask_value = -99

    train_x, train_y = build_data(units=train[:, 0], time=train[:, 1], x=train[:, 2:], max_time=max_time,
                                  is_test=False, mask_value=mask_value, n_units=100)
    test_x, _ = build_data(units=test[:, 0], time=test[:, 1], x=test[:, 2:], max_time=max_time,
                           is_test=True, mask_value=mask_value, n_units=100)

    # Creating event column (0/1 i.e. censored or uncensored)
    # For us it is uncensored
    test_y = test_y_orig.copy()
    test_y['E'] = 1
    test_y = test_y.values

    # modeling
    print('Training')
    trained_model = network(train_x, train_y, test_x, test_y, mask_value)

    return trained_model


if __name__ == '__main__':
    model = main()
    model.save('./model.h5')

