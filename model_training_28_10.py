import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"  # uncomment in case running ONLY on CPU is required

import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras import backend as k
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import History
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from activations import Activate
from losses import CustomLoss
from preprocessing import build_data

import numpy as np
import pandas as pd
import math
from datetime import datetime
import sys

from sklearn import pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


def network(train_X, train_y, net_cfg, cfg):
    k.set_epsilon(1e-10)
    history = History()
    nan_terminator = callbacks.TerminateOnNaN()
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss')
    early_stopping = callbacks.EarlyStopping(monitor='loss', patience=5)
    checkpoint_filepath = './saved_models/cp-{epoch:04d}.ckpt'
    checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='loss', verbose=1)
    logdir="logs/test1" #+ datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = callbacks.TensorBoard(log_dir=logdir)

    window = train_X.shape[1]
    n_features = train_X.shape[2]

    inputs = keras.Input(shape=(window, n_features))
    masking_layer = keras.layers.Masking(mask_value=cfg['mask_value'])(inputs)

    # recurrent layers
    if net_cfg['num_rec'] > 1:
        for i in np.arange(net_cfg['num_rec']-1):
            masking_layer = keras.layers.GRU(net_cfg['neuron_'+str(i)], activation=net_cfg['activation_'+str(i)],
                                    dropout=net_cfg['dropout_'+str(i)],
                                    recurrent_dropout=net_cfg['recurrent_dropout_'+str(i)],
                                    return_sequences=True)(masking_layer)
    last = i + 1
    gru_last = keras.layers.GRU(net_cfg['neuron_'+str(last)], activation=net_cfg['activation_'+str(last)],
                                dropout=net_cfg['dropout_'+str(last)],
                                recurrent_dropout=net_cfg['recurrent_dropout_'+str(last)],
                                return_sequences=False)(masking_layer)

    dense_1 = keras.layers.Dense(2)(gru_last)
    custom_activation = Activate(net_cfg=net_cfg)
    outputs = keras.layers.Activation(custom_activation)(dense_1)

    model = keras.Model(inputs=inputs, outputs=outputs, name="weibull_params")

    # rmse = tf.keras.metrics.RootMeanSquaredError()
    model.compile(loss=CustomLoss(kind='continuous', reduce_loss=True), optimizer=Adam(lr=net_cfg['lr'],
                                                                                       clipvalue=0.5))
    model.summary()  # uncomment for debugging

    batch_size=eval(net_cfg['batch'])
    model.fit(train_X, train_y,
              epochs=cfg['epochs'],
              batch_size=batch_size,
              verbose=1,
              callbacks=[nan_terminator, history, reduce_lr, early_stopping, checkpoint, tensorboard], 
              workers=32)

    return model, history


def load_data():

    np.random.seed(42)

    id_col = 'unit_number'
    time_col = 'time'
    feature_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3'] + ['sensor_measurement_{}'.format(x) for x in
                                                                       range(1, 22)]
    column_names = [id_col, time_col] + feature_cols

    train_x_orig = pd.read_csv('./DataSets/CMAPSS/train_FD001.csv', header=None, sep='\s+', decimal=".")
    train_x_orig.columns = column_names

    test_x_orig = pd.read_csv('./DataSets/CMAPSS/test_FD001.csv', header=None, sep='\s+', decimal=".")
    test_x_orig.columns = column_names

    test_y_orig = pd.read_csv('./DataSets/CMAPSS/RUL_FD001.csv', header=None, names=['T'])

    # Make engine numbers and days zero-indexed
    train_x_orig.iloc[:, 0:2] -= 1
    test_x_orig.iloc[:, 0:2] -= 1

    # Pre-processing data
    scaler = pipeline.Pipeline(steps=[
        ('minmax', MinMaxScaler(feature_range=(-1, 1))),
        ('remove_constant', VarianceThreshold())])

    train = train_x_orig.copy()
    train = np.concatenate([train[['unit_number', 'time']], scaler.fit_transform(train[feature_cols])], axis=1)

    train_x, train_y = build_data(units=train[:, 0], time=train[:, 1], x=train[:, 2:], max_time=net_cfg['max_time'],
                                    is_test=False, mask_value=cfg['mask_value'],
                                    original_data=None, net_cfg = net_cfg, label=net_cfg['rul_style'])

    test_or = test_x_orig.copy()
    test_or = np.concatenate([test_or[['unit_number', 'time']], scaler.transform(test_or[feature_cols])], axis=1)


    # Preparing data for the RNN (numpy arrays)
    test_or, _ = build_data(units=test_or[:, 0], time=test_or[:, 1], x=test_or[:, 2:], max_time=net_cfg['max_time'],
                                    is_test=True, mask_value=-99,
                                    original_data=np.repeat(200, test_or.shape[0]), net_cfg = net_cfg, label=net_cfg['rul_style'])

    

    return train_x, train_y, test_or, test_y_orig, train_x_orig

def main(net_cfg, cfg):
    k.clear_session()
    try:
        del model
    except:
        pass

    train_x, train_y, _, _, _ = load_data()

    _, _ = network(train_x, train_y, net_cfg, cfg)


if __name__ == '__main__':
    epochs = sys.argv[1]

    net_cfg = {'num_rec': 4, 'max_time': 24, 'neuron_0': 76, 'neuron_1': 75, 'neuron_2': 74, 'neuron_3': 66, 'activation_0': 'tanh',
        'activation_1': 'tanh', 'activation_2': 'sigmoid', 'activation_3': 'sigmoid', 'dropout_0': 0.018692516794622607,
        'dropout_1': 0.8002018342665917, 'dropout_2': 0.615094589188039, 'dropout_3': 0.08230738757019833, 'recurrent_dropout_0': 0.6421264747391056,
        'recurrent_dropout_1': 0.8933998465284962, 'recurrent_dropout_2': 0.6402495109098905, 'recurrent_dropout_3': 0.6693624215836003,
            'final_activation_0': 'softplus', 'final_activation_1': 'softplus', 'percentage': 62, 'rul': 124, 'rul_style': 'nonlinear',
            'lr': 0.0008896860421074306, 'batch': '32'}

    cfg = {'cv': 10, 'shuffle': True,
    'random_state': 21,
    'mask_value': -99,
    'reps': 30,
    'epochs': eval(epochs),
    'batches': 64}  # batches not used
    
    print(net_cfg)
    print('\n')
    print(cfg)

    main(net_cfg, cfg)