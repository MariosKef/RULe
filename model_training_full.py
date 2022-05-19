# Script to train model on entirety of data on selected
# hyperparameter configuration.

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ[
    "CUDA_VISIBLE_DEVICES"
] = "0,1,2,3,4,5,6,7,8,9"  # '-1' in case running ONLY on CPU is required

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
import time

from sklearn import pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


def network(train_X, train_y, net_cfg, cfg):

    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    k.set_epsilon(1e-10)
    history = History()
    nan_terminator = callbacks.TerminateOnNaN()
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="loss")
    early_stopping = callbacks.EarlyStopping(monitor="loss", patience=5)
    checkpoint_filepath = (
        "./Final_experiments/dataset_1/best/saved_models_4_2/cp-{epoch:04d}.ckpt"
    )
    checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, monitor="loss", verbose=1
    )
    logdir = "Final_experiments/dataset_1/best/logs"  # + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = callbacks.TensorBoard(log_dir=logdir)

    window = train_X.shape[1]
    n_features = train_X.shape[2]

    with strategy.scope():  # Create a MirroredStrategy.

        inputs = keras.Input(shape=(window, n_features))
        masking_layer = keras.layers.Masking(mask_value=cfg["mask_value"])(inputs)

        # recurrent layers
        last = 0
        if net_cfg["num_rec"] > 1:
            for i in np.arange(net_cfg["num_rec"] - 1):
                masking_layer = keras.layers.LSTM(
                    net_cfg["neuron_" + str(i)],
                    activation=net_cfg["activation_rec_" + str(i)],
                    dropout=net_cfg["rec_dropout_norm_" + str(i)],
                    recurrent_dropout=net_cfg["recurrent_dropout_" + str(i)],
                    return_sequences=True,
                )(masking_layer)
            last = i + 1

        gru_last = keras.layers.LSTM(
            net_cfg["neuron_" + str(last)],
            activation=net_cfg["activation_rec_" + str(last)],
            dropout=net_cfg["rec_dropout_norm_" + str(last)],
            recurrent_dropout=net_cfg["recurrent_dropout_" + str(last)],
            return_sequences=False,
        )(masking_layer)

        # dense layers
        last = 0
        if net_cfg["num_den"] > 1:
            for i in np.arange(net_cfg["num_den"] - 1):
                gru_last = keras.layers.Dense(
                    net_cfg["neuron_den_" + str(i)],
                    activation=net_cfg["activation_den_" + str(i)],
                )(gru_last)
                gru_last = keras.layers.Dropout(
                    rate=net_cfg["dropout_" + str(i)],
                )(gru_last)
            last = i + 1

        dense_ = keras.layers.Dense(2)(gru_last)
        custom_activation = Activate(net_cfg=net_cfg)
        outputs = keras.layers.Activation(custom_activation)(dense_)

        model = keras.Model(inputs=inputs, outputs=outputs, name="weibull_params")

        # rmse = tf.keras.metrics.RootMeanSquaredError()
        model.compile(
            loss=CustomLoss(kind="continuous", reduce_loss=True),
            optimizer=Adam(lr=eval(net_cfg["lr"]), clipvalue=0.5),
        )
    model.summary()  # uncomment for debugging

    batch_size = eval(net_cfg["batch"])
    model.fit(
        train_X,
        train_y,
        epochs=cfg["epochs"],
        batch_size=batch_size,
        verbose=1,
        callbacks=[
            nan_terminator,
            history,
            reduce_lr,
            early_stopping,
            checkpoint,
            tensorboard,
        ],
        workers=32,
    )

    return model, history


def load_data():

    np.random.seed(42)

    id_col = "unit_number"
    time_col = "time"
    feature_cols = ["op_setting_1", "op_setting_2", "op_setting_3"] + [
        "sensor_measurement_{}".format(x) for x in range(1, 22)
    ]
    column_names = [id_col, time_col] + feature_cols

    train_x_orig = pd.read_csv(
        "./DataSets/CMAPSS/train_FD001.csv", header=None, sep="\s+", decimal="."
    )
    train_x_orig.columns = column_names

    test_x_orig = pd.read_csv(
        "./DataSets/CMAPSS/test_FD001.csv", header=None, sep="\s+", decimal="."
    )
    test_x_orig.columns = column_names

    test_y_orig = pd.read_csv(
        "./DataSets/CMAPSS/RUL_FD001.csv", header=None, names=["T"]
    )

    # Make engine numbers and days zero-indexed
    train_x_orig.iloc[:, 0:2] -= 1
    test_x_orig.iloc[:, 0:2] -= 1

    # Pre-processing data
    scaler = pipeline.Pipeline(
        steps=[
            ("minmax", MinMaxScaler(feature_range=(-1, 1))),
            ("remove_constant", VarianceThreshold()),
        ]
    )

    train = train_x_orig.copy()
    train = np.concatenate(
        [train[["unit_number", "time"]], scaler.fit_transform(train[feature_cols])],
        axis=1,
    )

    train_x, train_y = build_data(
        units=train[:, 0],
        time=train[:, 1],
        x=train[:, 2:],
        max_time=net_cfg["max_time"],
        is_test=False,
        mask_value=cfg["mask_value"],
        original_data=None,
        net_cfg=net_cfg,
        label=net_cfg["rul_style"],
    )

    test_or = test_x_orig.copy()
    test_or = np.concatenate(
        [test_or[["unit_number", "time"]], scaler.transform(test_or[feature_cols])],
        axis=1,
    )

    # Preparing data for the RNN (numpy arrays)
    test_or, _ = build_data(
        units=test_or[:, 0],
        time=test_or[:, 1],
        x=test_or[:, 2:],
        max_time=net_cfg["max_time"],
        is_test=True,
        mask_value=-99,
        original_data=np.repeat(200, test_or.shape[0]),
        net_cfg=net_cfg,
        label=net_cfg["rul_style"],
    )

    return train_x, train_y, test_or, test_y_orig, train_x_orig


def main(net_cfg, cfg):
    k.clear_session()

    train_x, train_y, _, _, _ = load_data()
    print(train_x.shape)
    print(train_y.shape)

    _, _ = network(train_x, train_y, net_cfg, cfg)


if __name__ == "__main__":
    epochs = sys.argv[1]
    
    # enter hyperparameter
    # configuration here.
    net_cfg = net_cfg = {
        "num_rec": 1,
        "max_time": 38,
        "neuron_0": 49,
        "neuron_1": 25,
        "neuron_2": 89,
        "activation_rec_0": "sigmoid",
        "activation_rec_1": "tanh",
        "activation_rec_2": "sigmoid",
        "rec_dropout_norm_0": 0.43733610344927026,
        "rec_dropout_norm_1": 0.5264641431370682,
        "rec_dropout_norm_2": 0.39651681655854976,
        "recurrent_dropout_0": 0.6246405485246954,
        "recurrent_dropout_1": 0.35541873905868654,
        "recurrent_dropout_2": 0.5826687367719754,
        "final_activation_0": "softplus",
        "final_activation_1": "softplus",
        "percentage": 40,
        "rul": 118,
        "rul_style": "nonlinear",
        "lr": "1e-4",
        "batch": "64",
        "num_den": 1,
        "neuron_den_0": 52,
        "neuron_den_1": 41,
        "neuron_den_2": 55,
        "activation_den_0": "tanh",
        "activation_den_1": "sigmoid",
        "activation_den_2": "tanh",
        "dropout_0": 0.15695483248610206,
        "dropout_1": 0.4514125634405938,
        "dropout_2": 0.6310550702551022,
    }
    cfg = {
        "cv": 10,
        "shuffle": True,
        "random_state": 21,
        "mask_value": -99,
        "reps": 30,
        "epochs": eval(epochs),
        "batches": 64,
    }  # batches not used

    print(net_cfg)
    print("\n")
    print(cfg)

    start = time.time()
    main(net_cfg, cfg)
    end = time.time()

    print(f"Elapsed time: {(end-start)/60} minutes")
