import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ[
    "CUDA_VISIBLE_DEVICES"
] = "0,1,2,11,12"  # uncomment in case running ONLY on CPU is required

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
    checkpoint_filepath = "./saved_models_9_12/cp-{epoch:04d}.ckpt"
    checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, monitor="loss", verbose=1
    )
    logdir = "logs/test_9_12"  # + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = callbacks.TensorBoard(log_dir=logdir)

    window = train_X.shape[1]
    n_features = train_X.shape[2]

    with strategy.scope():  # Create a MirroredStrategy.

        inputs = keras.Input(shape=(window, n_features))
        masking_layer = keras.layers.Masking(mask_value=cfg["mask_value"])(inputs)

        # recurrent layers
        if net_cfg["num_rec"] > 1:
            for i in np.arange(net_cfg["num_rec"] - 1):
                masking_layer = keras.layers.GRU(
                    net_cfg["neuron_" + str(i)],
                    activation=net_cfg["activation_" + str(i)],
                    dropout=net_cfg["dropout_" + str(i)],
                    recurrent_dropout=net_cfg["recurrent_dropout_" + str(i)],
                    return_sequences=True,
                )(masking_layer)
        last = i + 1
        gru_last = keras.layers.GRU(
            net_cfg["neuron_" + str(last)],
            activation=net_cfg["activation_" + str(last)],
            dropout=net_cfg["dropout_" + str(last)],
            recurrent_dropout=net_cfg["recurrent_dropout_" + str(last)],
            return_sequences=False,
        )(masking_layer)

        dense_1 = keras.layers.Dense(2)(gru_last)
        custom_activation = Activate(net_cfg=net_cfg)
        outputs = keras.layers.Activation(custom_activation)(dense_1)

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

    _, _ = network(train_x, train_y, net_cfg, cfg)


if __name__ == "__main__":
    epochs = sys.argv[1]

    net_cfg = {
        "num_rec": 5,
        "max_time": 45,
        "neuron_0": 76,
        "neuron_1": 177,
        "neuron_2": 96,
        "neuron_3": 150,
        "neuron_4": 195,
        "neuron_5": 140,
        "neuron_6": 117,
        "neuron_7": 113,
        "neuron_8": 80,
        "neuron_9": 127,
        "activation_0": "sigmoid",
        "activation_1": "tanh",
        "activation_2": "tanh",
        "activation_3": "sigmoid",
        "activation_4": "sigmoid",
        "activation_5": "sigmoid",
        "activation_6": "sigmoid",
        "activation_7": "tanh",
        "activation_8": "tanh",
        "activation_9": "sigmoid",
        "dropout_0": 0.1317451292061026,
        "dropout_1": 0.33078730471815543,
        "dropout_2": 0.06136439672385036,
        "dropout_3": 0.02324045813546879,
        "dropout_4": 0.12165830857644894,
        "dropout_5": 0.8668039692791406,
        "dropout_6": 0.49458108701015635,
        "dropout_7": 0.8692402868162864,
        "dropout_8": 0.3892537655183355,
        "dropout_9": 0.2806414016154028,
        "recurrent_dropout_0": 0.417691002556559,
        "recurrent_dropout_1": 0.8671496818064206,
        "recurrent_dropout_2": 0.46713386179379257,
        "recurrent_dropout_3": 0.6097912652903833,
        "recurrent_dropout_4": 0.28068435348661575,
        "recurrent_dropout_5": 0.6965945004237047,
        "recurrent_dropout_6": 0.6956316044736848,
        "recurrent_dropout_7": 0.46917122992824856,
        "recurrent_dropout_8": 0.8784126808130595,
        "recurrent_dropout_9": 0.11300393778424042,
        "final_activation_0": "exp",
        "final_activation_1": "exp",
        "percentage": 66,
        "rul": 124,
        "rul_style": "nonlinear",
        "lr": "1e-5",
        "batch": "128",
    }
    # net_cfg = {
    #     "num_rec": 3,
    #     "max_time": 26,
    #     "neuron_0": 73,
    #     "neuron_1": 71,
    #     "neuron_2": 82,
    #     "neuron_3": 82,
    #     "activation_0": "tanh",
    #     "activation_1": "sigmoid",
    #     "activation_2": "sigmoid",
    #     "activation_3": "tanh",
    #     "dropout_0": 0.06943171652267692,
    #     "dropout_1": 0.12639579059484615,
    #     "dropout_2": 0.3822443511564662,
    #     "dropout_3": 0.4580962846531429,
    #     "recurrent_dropout_0": 0.3280089650917844,
    #     "recurrent_dropout_1": 0.69930466502713,
    #     "recurrent_dropout_2": 0.24506744915217923,
    #     "recurrent_dropout_3": 0.7699919737017498,
    #     "final_activation_0": "exp",
    #     "final_activation_1": "softplus",
    #     "percentage": 73,
    #     "rul": 121,
    #     "rul_style": "nonlinear",
    #     "lr": "0.005357912753227542",
    #     "batch": "128",
    # }

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
