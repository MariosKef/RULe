# Modeling script. Used during HPO

# imports
import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2"  # uncomment in case running ONLY on CPU is required

import tensorflow as tf

tf.random.set_seed(42)
from tensorflow.keras import backend as k
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import History
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from activations import Activate
from losses import CustomLoss

import numpy as np


def network(train_X, train_y, test_X, test_y, net_cfg, cfg):
    k.set_epsilon(1e-10)
    history = History()
    nan_terminator = callbacks.TerminateOnNaN()
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss")
    early_stopping = callbacks.EarlyStopping(patience=5)
    # tensorboard = callbacks.TensorBoard(log_dir = './logs_2D')

    window = train_X.shape[1]
    n_features = train_X.shape[2]

    inputs = keras.Input(shape=(window, n_features))
    masking_layer = keras.layers.Masking(mask_value=cfg["mask_value"])(inputs)

    # recurrent layers
    last = 0
    if net_cfg["num_rec"] > 1:
        for i in np.arange(net_cfg["num_rec"] - 1):
            masking_layer = keras.layers.GRU(
                net_cfg["neuron_" + str(i)],
                activation=net_cfg["activation_rec_" + str(i)],
                dropout=net_cfg["rec_dropout_norm_" + str(i)],
                recurrent_dropout=net_cfg["recurrent_dropout_" + str(i)],
                return_sequences=True,
            )(masking_layer)
        last = i + 1

    gru_last = keras.layers.GRU(
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
    model.summary()
    model.fit(
        train_X,
        train_y,
        epochs=cfg["epochs"],
        batch_size=eval(net_cfg["batch"]),
        validation_data=(test_X, test_y),
        verbose=1,
        callbacks=[nan_terminator, history, reduce_lr, early_stopping],  # , tensorboard
        workers=32,
    )

    return model, history
