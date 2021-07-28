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
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss')
    early_stopping = callbacks.EarlyStopping(patience=10)
    #     tensorboard = callbacks.TensorBoard(log_dir = './logs_2D')

    window = train_X.shape[1]
    n_features = train_X.shape[2]

    inputs = keras.Input(shape=(window, n_features))
    masking_layer = keras.layers.Masking(mask_value=cfg['mask_value'])(inputs)

    # recurrent layers
    if net_cfg['num_rec'] > 1:
        for _ in np.arange(net_cfg['num_rec']-1):
            masking_layer = keras.layers.GRU(net_cfg['neuron'], activation=net_cfg['activation'],
                                    dropout=net_cfg['dropout'],
                                    ecurrent_dropout=net_cfg['recurrent_dropout'],
                                    return_sequences=True)(masking_layer)
    gru_last = keras.layers.GRU(net_cfg['neuron'], activation=net_cfg['activation'],
                                dropout=net_cfg['dropout'],
                                ecurrent_dropout=net_cfg['recurrent_dropout'],
                                return_sequences=False)(masking_layer)

    dense_1 = keras.layers.Dense(2)(gru_last)
    custom_activation = Activate()
    outputs = keras.layers.Activation(custom_activation)(dense_1)

    model = keras.Model(inputs=inputs, outputs=outputs, name="weibull_params")

    # rmse = tf.keras.metrics.RootMeanSquaredError()
    model.compile(loss=CustomLoss(kind='continuous', reduce_loss=True), optimizer=Adam(lr=.01,
                                                                                       clipvalue=0.5))
    # model.summary() uncomment for debugging

    model.fit(train_X, train_y,
              epochs=net_cfg['epochs'],
              batch_size=net_cfg['batch_size'],
              validation_data=(test_X, test_y),
              verbose=1,
              callbacks=[nan_terminator, history, reduce_lr, early_stopping],  # , tensorboard
              workers=32)

    return model, history
