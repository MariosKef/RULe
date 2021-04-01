from tensorflow.keras import backend as k
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Dense, Activation, Masking, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from activations import activate
from losses import weibull_loglik_discrete
import numpy as np


def network(train_X, train_y, test_X, test_y, mask_value):

    tte_mean_train = np.nanmean(train_y[:, 0])
    mean_u = np.nanmean(train_y[:, 1])

    # Initialization value for alpha-bias
    init_alpha = -1.0 / np.log(1.0 - 1.0 / (tte_mean_train + 1.0))
    init_alpha = init_alpha / mean_u
    print('tte_mean_train', tte_mean_train, 'init_alpha: ', init_alpha, 'mean uncensored train: ', mean_u)

    k.set_epsilon(1e-10)
    history = History()
    nan_terminator = callbacks.TerminateOnNaN()

    # reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss',
    #                                         factor=0.5,
    #                                         patience=50,
    #                                         verbose=0,
    #                                         mode='auto',
    #                                         epsilon=0.0001,
    #                                         cooldown=0,
    #                                         min_lr=1e-8)

    n_features = train_X.shape[-1]

    # Start building our model
    model = Sequential()
    # Mask parts of the lookback period that are all zeros (i.e., unobserved) so they don't skew the model
    model.add(Masking(mask_value=mask_value, input_shape=(None, n_features)))
    model.add(GRU(20, activation='tanh', recurrent_dropout=0.25, return_sequences=True))
    model.add(GRU(10, activation='tanh', recurrent_dropout=0.25))
    model.add(Dense(2))
    model.add(Activation(activate))

    # Use the discrete log-likelihood for Weibull survival data as our loss function
    model.compile(loss=weibull_loglik_discrete, optimizer=Adam(lr=.01, clipvalue=0.5))

    model.fit(train_X, train_y,
              epochs=20,
              batch_size=100,
              verbose=1,
              validation_data=(test_X, test_y),
              callbacks=[nan_terminator, history],
              workers=32)

    return model
