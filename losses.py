# Tensorflow
import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow import keras


def weibull_ttf(y, u, a, b):
    mttf = a * tf.exp(tf.math.lgamma(1 + 1 / b))

    return k.square(y - mttf)


def loglik_discrete(y, u, a, b, epsilon=k.epsilon()):
    hazard0 = k.pow((y + epsilon) / a, b)
    hazard1 = k.pow((y + 1.0) / a, b)

    loglikelihoods = u * k.log(k.exp(hazard1 - hazard0) - (1.0 - epsilon)) - hazard1
    return loglikelihoods


def loglik_continuous(y, u, a, b, epsilon=k.epsilon()):
    ya = (y + epsilon) / a

    loglikelihoods = u * (k.log(b) + b * k.log(ya)) - k.pow(ya, b)
    return loglikelihoods


class CustomLoss(keras.losses.Loss):
    """Creates a keras WTTE-loss function.
    :param kind:  One of 'discrete' or 'continuous'
    :type reduce_loss: Boolean
    """

    def __init__(self, kind, reduce_loss=False, **kwargs):
        super().__init__(**kwargs)
        self.kind = kind
        self.reduce_loss = reduce_loss

    def call(self, y_true, y_pred):

        y = tf.cast(y_true, tf.float32)
        y = tf.reshape(y, [-1])

        u = tf.constant(1, dtype=tf.float32)
        a, b = tf.unstack(y_pred, axis=-1)

        if self.kind == "discrete":
            loglikelihoods = loglik_discrete(y, u, a, b)
        elif self.kind == "continuous":
            loglikelihoods = loglik_continuous(y, u, a, b)
        elif self.kind == "mttf":
            loglikelihoods = weibull_ttf(y, u, a, b)

        if self.reduce_loss:
            loss = -1.0 * (
                k.mean(loglikelihoods, axis=-1)
            )  # apparently this returns the same as the case below
        else:
            loss = -1.0 * loglikelihoods

        return loss

    def get_config(self):
        return {"kind": self.kind}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
