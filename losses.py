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

    loglikelihoods = u * \
                     k.log(k.exp(hazard1 - hazard0) - (1.0 - epsilon)) - hazard1
    return loglikelihoods


def loglik_continuous(y, u, a, b, epsilon=k.epsilon()):
    ya = (y + epsilon) / a

    loglikelihoods = u * (k.log(b) + b * k.log(ya)) - k.pow(ya, b)
    return loglikelihoods


class CustomLoss(keras.losses.Loss):
    """ Creates a keras WTTE-loss function.
        - Usage
            :Example:
            .. code-block:: python
               loss = wtte.Loss(kind='discrete').loss_function
               model.compile(loss=loss, optimizer=RMSprop(lr=0.01))
               # And with masking:
               loss = wtte.Loss(kind='discrete',reduce_loss=False).loss_function
               model.compile(loss=loss, optimizer=RMSprop(lr=0.01),
                              sample_weight_mode='temporal')
        .. note::
            With masking keras needs to access each loss-contribution individually.
            Therefore we do not sum/reduce down to scalar (dim 1), instead return a 
            tensor (with reduce_loss=False).
        :param kind:  One of 'discrete' or 'continuous'
        :param reduce_loss: 
        :param clip_prob: Clip likelihood to [log(clip_prob),log(1-clip_prob)]
        :param regularize: Deprecated.
        :param location: Deprecated.
        :param growth: Deprecated.
        :type reduce_loss: Boolean
    """

    def __init__(self,
                 kind,
                 reduce_loss=False):
        super().__init__()
        self.kind = kind
        self.reduce_loss = reduce_loss

    def call(self, y_true, y_pred):

        #         y, u = tf.unstack(y_true, axis=-1) # (uncomment when adding event)
        y = tf.cast(y_true, tf.float32)  # (replace y_true -> y when adding event)
        y = tf.reshape(y, [-1])  # (coment when adding event)
        #         u = tf.cast(u, tf.float32) # (uncomment when adding event)
        u = tf.constant(1, dtype=tf.float32)
        a, b = tf.unstack(y_pred, axis=-1)

        if self.kind == 'discrete':
            loglikelihoods = loglik_discrete(y, u, a, b)
        elif self.kind == 'continuous':
            loglikelihoods = loglik_continuous(y, u, a, b)
        elif self.kind == 'mttf':
            loglikelihoods = weibull_ttf(y, u, a, b)

        if self.reduce_loss:
            loss = -1.0 * (k.mean(loglikelihoods, axis=-1))  # apparently this returns the same as the case below
        else:
            loss = -1.0 * loglikelihoods

        return loss

    def get_config(self):
        return {"kind": self.kind}

    @classmethod
    def from_config(cls, config):
        return cls(**config)