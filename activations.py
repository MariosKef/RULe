from tensorflow.keras import backend as k
from tensorflow import keras
import tensorflow as tf


class Activate(keras.layers.Layer):
    """ Elementwise computation of alpha and regularized beta.
        Wrapper to `output_lambda` using keras.layers.Activation.
        See this for details.
        - Usage
            .. code-block:: python
               wtte_activation = wtte.OuputActivation(init_alpha=1.,
                                                 max_beta_value=4.0).activation
               model.add(Dense(2))
               model.add(Activation(wtte_activation))
    """

    def __init__(self, init_alpha=1.0, max_beta_value=1.0):
        super().__init__()
        self.init_alpha = init_alpha
        self.max_beta_value = max_beta_value

    def call(self, ab):
        """ (Internal function) Activation wrapper
        :param ab: original tensor with alpha and beta.
        :return ab: return of `output_lambda` with `init_alpha` and `max_beta_value`.
        """

        a, b = tf.unstack(ab, axis=-1)
        #         print(a)
        #         print(b)

        #         a = k.exp(a)
        #         b = k.softplus(b)

        #         print(a)
        #         print(b)

        # Implicitly initialize alpha:
        a = self.init_alpha * k.exp(a)

        #         if self.max_beta_value > 1.05:  # some value >>1.0
        #             # shift to start around 1.0
        #             # assuming input is around 0.0
        #             _shift = np.log(self.max_beta_value - 1.0)

        #             b = b - _shift

        b = k.exp(b)  # self.max_beta_value * k.sigmoid(b) # this was affecting the max value of beta

        #         a = a * tf.exp(tf.math.lgamma(1 + 1/b))
        #         b = tf.repeat(1.0, tf.size(a))

        x = k.stack([a, b], axis=-1)

        return x

    def get_config(self):
        return {"init_alpha": self.init_alpha, "max_beta_value": self.max_beta_value}

    @classmethod
    def from_config(cls, config):
        return cls(**config)