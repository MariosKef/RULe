from tensorflow.keras import backend as k
from tensorflow import keras
import tensorflow as tf
import importlib


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

    def __init__(self, net_cfg):
        super().__init__()
        self.func1 = net_cfg['final_activation_0']
        self.func2 = net_cfg['final_activation_1']

    def call(self, ab):
        """ (Internal function) Activation wrapper
        :param ab: original tensor with alpha and beta.
        :return ab: return of `output_lambda` with `init_alpha` and `max_beta_value`.
        """

        a, b = tf.unstack(ab, axis=-1)
        a = eval('k.'+self.func1+"(a)")  # a = k.exp(a)
        b = eval('k.'+self.func2+"(b)")  # b = k.softplus(b)

        x = k.stack([a, b], axis=-1)

        return x

    def get_config(self):
        return {"func1": self.func1, "func2": self.func2}

    @classmethod
    def from_config(cls, config):
        return cls(**config)