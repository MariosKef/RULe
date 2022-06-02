# Script defining activation functions for 
# final dense layer of the network.

from tensorflow.keras import backend as k
from tensorflow import keras
import tensorflow as tf
import importlib


class Activate(keras.layers.Layer):
    """ 
        Activation functions for final dense layer of the
        network.
        Elementwise computation of alpha and regularized beta.
        Uses keras.layers.Activation (see modeling.py).
    """

    def __init__(self, net_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.net_cfg = net_cfg
        

    def call(self, ab):
        """ (Internal function) Activation wrapper
        :param ab: original tensor with alpha and beta.
        :return x: return of keras.layers.Activation with `alpha` and `beta`.
        """

        self.func1 = self.net_cfg['final_activation_0']
        self.func2 = self.net_cfg['final_activation_1']

        a, b = tf.unstack(ab, axis=-1)
        # print('f1: k.'+self.func1+"(a)")
        a = eval('k.'+self.func1+"(a)")  # uncomment for debugging
        # print('f2: k.'+self.func2+"(b)")
        b = eval('k.'+self.func2+"(b)")  # uncomment for debugging

        x = k.stack([a, b], axis=-1)

        return x

    def get_config(self):
        return {"net_cfg": self.net_cfg}

    @classmethod
    def from_config(cls, config):
        return cls(**config)