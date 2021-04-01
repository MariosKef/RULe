from tensorflow.keras import backend as k


def activate(ab):
    """
    Custom Keras activation function, outputs alpha neuron using exponentiation and beta using softplus
    :param ab:
    :return:
    """
    a = k.exp(ab[:, 0])
    b = k.softplus(ab[:, 1])

    a = k.reshape(a, (k.shape(a)[0], 1))
    b = k.reshape(b, (k.shape(b)[0], 1))

    return k.concatenate((a, b), axis=1)
