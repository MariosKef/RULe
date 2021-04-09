import tensorflow as tf
from tensorflow.keras import backend as k


def weibull_loglik_discrete(y_true, ab_pred, name=None):
    """

    Discrete log-likelihood for Weibull hazard function on censored survival data
    For math, see https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (Page 35)

    :param y_true: y_true is a (samples, 2) tensor containing time-to-event (y), and an event indicator (u)
    :param ab_pred: ab_pred is a (samples, 2) tensor containing predicted Weibull alpha (a) and beta (b) parameters
    :param name: name of the loss
    :return:
    """
    y_ = y_true[:, 0]
    y_ = tf.cast(y_, tf.float32)
    u_ = y_true[:, 1]
    u_ = tf.cast(u_, tf.float32)
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    hazard0 = k.pow((y_ + k.epsilon()) / a_, b_)
    hazard1 = k.pow((y_ + 1.0) / a_, b_)

    return -1 * k.mean(u_ * k.log(k.exp(hazard1 - hazard0) - (1.0 - k.epsilon())) - hazard1)


"""
    Not used for this model, but included in case somebody needs it
    For math, see https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (Page 35)
"""


def weibull_loglik_continuous(y_true, ab_pred, name=None):
    """
    Continuous log-likelihood for Weibull hazard function on censored survival data
    For math, see https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (Page 35)

    :param y_true: y_true is a (samples, 2) tensor containing time-to-event (y), and an event indicator (u)
    :param ab_pred: ab_pred is a (samples, 2) tensor containing predicted Weibull alpha (a) and beta (b) parameters
    :param name: name of the loss
    :return:

    """
    y_ = y_true[:, 0]
    y_ = tf.cast(y_, tf.float32)
    u_ = y_true[:, 1]
    u_ = tf.cast(u_, tf.float32)
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    ya = (y_ + k.epsilon()) / a_
    return -1 * k.mean(u_ * (k.log(b_) + b_ * k.log(ya)) - k.pow(ya, b_))
