import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def return_rate(updown_rate, y_pred):
    updown_rate, y_pred = np.asarray(updown_rate), np.asarray(y_pred, dtype=np.bool)
    return np.prod(1 + updown_rate[y_pred]) - 1


def cum_return_rate(updown_rate, y_pred):
    updown_rate, y_pred = np.asarray(updown_rate), np.asarray(y_pred, dtype=np.bool)
    return np.cumprod(1 + updown_rate * y_pred) - 1


def profit(updown, y_pred):
    updown, y_pred = np.asarray(updown), np.asarray(y_pred, dtype=np.bool)
    return np.sum(updown[y_pred])


def cum_profit(updown, y_pred):
    updown, y_pred = np.asarray(updown), np.asarray(y_pred, dtype=np.bool)
    return np.cumsum(updown*y_pred)

