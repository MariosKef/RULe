# Script for preparing the data for the RNN

import numpy as np
import tqdm
from tqdm import tqdm


def build_data(
    units,
    time,
    x,
    max_time,
    is_test,
    mask_value,
    original_data,
    net_cfg,
    label="linear",
    **kwargs
):
    """
    This function prepares the data by segmenting it into subsequences of length max_time
    by also padding, by pad_value, the time-steps when there is no data.

    :param units: identifier feature
    :param time: time-index feature
    :param x: sensor values (time-series)
    :param max_time: maximum lookback
    :param is_test: (boolean) test set or train set
    :param mask_value: value to pad the sequences
    :param label: the label creation method ('linear' or 'nonlinear')
    :param **kwargs: additional arguments that might be used for other datasets
    """

    # initializing output
    out_y = []

    # number of features/sensors
    d = x.shape[1]

    # A full history of sensor readings to date for each x
    out_x = []
    n_units = set(units)
    # print(n_units)
    for i in tqdm(n_units):
        # When did the engine fail? (Last day + 1 for train data. This is irrelevant for test data.)
        max_unit_time = int(np.max(time[units == i])) + 1

        if is_test:
            start = max_unit_time - 1
        else:
            start = 0

        this_x = []

        for j in range(start, max_unit_time):

            engine_x = x[units == i]

            if is_test:

                original_max = original_data[int(i)]

                if label == "linear":
                    out_y.append(original_max - j)
                else:
                    if j <= int(original_max * net_cfg["percentage"] / 100):
                        out_y.append(net_cfg["rul"])

                    else:
                        p = (0 - net_cfg["rul"]) / (
                            original_max
                            - int(original_max * net_cfg["percentage"] / 100)
                        )
                        rul = p * j - p * original_max
                        out_y.append(rul)

            else:
                if label == "linear":
                    out_y.append(max_unit_time - j)
                else:
                    if j <= int(max_unit_time * net_cfg["percentage"] / 100):
                        out_y.append(net_cfg["rul"])

                    else:
                        p = (0 - net_cfg["rul"]) / (
                            max_unit_time
                            - int(max_unit_time * net_cfg["percentage"] / 100)
                        )
                        rul = p * j - p * max_unit_time

                        out_y.append(rul)

            xtemp = np.zeros((1, max_time, d))
            xtemp += mask_value

            xtemp[:, max_time - min(j, max_time - 1) - 1 : max_time, :] = engine_x[
                max(0, j - max_time + 1) : j + 1, :
            ]
            this_x.append(xtemp)

        this_x = np.concatenate(this_x)
        out_x.append(this_x)
    out_x = np.concatenate(out_x)
    out_y = np.array(out_y).reshape(len(out_y), 1)

    return out_x, out_y
