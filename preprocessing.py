import numpy as np
import tqdm
from tqdm import tqdm


def build_data(units, time, x, max_time, is_test, mask_value, original_data, label='linear', **kwargs):
    """
    This function prepares the data by segmenting it into subsequences of length max_time
    by also padding, by pad_value, the time-steps when there is no data.

    :param units: identifier feature
    :param time: time-index feature
    :param x: sensor values (time-series)
    :param max_time: maximum lookback
    :param is_test: (boolean) test set or train set
    :param mask_value: value to pad the sequences
    :param **kwargs: additional arguments that might be used for other datasets
    :return: (ndarray) y. y[0] will be time remaining to an event, y[1] will be event indicator
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
        # When did the engine fail? (Last day + 1 for train data, irrelevant for test.)
        max_unit_time = int(np.max(time[units == i])) + 1

        if is_test:
            start = max_unit_time - 1
        else:
            start = 0

        this_x = []

        for j in range(start, max_unit_time):

            engine_x = x[units == i]

            if is_test:
                original_max = original_data[original_data.unit_number == i].time.max()
                if label == 'linear':
                    out_y.append(original_max - j)
                else:
                    if j <= original_max / 2:
                        out_y.append(130)  # value taken from Heimes et al. (2008)

                    else:
                        p = (0 - 130) / (original_max - original_max / 2)
                        rul = p * j - p * original_max
                        out_y.append(rul)

            else:
                if label == 'linear':
                    out_y.append(max_unit_time - j)
                else:
                    if j <= max_unit_time / 2:
                        out_y.append(130)  # value taken from Heimes et al. (2008)

                    else:
                        p = (0 - 130) / (max_unit_time - max_unit_time / 2)
                        rul = p * j - p * max_unit_time
                        #                     out_y.append(max_unit_time - j)
                        out_y.append(rul)

            xtemp = np.zeros((1, max_time, d))
            xtemp += mask_value

            xtemp[:, max_time - min(j, max_time - 1) - 1:max_time, :] = engine_x[max(0, j - max_time + 1):j + 1, :]
            this_x.append(xtemp)

        this_x = np.concatenate(this_x)
        out_x.append(this_x)
    out_x = np.concatenate(out_x)
    out_y = np.array(out_y).reshape(len(out_y),
                                    1)  # np.concatenate(out_y) (uncomment when adding event. See comment above)
    return out_x, out_y