# Data ETL

import pandas as pd
import numpy as np


def load_data():
    
    # for reproducibility
    np.random.seed(42)

    id_col = "unit_number"
    time_col = "time"
    feature_cols = ["op_setting_1", "op_setting_2", "op_setting_3"] + [
        "sensor_measurement_{}".format(x) for x in range(1, 22)
    ]
    column_names = [id_col, time_col] + feature_cols

    train_x_orig = pd.read_csv(
        "./DataSets/CMAPSS/train_FD001.csv", header=None, sep="\s+", decimal="."
    )
    train_x_orig.columns = column_names

    test_x_orig = pd.read_csv(
        "./DataSets/CMAPSS/test_FD001.csv", header=None, sep="\s+", decimal="."
    )
    test_x_orig.columns = column_names

    test_y_orig = pd.read_csv(
        "./DataSets/CMAPSS/RUL_FD001.csv", header=None, names=["T"]
    )

    # Make engine numbers and days zero-indexed
    train_x_orig.iloc[:, 0:2] -= 1
    test_x_orig.iloc[:, 0:2] -= 1

    print(train_x_orig.shape)
    print(test_x_orig.shape)

    train_idx = np.random.choice(
        range(train_x_orig.unit_number.unique().max() + 1), replace=False, size=80
    )  # selecting 80% units for training
    train_idx.sort()

    vld_idx = np.array(
        [
            x
            for x in range(train_x_orig.unit_number.unique().max() + 1)
            if x not in train_idx
        ]
    )  # remaining are validation indices

    train = train_x_orig[train_x_orig.unit_number.isin(train_idx)]  # training data
    train.reset_index(drop=True, inplace=True)
    vld = train_x_orig[train_x_orig.unit_number.isin(vld_idx)]  # validation data
    vld.reset_index(drop=True, inplace=True)

    # Truncating the validation data randomly 5 times each
    vld_trunc = []
    test_y = []
    max_cycle = []
    test_index = []  # for debugging purposes
    temp_or_test_cycles = []
    counter = -1

    for i in set(vld.unit_number.unique()):
        # print(f'unit number is {i}')
        for j in range(1, 6):  # 5 truncations per instance
            counter += 1
            np.random.seed(i * j)
            temp_df = vld[vld.unit_number == i]
            temp_df.reset_index(drop=True, inplace=True)  # important
            length = temp_df.shape[0]
            temp_or_test_cycles.append(length)
            level = np.random.choice(np.arange(5, 96), 1)[0]
            r = np.int(length * (1 - level / 100))
            temp_df = temp_df.truncate(after=r)
            temp_df["unit_number"] = np.repeat(counter, temp_df.shape[0])
            vld_trunc.append(temp_df)
            max_cycle.append(length)

    vld_trunc = pd.concat(vld_trunc)
    vld_trunc.reset_index(drop=True, inplace=True)

    return (
        train,
        feature_cols,
        vld_trunc,
        vld,
        np.array(max_cycle),
        test_x_orig,
        test_y_orig,
    )
