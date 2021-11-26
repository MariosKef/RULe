# sklearn
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np

# Various
import pandas as pd
import json
import sys
from datetime import datetime

# tensorflow
import tensorflow as tf

tf.random.set_seed(42)
from sklearn import pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as k

# utilities
from data import load_data
from modeling import network
from preprocessing import build_data


def weibull_mean(alpha, beta):
    return alpha * math.gamma(1 + 1 / beta)


model = None

date = datetime.today().strftime("%d_%m_%Y")


def obj_function(net_cfg, cfg=None):

    if cfg == None:
        cfg = {
            "cv": 10,
            "shuffle": True,
            "random_state": 21,
            "mask_value": -99,
            "reps": 30,
            "epochs": 100,
            "batches": 64,
            "in_reps": 10,
        }

    # deleting model if it exists
    k.clear_session()

    (
        train_x_orig,
        feature_cols,
        vld_trunc,
        vld_x_orig,
        original_len,
        test_x_orig,
        test_y_orig,
    ) = load_data()
    print(train_x_orig.shape)
    print(vld_trunc.shape)
    print(original_len.shape)
    # print(original_len)
    # print(vld_trunc.unit_number.unique())

    rmse_train = []
    r2_train = []
    mae_train = []
    std_train = []

    rmse_test = []
    r2_test = []
    mae_test = []
    std_test = []

    train_all = []
    test_all = []

    file = "results_no_cv_HO_" + date
    columns = [
        "fold",
        "rmse_train",
        "mae_train",
        "r2_train",
        "std_train",
        "rmse_test",
        "mae_test",
        "r2_test",
        "std_test",
        "net_cfg",
    ]
    results = pd.DataFrame(columns=columns)
    start = time.time()

    # Pre-processing data
    scaler = pipeline.Pipeline(
        steps=[
            ("minmax", MinMaxScaler(feature_range=(-1, 1))),
            ("remove_constant", VarianceThreshold()),
        ]
    )

    train = train_x_orig.copy()
    train = np.concatenate(
        [train[["unit_number", "time"]], scaler.fit_transform(train[feature_cols])],
        axis=1,
    )

    vld = vld_trunc.copy()
    vld = np.concatenate(
        [vld[["unit_number", "time"]], scaler.transform(vld[feature_cols])], axis=1
    )

    # Preparing data for the RNN (numpy arrays)
    train_x, train_y = build_data(
        units=train[:, 0],
        time=train[:, 1],
        x=train[:, 2:],
        max_time=net_cfg["max_time"],
        is_test=False,
        mask_value=cfg["mask_value"],
        original_data=None,
        net_cfg=net_cfg,
        label=net_cfg["rul_style"],
    )

    test_x, test_y = build_data(
        units=vld[:, 0],
        time=vld[:, 1],
        x=vld[:, 2:],
        max_time=net_cfg["max_time"],
        is_test=True,
        mask_value=cfg["mask_value"],
        original_data=original_len,
        net_cfg=net_cfg,
        label=net_cfg["rul_style"],
    )

    # print(test_y)
    # only for debugging
    print(
        "train_x",
        train_x.shape,
        "train_y",
        train_y.shape,
        "test_x",
        test_x.shape,
        "test_y",
        test_y.shape,
    )

    # training
    model, history = network(train_x, train_y, test_x, test_y, net_cfg, cfg)

    # For debugging
    # plt.plot(history.history['loss'], label='training')
    # plt.plot(history.history['val_loss'], label='validation')
    # plt.title('loss')
    # plt.legend()

    success = True
    # predicting the rul on the train fold
    total_res_train = []
    total_res_test = []

    try:
        for i in range(cfg["reps"]):
            tf.random.set_seed(i)
            train_predict = model(train_x, training=True).numpy()
            a, b = train_predict[:, 0], train_predict[:, 1]

            res = []
            for j in range(a.shape[0]):
                for _ in range(cfg["in_reps"]):
                    sample = a[j] * np.random.weibull(b[j])
                    while np.isinf(sample):
                        sample = a[j] * np.random.weibull(b[j])
                    res.append(sample)
        total_res_train.append(np.array(res))

        total_res_train = np.array(total_res_train)
        total_res_train = np.reshape(
            total_res_train, (cfg["reps"], a.shape[0], cfg["in_reps"])
        )

        train_predict = np.mean(total_res_train, axis=(0, 2))
        train_predict = np.reshape(train_predict, (train_x.shape[0], 1))
        train_result = np.concatenate((train_y, train_predict), axis=1)
        train_results_df = pd.DataFrame(train_result, columns=["T", "predicted_mu"])
        train_results_df["unit_number"] = train_x_orig["unit_number"].to_numpy()
        train_results_df["time"] = train_x_orig["time"].to_numpy()

        # predicting the rul on the test fold
        for i in range(cfg["reps"]):
            tf.random.set_seed(i)
            test_predict = model(test_x, training=True).numpy()
            a, b = test_predict[:, 0], test_predict[:, 1]

            res = []
            for j in range(a.shape[0]):
                for _ in range(cfg["in_reps"]):
                    sample = a[j] * np.random.weibull(b[j])
                    while np.isinf(sample):
                        sample = a[j] * np.random.weibull(b[j])
                    res.append(sample)
        total_res_test.append(np.array(res))

        total_res_test = np.array(total_res_test)
        total_res_test = np.reshape(
            total_res_test, (cfg["reps"], a.shape[0], cfg["in_reps"])
        )

        test_predict = np.mean(total_res_test, axis=(0, 2))
        test_predict = np.reshape(test_predict, (test_x.shape[0], 1))
        test_result = np.concatenate((test_y, test_predict), axis=1)
        test_results_df = pd.DataFrame(test_result, columns=["T", "predicted_mu"])

        # General administration
        train_all.append(train_results_df)
        test_all.append(test_results_df)

        # Performance evaluation
        # train:
        rmse_train.append(
            np.sqrt(
                mean_squared_error(
                    train_results_df["predicted_mu"], train_results_df["T"]
                )
            )
        )
        mae_train.append(
            (
                mean_absolute_error(
                    train_results_df["predicted_mu"], train_results_df["T"]
                )
            )
        )
        r2_train.append(
            r2_score(train_results_df["predicted_mu"], train_results_df["T"])
        )

        # test:
        rmse_test.append(
            np.sqrt(
                mean_squared_error(
                    test_results_df["predicted_mu"], test_results_df["T"]
                )
            )
        )
        mae_test.append(
            (mean_absolute_error(test_results_df["predicted_mu"], test_results_df["T"]))
        )
        r2_test.append(r2_score(test_results_df["predicted_mu"], test_results_df["T"]))

    except:
        success = False

    k.clear_session()
    # del model

    if success == False:
        print("Failed")
        return 1e4, 1e4, False  # not successful

    # registering results
    # results['fold'] = np.arange(cfg['cv'])
    results["rmse_train"] = rmse_train
    results["mae_train"] = mae_train
    results["r2_train"] = r2_train
    results["std_train"] = np.sum(np.std(total_res_train))
    results["rmse_test"] = rmse_test
    results["mae_test"] = mae_test
    results["r2_test"] = r2_test
    results["std_test"] = np.std(total_res_test)
    results["net_cfg"] = json.dumps(net_cfg)

    print(results)

    # return (
    #     model,
    #     train_results_df,
    #     test_results_df,
    #     test_x_orig,
    #     test_y_orig,
    #     scaler,
    #     train_x,
    #     test_x,
    #     total_res_train,
    # )

    if os.path.isfile(file):
        results.to_csv("./" + file, mode="a", index=False, header=False)
    else:
        results.to_csv("./" + file, mode="w", index=False, header=True)

    if np.isfinite(results["rmse_test"].mean()) and np.isfinite(
        results["std_test"].mean()
    ):
        return results["rmse_test"].mean(), results["std_test"].mean(), True
    else:
        return 1e4, 1e4, False  # not successful
    # end = time.time()
    # print(f'Elapsed time: {(end - start) / 60} minutes')


# system arguments (configuration)
if len(sys.argv) > 2 and sys.argv[1] == "--cfg":
    cfg = eval(sys.argv[2])
    if len(sys.argv) > 3:
        gpu = sys.argv[3]

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        physical_devices = tf.config.list_physical_devices("GPU")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    print(obj_function(cfg, None))
    k.clear_session()
