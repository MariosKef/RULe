# various
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7,8"
import subprocess, sys
from subprocess import STDOUT, check_output
import re
import numpy as np
import traceback


# Mipego
from mipego import mipego
from mipego.Surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

from objective import obj_function

# from objective_total_var import obj_function


class obj_func(object):
    def __init__(self, program):
        self.program = program

    def __call__(self, cfg, gpu_no):
        print("calling program with gpu " + str(gpu_no))
        cmd = ["python3", self.program, "--cfg", str(cfg), str(gpu_no)]
        outs = ""
        outputval = [1e4, 1e4, False]
        try:
            # we use a timeout to cancel very long evaluations.
            outs = str(check_output(cmd, stderr=STDOUT, timeout=40000, encoding="utf8"))
            outs = eval(outs.split("\n")[-2])

            outputval = [float(outs[0]), float(outs[1]), bool(outs[2])]

            if np.isnan(outputval).any():
                outputval = [1e4, 1e4, False]
        except subprocess.CalledProcessError as e:
            # exception handling
            traceback.print_exc()
            print(e.output)
        except:
            print("Unexpected error:")
            traceback.print_exc()
            outputval = [1e4, 1e4, False]
        return outputval


def main():

    # objective function
    objective = obj_func("./objective.py")

    # hyperparameter configuration
    max_time = OrdinalSpace([20, 40], "max_time")  # maximum lookback
    # lr_rate = ContinuousSpace([1e-4, 1.0e-1], "lr")  # learning rate
    lr_rate = NominalSpace(
        ["1e-1", "1e-2", "1e-3", "1e-4", "2e-5", "1e-5"], "lr"
    )  # learning rate
    num_rec = OrdinalSpace([2, 4], "num_rec")  # maximum number of recurrent layers

    activations = ["tanh", "sigmoid"]  # activations of recurrent layers
    final_activations = ["softplus", "exp"]  # output activations
    neurons = OrdinalSpace([65, 100], "neuron") * num_rec._ub[0]  # number of neurons
    acts = (
        NominalSpace(activations, "activation") * num_rec._ub[0]
    )  # activations of recurrent layers
    dropout = ContinuousSpace([1e-5, 0.9], "dropout") * num_rec._ub[0]  # normal dropout
    rec_dropout = (
        ContinuousSpace([1e-5, 0.9], "recurrent_dropout") * num_rec._ub[0]
    )  # recurrent dropout
    f_acts = (
        NominalSpace(final_activations, "final_activation") * 2
    )  # final activations. The "2" because we have 2
    # outputs
    percentage = OrdinalSpace([60, 75], "percentage")
    rul = OrdinalSpace([120, 130], "rul")

    rul_style = NominalSpace(["linear", "nonlinear"], "rul_style")

    batch_size = NominalSpace(["32", "64", "128"], "batch")

    search_space = (
        num_rec
        * max_time
        * neurons
        * acts
        * dropout
        * rec_dropout
        * f_acts
        * percentage
        * rul
        * rul_style
        * lr_rate
        * batch_size
    )

    # values = search_space.sampling(1)
    # names = search_space.var_name
    # net_cfg = {}
    # for i in range(len(names)):
    #    net_cfg[names[i]] = values[0][i]

    # # Uncomment for debugging purposes.
    net_cfg = {
        "max_time": 100,
        "lr": "0.001",
        "num_rec": 3,
        "neuron_0": 100,
        "activation_0": "tanh",
        "dropout_0": 0.25,
        "recurrent_dropout_0": 0.25,
        "neuron_1": 50,
        "activation_1": "tanh",
        "dropout_1": 0.25,
        "recurrent_dropout_1": 0.25,
        "neuron_2": 20,
        "activation_2": "tanh",
        "dropout_2": 0.25,
        "recurrent_dropout_2": 0.25,
        "final_activation_0": "exp",
        "final_activation_1": "softplus",
        "percentage": 50,
        "rul": 115,
        "rul_style": "nonlinear",
        "batch": "128",
    }

    """
    self, search_space, obj_func, surrogate, second_surrogate=None, ftarget=None,
                 minimize=True, noisy=False, max_eval=None, 
                 infill='MGFI', t0=2, tf=1e-1, scheduI can call tomorrow morning. Would that bele=None,
                 n_init_sample=None, n_point=1, n_job=1, backend='multiprocessing',
                 n_restart=None, max_infill_eval=None, wait_iter=3, optimizer='MIES', 
                 log_file=None, data_file=None, verbose=False, random_seed=None,
                 available_gpus=[],bi_objective=False,
                 ref_time=3000.0, ref_loss=3.0, hvi_alpha=0.1, ignore_gpu=[],
                 **obj_func_params
    """

    # print(search_space.levels)

    model1 = RandomForest(levels=search_space.levels)
    model2 = RandomForest(levels=search_space.levels)

    available_gpus = [1, 2, 3, 4, 5, 6, 7, 8]
    ignore_gpu = np.append([0], np.arange(available_gpus[-1] + 1, 20)).tolist()

    # now define the optimizer.
    opt = mipego(
        search_space,
        objective,
        model1,
        second_surrogate=model2,
        minimize=True,
        max_eval=200,
        infill="HVI",
        n_init_sample=30,
        n_point=1,
        n_job=8,
        optimizer="MIES",
        verbose=True,
        random_seed=42,
        available_gpus=available_gpus,
        ignore_gpu=ignore_gpu,
        bi_objective=True,
        log_file="./log_file_26_11.txt",
    )

    # run
    opt.run()
    # incumbent, stop_dict = opt.run()
    # print(incumbent)

    # net_cfg = {
    #     "num_rec": 4,
    #     "max_time": 24,
    #     "neuron_0": 76,
    #     "neuron_1": 75,
    #     "neuron_2": 74,
    #     "neuron_3": 66,
    #     "activation_0": "tanh",
    #     "activation_1": "tanh",
    #     "activation_2": "sigmoid",
    #     "activation_3": "sigmoid",
    #     "dropout_0": 0.018692516794622607,
    #     "dropout_1": 0.8002018342665917,
    #     "dropout_2": 0.615094589188039,
    #     "dropout_3": 0.08230738757019833,
    #     "recurrent_dropout_0": 0.6421264747391056,
    #     "recurrent_dropout_1": 0.8933998465284962,
    #     "recurrent_dropout_2": 0.6402495109098905,
    #     "recurrent_dropout_3": 0.6693624215836003,
    #     "final_activation_0": "softplus",
    #     "final_activation_1": "softplus",
    #     "percentage": 62,
    #     "rul": 124,
    #     "rul_style": "nonlinear",
    #     "lr": "0.0008896860421074306",
    #     "batch": "128",
    # }
    # print(net_cfg)
    # (
    #     model,
    #     train_results_df,
    #     test_results_df,
    #     test_x_orig,
    #     test_y_orig,
    #     scaler,
    #     train_x,
    #     test_x,
    # ) = obj_function(net_cfg, cfg=None)

    # return (
    #     model,
    #     train_results_df,
    #     test_results_df,
    #     test_x_orig,
    #     test_y_orig,
    #     scaler,
    #     train_x,
    #     test_x,
    # )


if __name__ == "__main__":
    # General hyperparameters
    # cfg = {'cv': 2, 'shuffle': True,
    #    'random_state': 21,
    #    'mask_value': -99,
    #    'reps': 30,
    #    'epochs': 2,
    #    'batches': 64}

    # incumbent = main()
    main()
    # rmse, std =  obj_function(net_cfg, cfg)
    # print(incumbent)
