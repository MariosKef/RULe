# various
import os

gpus = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
str_available_gpus = [str(gpu) for gpu in gpus]
str_available_gpus = ",".join(str_available_gpus)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str_available_gpus
import subprocess
from subprocess import STDOUT, check_output
import re
import numpy as np
import traceback
import time


# Mipego
from mipego import ParallelBO, BO
from mipego.Surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

from objective import obj_function
import GPUtil as gp

# from objective_total_var import obj_function

reserved_gpus = list(np.arange(gpus[-1] + 1, 20))


class obj_func(object):
    def __init__(self, program):
        self.program = program

    def __call__(self, cfg):
        global reserved_gpus
        print(f" Reserved GPUs: {reserved_gpus}")
        available_gpus = gp.getAvailable(limit=10, excludeID=reserved_gpus)
        print(f"available gpus {available_gpus}")
        gpu = np.random.choice(available_gpus, replace=False)
        print(f" selected gpu:{gpu}")
        reserved_gpus.append(gpu)
        cmd = ["python3", self.program, "--cfg", str(cfg), str(gpu)]
        outs = ""
        outputval = 1e4
        try:
            # we use a timeout to cancel very long evaluations.
            outs = str(check_output(cmd, stderr=STDOUT, timeout=40000, encoding="utf8"))
            print(f"outs: {outs}")
            outs = eval(outs.split("\n")[-2])
            print(f"outs after splitting: {outs}")

            outputval = float(outs[0])
            print(f"outputval: {outputval}")

            if np.isnan(outputval).any():
                outputval = 1e4
        except subprocess.CalledProcessError as e:
            # exception handling
            traceback.print_exc()
            print(e.output)
        except:
            print("Unexpected error:")
            traceback.print_exc()
            outputval = 1e4
        # print(f" Reserved GPUs after execution: {reserved_gpus}")
        # print("\n")
        reserved_gpus.remove(gpu)
        # print(f" Unreserved GPUs: {reserved_gpus}")
        print("\n")
        return outputval


def main():

    # objective function
    objective = obj_func("objective.py")

    # Hyperparameter configuration
    # Pre-processing
    max_time = OrdinalSpace([20, 50], "max_time")  # maximum lookback
    percentage = OrdinalSpace([25, 75], "percentage")
    rul = OrdinalSpace([110, 130], "rul")
    rul_style = NominalSpace(["linear", "nonlinear"], "rul_style")

    # General training
    # lr_rate = ContinuousSpace([1e-4, 1.0e-1], "lr")  # learning rate
    lr_rate = NominalSpace(
        ["1e-1", "1e-2", "1e-3", "1e-4", "2e-5", "1e-5"], "lr"
    )  # learning rate
    batch_size = NominalSpace(["32", "64", "128"], "batch")

    activations = ["tanh", "sigmoid"]  # activations of recurrent layers
    final_activations = ["softplus", "exp"]  # output activations

    # Recurrent layers
    num_rec = OrdinalSpace([1, 3], "num_rec")  # maximum number of recurrent layers
    neurons = (
        OrdinalSpace([10, 100], "neuron") * num_rec._ub[0]
    )  # number of neurons of RNN layers
    acts_rec = (
        NominalSpace(activations, "activation_rec") * num_rec._ub[0]
    )  # activations of RNN layers

    rec_dropout_norm = (
        ContinuousSpace([1e-5, 0.9], "rec_dropout_norm") * num_rec._ub[0]
    )  # normal dropout for RNN

    rec_dropout = (
        ContinuousSpace([1e-5, 0.9], "recurrent_dropout") * num_rec._ub[0]
    )  # recurrent dropout

    # Dense layers
    num_den = OrdinalSpace([1, 3], "num_den")  # maximum number of dense layers
    neurons_den = (
        OrdinalSpace([10, 100], "neuron_den") * num_den._ub[0]
    )  # number of neurons of Dense layers
    acts_den = (
        NominalSpace(activations, "activation_den") * num_den._ub[0]
    )  # activations of recurrent layers

    den_dropout_norm = (
        ContinuousSpace([1e-5, 0.9], "dropout") * num_den._ub[0]
    )  # normal dropout

    f_acts = (
        NominalSpace(final_activations, "final_activation") * 2
    )  # final activations. The "2" because we have 2
    # outputs

    search_space = (
        num_rec
        + max_time
        + neurons
        + acts_rec
        + rec_dropout_norm
        + rec_dropout
        + f_acts
        + percentage
        + rul
        + rul_style
        + lr_rate
        + batch_size
        + num_den
        + neurons_den
        + acts_den
        + den_dropout_norm
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
    # model2 = RandomForest(levels=search_space.levels)

    # now define the optimizer.
    opt = ParallelBO(
        search_space=search_space,
        obj_fun=objective,
        model=model1,
        minimize=True,
        max_FEs=300,
        acquisition_fun="MGFI",
        DoE_size=100,
        n_point=10,
        n_job=10,
        verbose=True,
        random_seed=42,
        logger="log_file_single_objective_dataset_1_2_3_retake.txt",
        eval_type="dict",
    )

    # run
    # opt.run()
    xopt, fopt, stop_dict = opt.run()
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

    return xopt, fopt, stop_dict


if __name__ == "__main__":
    # General hyperparameters
    # cfg = {'cv': 2, 'shuffle': True,
    #    'random_state': 21,
    #    'mask_value': -99,
    #    'reps': 30,
    #    'epochs': 2,
    #    'batches': 64}

    # incumbent = main()

    start = time.time()
    xopt, fopt, stop_dict = main()
    print("xopt: {}".format(xopt))
    print("fopt: {}".format(fopt))
    print("stop criteria: {}".format(stop_dict))
    end = time.time()
    print(f"Elapsed time: {(end-start)/60} minutes")
