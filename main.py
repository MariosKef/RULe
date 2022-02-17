# various
import os

available_gpus = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
str_available_gpus = [str(gpu) for gpu in available_gpus]
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
from mipego_multi import mipego
from mipego_multi.Surrogate import RandomForest
from mipego_multi.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace


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

    search_space = (
        num_rec
        * max_time
        * neurons
        * acts_rec
        * rec_dropout_norm
        * rec_dropout
        * f_acts
        * percentage
        * rul
        * rul_style
        * lr_rate
        * batch_size
        * num_den
        * neurons_den
        * acts_den
        * den_dropout_norm
    )

    # # Uncomment for debugging purposes.
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

    model1 = RandomForest(levels=search_space.levels)
    model2 = RandomForest(levels=search_space.levels)

    ignore_gpu = np.arange(available_gpus[-1] + 1, 20).tolist()

    print(ignore_gpu)

    # Optimizer contructor
    opt = mipego(
        search_space,
        objective,
        model1,
        second_surrogate=model2,
        minimize=True,
        max_eval=300,
        infill="HVI",
        n_init_sample=100,
        n_point=1,
        n_job=1,
        optimizer="MIES",
        verbose=True,
        random_seed=42,
        available_gpus=available_gpus,
        ignore_gpu=ignore_gpu,
        bi_objective=True,
        log_file="./log_file_28_1.txt",
    )

    # run
    opt.run()


if __name__ == "__main__":

    start = time.time()
    main()
    end = time.time()
    print(f"Elapsed time: {(end-start)/60} minutes")
