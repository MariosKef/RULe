gpus = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
str_available_gpus = [str(gpu) for gpu in gpus]
str_available_gpus = ",".join(str_available_gpus)
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str_available_gpus

import GPUtil as gp
import numpy as np
import time
import subprocess
from subprocess import STDOUT, check_output
import traceback
from joblib import Parallel, delayed


reserved_gpus = list(np.arange(gpus[-1] + 1, 20))


class A(object):
    def __init__(self, program) -> None:
        self.program = program

    def __call__(self):
        global reserved_gpus
        print(f" Reserved GPUs: {reserved_gpus}")
        available_gpus = gp.getAvailable(limit=10, excludeID=reserved_gpus)
        print(f"available gpus {available_gpus}")
        gpu = np.random.choice(available_gpus, replace=False)
        print(f"selected gpu {gpu}")
        print("\n")
        reserved_gpus.append(gpu)
        cmd = ["python3", self.program]
        outs = ""
        outputval = 1e4
        try:
            # we use a timeout to cancel very long evaluations.
            outs = str(check_output(cmd, stderr=STDOUT, timeout=40000, encoding="utf8"))
            outputval = outs
        except subprocess.CalledProcessError as e:
            # exception handling
            traceback.print_exc()
            print(e.output)
        except:
            print("Unexpected error:")
            traceback.print_exc()
            outputval = 1e4
        # reserved_gpus.remove(gpu)
        return outputval


def main():

    # TODO: Parallelize this
    # for i in range(10):
    #     obj = A("./toy_callable.py")
    #     outputval = obj()
    #     print(f"i: {i} = {outputval}")
    #     # time.sleep(5)
    obj = A("./toy_callable.py")
    outputval = Parallel(n_jobs=1)(delayed(obj)() for _ in range(10))
    print(outputval)


if __name__ == "__main__":
    print(f'os.environ["CUDA_VISIBLE_DEVICES"]: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    print(f"Available GPUs: {gp.getAvailable(limit=10)})")
    print("\n")
    main()
