# An End-to-End Pipeline for Uncertainty Quantification and Remaining Useful Life Estimation: An Application on Aircraft Engines

![MIT License](https://img.shields.io/github/license/MariosKef/RULe?style=plastic) 
[![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)](https://www.python.org/downloads/release/python-388/)


## Introduction

This repository holds the source code used for the work on the paper *An End-to-End Pipeline for Uncertainty Quantification and Remaining Useful Life Estimation: An Application on Aircraft Engines*

This repository is the work of Marios Kefalas, PhD candidate at the Leiden Institute of Advanced Computer Science (LIACS), Leiden University, Leiden, The Netherlands.

## Installation
To install the pipeline you can do the following:
* Clone the repository 

```git clone https://github.com/MariosKef/RULe.git ~/rule```

* cd to the directory of the cloned repository

``` cd ~/rule```

* Create the environment and install the requirements

```\<path to python 3.8 binaries\> -m venv rule_env``` or 
```virtualenv --python=\<path to python 3.8 binaries\> rule_env```

Activate the environment:

```source rule_env/bin/activate``` (Linux/Unix)

```source rule_env/Scripts/activate``` (Windows)

Install the requirements:

```python -m pip install -r requirements.txt```

Install locally to work on your own version:

```cd ~/rule```

```python -m pip install -e .```

## Usage
* For the hyperparameter optimization (HPO), run:

``` cd ./RULe ```

``` python main.py ``` (note: be sure to update the ```log_file``` in the contructor of mipego inside the main.py with the path of your choice and the ```file``` variable for the results file in the objective.py)

* For the training of the full model, run:

``` cd ./RULe ```

``` python model_training_full.py 100 ``` (note: be sure to update the ```net_cfg``` with the cfg of your choice from the HPO (previous step). The extra command line argument (here 100) indicates the number of training epochs.)

## Acknowledgements 
This work is part of the research programme Smart Industry SI2016 with project name CIMPLO and project number 15465, which is partly financed by the Netherlands Organisation for Scientific Research (NWO).
