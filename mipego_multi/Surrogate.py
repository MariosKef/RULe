#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:48:14 2017

@author: Hao Wang
@email: wangronin@gmail.com
"""
from __future__ import print_function
import pdb

import pandas as pd
import numpy as np
from numpy import std, array, atleast_2d

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble._base import _partition_estimators
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import r2_score

from joblib import Parallel, delayed

# this function has to be globally visible
def save(predict, X, index, out):
    out[:, index] = predict(X, check_input=False)

class RandomForest(RandomForestRegressor):
    """
    Extension on the sklearn RandomForestRegressor class
    Added functionality: empirical MSE of predictions
    """
    def __init__(self, levels=None,n_estimators=100, workaround=False, **kwargs):
        """
        parameter
        ---------
        levels : dict, for categorical inputs
            keys: indices of categorical variables
            values: list of levels of categorical variables
        """
        super(RandomForest, self).__init__(**kwargs)
        
        self.n_estimators=n_estimators #now able to set number of trees in forest
        self.workaround = workaround
        if levels is not None and not workaround:
            assert isinstance(levels, dict)
            self._levels = levels
            self._cat_idx = sorted(self._levels.keys())
            self._n_values = [len(self._levels[i]) for i in self._cat_idx]
            # encode categorical variables to integer type
            self._le = [LabelEncoder().fit(self._levels[i]) for i in self._cat_idx]
            # encode integers to binary
            _max = max(self._n_values)
            data = atleast_2d([list(range(n)) * (_max // n) + \
                list(range(_max % n)) for n in self._n_values]).T
            self._enc = OneHotEncoder(sparse=False)
            self._enc.fit(data)
            # TODO: using such encoding, feature number will increase drastically
            # TODO: investigate the upper bound (in the sense of cpu time)
            # for categorical levels/variable number
            # in the future, maybe implement binary/multi-value split
        else:
            assert isinstance(levels, dict)
            self._levels = levels
            self._cat_idx = sorted(self._levels.keys())
            self._n_values = [len(self._levels[i]) for i in self._cat_idx]
            # encode categorical variables to integer type
            self._le = [LabelEncoder().fit(self._levels[i]) for i in self._cat_idx]
            # encode integers to binary
            _max = max(self._n_values)
            data = atleast_2d([list(range(n)) * (_max // n) + \
                               list(range(_max % n)) for n in self._n_values]).T
            #self._enc = [LabelEncoder().fit(self._levels[i]) for i in self._cat_idx]

    def _check_X(self, X):
        # X_ = array(X, dtype=object)
        X_ = atleast_2d(X)
        if hasattr(self, '_levels'):
            X_cat = array([self._le[i].transform(X_[:, k]) for i, k in enumerate(self._cat_idx)]).T
            if (not self.workaround):
                X_cat = self._enc.transform(X_cat)
            X = np.c_[np.delete(X_, self._cat_idx, 1).astype(float), X_cat]
        return X

    def fit(self, X, y):
        X = self._check_X(X)
        self.y = y
        return super(RandomForest, self).fit(X, y)

    def predict(self, X, eval_MSE=False):
        check_is_fitted(self, 'estimators_')
        # Check data
        X = self._check_X(X)
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat_all = np.zeros((X.shape[0], self.n_outputs_, self.n_estimators), dtype=np.float64)
        else:
            y_hat_all = np.zeros((X.shape[0], self.n_estimators), dtype=np.float64)

        # Parallel loop
        Parallel(n_jobs=n_jobs, verbose=self.verbose, backend="threading")(
            delayed(save)(e.predict, X, i, y_hat_all) for i, e in enumerate(self.estimators_))

        y_hat = np.mean(y_hat_all, axis=1).flatten()
        if eval_MSE:
            sigma2 = np.std(y_hat_all, axis=1, ddof=1) ** 2.
            sigma2 = sigma2.flatten()
        return (y_hat, sigma2) if eval_MSE else y_hat


# import rpy2.robjects as ro
# from rpy2.robjects.packages import importr
# from rpy2.robjects import r, pandas2ri, numpy2ri

# numpy and pandas data type conversion to R
# numpy2ri.activate()
# pandas2ri.activate()

# class RrandomForest(object):
#     """
#     Python wrapper for the R 'randomForest' library for regression
#     TODO: verify R randomForest uses CART trees instead of C45...
#     """
#     def __init__(self, levels=None, n_estimators=10, max_features='auto',
#                  min_samples_leaf=1, max_leaf_nodes=None, importance=False,
#                  nPerm=1, corr_bias=False, seed=None):
#         """
#         parameter
#         ---------
#         levels : dict
#             dict keys: indices of categorical variables
#             dict values: list of levels of categorical variables
#         seed : int, random seed
#         """
#         if max_leaf_nodes is None:
#             max_leaf_nodes = ro.NULL

#         if max_features == 'auto':
#             mtry = 'p'
#         elif max_features == 'sqrt':
#             mtry = 'int(np.sqrt(p))'
#         elif max_features == 'log':
#             mtry = 'int(np.log2(p))'
#         else:
#             mtry = max_features

#         self.pkg = importr('randomForest')
#         self._levels = levels
#         self.param = {'ntree' : int(n_estimators),
#                       'mtry' : mtry,
#                       'nodesize' : int(min_samples_leaf),
#                       'maxnodes' : max_leaf_nodes,
#                       'importance' : importance,
#                       'nPerm' : int(nPerm),
#                       'corr_bias' : corr_bias}

#         # make R code reproducible
#         if seed is not None:
#             r['set.seed'](seed)

#     def _check_X(self, X):
#         """
#         Convert all input types to R data.frame
#         """
#         if isinstance(X, list):
#             if isinstance(X[0], list):
#                 X = array(X, dtype=object)
#             else:
#                 X = array([X], dtype=object)
#         elif isinstance(X, np.ndarray):
#             if hasattr(self, 'columns'):
#                 if X.shape[1] != len(self.columns):
#                     X = X.T
#         elif isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
#             X = X.values

#         # be carefull: categorical columns should be converted as FactorVector
#         to_r = lambda index, column: ro.FloatVector(column) if index not in self._levels.keys() else \
#             ro.FactorVector(column, levels=ro.StrVector(self._levels[index]))
#         d = {'X' + str(i) : to_r(i, X[:, i]) for i in range(X.shape[1])}
#         X_r = ro.DataFrame(d)

#         return X_r

#     def fit(self, X, y):
#         self.X = self._check_X(X)
#         y = array(y).astype(float)

#         self.columns = numpy2ri.ri2py(self.X.colnames)
#         n_sample, self.n_feature = self.X.nrow, self.X.ncol

#         if isinstance(self.param['mtry'], basestring):
#             p = self.n_feature
#             self.param['mtry'] = eval(self.param['mtry'])

#         self.rf = self.pkg.randomForest(x=self.X, y=y, **self.param)
#         return self

#     def predict(self, X, eval_MSE=False):
#         """
#         X should be a dataframe
#         """
#         X = self._check_X(X)
#         _ = self.pkg.predict_randomForest(self.rf, X, predict_all=eval_MSE)

#         if eval_MSE:
#             y_hat = numpy2ri.ri2py(_[0])
#             mse = std(numpy2ri.ri2py(_[1]), axis=1, ddof=1) ** 2.
#             return y_hat, mse
#         else:
#             return numpy2ri.ri2py(_)

if __name__ == '__main__':
    # TODO: this part goes into test 
    # simple test for mixed variables...
    np.random.seed(12)

    n_sample = 110
    levels = ['OK', 'A', 'B', 'C', 'D', 'E']
    X = np.c_[np.random.randn(n_sample, 2).astype(object),
              np.random.choice(levels, size=(n_sample, 1))]
    y = np.sum(X[:, 0:-1] ** 2., axis=1) + 5 * (X[:, -1] == 'OK')

    X_train, y_train = X[:100, :], y[:100]
    X_test, y_test = X[100:, :], y[100:]

    # sklearn-random forest
    rf = RandomForest(levels={2: levels}, max_features='sqrt')
    rf.fit(X_train, y_train)
    y_hat, mse = rf.predict(X_test, eval_MSE=True)

    print('sklearn random forest:')
    print('target :', y_test)
    print('predicted:', y_hat)
    print('MSE:', mse)
    print('r2:', r2_score(y_test, y_hat))
    print()

    # R randomForest
    rf = RrandomForest(levels={2: levels}, seed=1, max_features='sqrt')
    rf.fit(X_train, y_train)
    y_hat, mse = rf.predict(X_test, eval_MSE=True)

    print('R randomForest:')
    print('target :', y_test)
    print('predicted:', y_hat)
    print('MSE:', mse)
    print('r2:', r2_score(y_test, y_hat))

    # TODO: those settings should be in test file as inputs to surroagtes
    # leaf_size = max(1, int(n_sample / 20.))
    # ntree=100,
    # mtry=ceil(self.n_feature * 5 / 6.),
    # nodesize=leaf_size
