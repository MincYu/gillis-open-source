import os
import onnx
from onnx import helper
import math
import copy
from parallel_util import *
import time
import pickle
from functools import reduce
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import warnings
from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import multivariate_normal
warnings.simplefilter("ignore", UserWarning)

choice_num = len(shape_choice) + 1 # the first one indicates whether to parallel

# Inspired by http://krasserm.github.io/2018/03/21/bayesian-optimization/
def expected_improvement(X, T, X_sample, Y_sample, gpr_cost, gpr_lat, xi=0.01):
    mu, sigma = gpr_cost.predict(X, return_std=True)
    lat_m, lat_cov = gpr_lat.predict(X, return_cov=True)

    sigma = sigma.reshape(-1, 1)
    
    lat_var = multivariate_normal(mean=lat_m, cov=lat_cov)
    delta = lat_var.cdf([T])
    # delta = 1 if lat_m < T else 0

    mu_sample_opt = np.max(Y_sample)
    with np.errstate(divide='ignore'):
        # imp = mu - mu_sample_opt - xi
        imp = mu_sample_opt - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei = delta * ei

        ei[sigma == 0.0] = 0.0

    return ei

def propose_location(acquisition, T, X_sample, Y_sample, gpr_cost, gpr_lat, bounds, n_restarts=10):
    dim = X_sample.shape[1]
    min_val = float('inf')
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), T, X_sample, Y_sample, gpr_cost, gpr_lat)
    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in sample(n_restarts):
    # for x0 in np.random.randint(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return np.floor(min_x.reshape(-1, 1).flatten()).astype(int)

def get_y_value(action):
    # print(action)
    all_stage_range = gen_stage_range(action, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes)
    if require_mem_limit:
        mem_action = [0] * len(all_stage_range)
        all_stage_range = modify_stage_range(all_stage_range, mem_action)
        action = (action, mem_action)
    plan = get_plan(info_graph, hybrid_layers, all_stage_range)
    if plan:
        latency, cost = plan.cal_latency(), plan.cal_cost()
        return latency, cost, action

no_par_prob = 0.9
par_prob = (1 - no_par_prob) / (choice_num - 1)
probs = np.array([no_par_prob])
for _ in range(choice_num - 1):
    probs = np.append(probs, par_prob)
probs = probs / probs.sum()

def sample(size):
    X_sample = []
    for _ in range(size):
        action = [np.random.choice(range(choice_num), p=probs) for _ in range(len(hybrid_layers))]
        X_sample.append(action)
    return np.array(X_sample)

def bayesian_optimization(T):
    # Gaussian process with Mat??rn kernel as surrogate model
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr_cost = GaussianProcessRegressor(kernel=m52)
    gpr_lat = GaussianProcessRegressor(kernel=m52)

    # Initialize samples
    X_sample = sample(10)
    sample_res = [ (s, get_y_value(s)) for s in X_sample]
    X_sample, sample_res = np.array([ s for s, v in sample_res if v]), [ v for s, v in sample_res if v]
    Y_sample, T_sample = [ c for l, c, _ in sample_res], [ l for l, c, _ in sample_res]
    Y_sample, T_sample = np.array(Y_sample), np.array(T_sample)
    bounds = np.array([0, choice_num-.0001])
    bounds = np.tile(bounds, (layer_size, 1))

    # Number of iterations
    n_iter = 60
    
    print('='*100)

    lat_cache = None
    lat, cost, action = float('inf'), float('inf'), None
    for i in range(n_iter):
        # Update Gaussian process with existing samples
        gpr_cost.fit(X_sample, Y_sample)
        gpr_lat.fit(X_sample, T_sample)

        # Obtain next sampling point from the acquisition function (expected_improvement)
        action = propose_location(expected_improvement, T, X_sample, Y_sample, gpr_cost, gpr_lat, bounds)
        
        # Obtain next noisy sample from the objective function
        res = get_y_value(action)
        if res:
            lat, cost, action = res

            print(f'iter {i}. action {action}. lat {lat}. cost {cost}.')

            if lat_cache and abs(lat - lat_cache) / lat_cache < 0.03:
                print('Find the similar latency. Break.')
                break
            lat_cache = lat
            
            X_sample = np.vstack((X_sample, action[0] if require_mem_limit else action))
            Y_sample = np.append(Y_sample, cost)
            T_sample = np.append(T_sample, lat)

    return lat, cost, action

def test_benchmark(benchmarks):
    global all_records
    all_records = []

    res = {}
    for name, lat_thresholds in benchmarks.items():
        global layer_size, info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes, require_mem_limit
        _, info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes = get_graph_metrics(name)
        require_mem_limit = info_graph.get_model_size() > hyper_params['func_limit']

        layer_size = len(hybrid_layers)
        for lat_threshold in lat_thresholds:
            opt_latency, opt_cost, opt_action = bayesian_optimization(lat_threshold)
            res[(name, lat_threshold)] = (opt_latency, opt_cost, opt_action)
    
    for key, re in res.items():
        print(f'{key[0]:15} {key[1]:20} {re[0]:20} {re[1]:20}')
        # print(f'{key[0]}  {key[1]}  {re[0]}  {re[1]}  {re[2]}')
    return res