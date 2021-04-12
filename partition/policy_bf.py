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

choice_num = len(shape_choice) + 1 # the first one indicates whether to parallel

def record_all_recursive(pre_action, info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes):
    if len(pre_action) == layer_size:
        all_stage_range = gen_stage_range(pre_action, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes)
        plan = get_plan(info_graph, hybrid_layers, all_stage_range)
        latency, cost = plan.cal_latency(), plan.cal_cost()
        return [(latency, cost, pre_action)]

    all_records = []
    for c in range(choice_num):
        new_action = copy.deepcopy(pre_action)
        new_action.append(c)
        records = record_all_recursive(new_action, info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes)
        all_records += records
    
    return all_records

def gen_opt_action_recursive(pre_action, info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes, lat_threshold):
    if len(pre_action) == layer_size:
        all_stage_range = gen_stage_range(pre_action, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes)
        plan = get_plan(info_graph, hybrid_layers, all_stage_range)
        latency, cost = plan.cal_latency(), plan.cal_cost()
        if latency < lat_threshold:
            return latency, cost, pre_action
        else:
            return float('inf'), float('inf'), None

    opt_cost = float('inf')
    opt_latency = float('inf')
    opt_action = None

    for c in range(choice_num):
        new_action = copy.deepcopy(pre_action)
        new_action.append(c)
        ret_latency, ret_cost, ret_action = gen_opt_action_recursive(new_action, info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes, lat_threshold)
        if ret_cost < opt_cost:
            opt_cost = ret_cost
            opt_latency = ret_latency
            opt_action = ret_action
    
    return opt_latency, opt_cost, opt_action

def traverse_all(info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes, lat_threshold):
    """
    The optimal solution that travers all possible choices
    """
    require_mem_limit = info_graph.get_model_size() > hyper_params['func_limit']
    if require_mem_limit:
        return
    
    base_action = [0] * 6
    # opt_latency, opt_cost, opt_action = gen_opt_action_recursive(base_action, info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes, lat_threshold)
    # return opt_latency, opt_cost, opt_action
    
    pool = ProcessPoolExecutor(max_workers=choice_num)
    futures = []

    global all_records
    if not all_records:
        for i in range(choice_num):
            if not os.path.isfile(f'vgg11_{i}.pickle'):
                new_action = copy.deepcopy(base_action)
                new_action.append(i)
                print(f'submit traversing job for option {i}')
                fu = pool.submit(record_all_recursive, new_action, info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes)
                futures.append((i, fu))

        for i, fu in futures:
            res = fu.result()
            with open(f'vgg11_{i}.pickle', 'wb') as handle:
                pickle.dump(res, handle)
            print(f'action {i} saved')

        for i in range(choice_num):
            file_name = f'vgg11_{i}.pickle'
            with open(file_name, 'rb') as handle:
                print(f'Loading {file_name} ...')
                res = pickle.load(handle)
                all_records += res

    opt_latency, opt_cost, opt_action = float('inf'), float('inf'), None
    for lat, cost, action in all_records:
        if lat < lat_threshold and cost < opt_cost:
            opt_latency, opt_cost, opt_action = lat, cost, action
    return opt_latency, opt_cost, opt_action

'''
def random_sample(info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes, lat_threshold):
    """
    Sampling action follows a zip-f distribution (alpha = 0.5)
    """
    require_mem_limit = info_graph.get_model_size() > hyper_params['func_limit']

    no_par_prob = 0.9
    par_prob = (1 - no_par_prob) / (choice_num - 1)
    probs = np.array([no_par_prob])
    for _ in range(choice_num - 1):
        probs = np.append(probs, par_prob)

    opt_cost = float('inf')
    opt_latency = float('inf')
    opt_action = None
    probs = probs / probs.sum()

    mem_probs = np.array([0.5, 0.5])

    trial_num = 1000
    for _ in range(trial_num):
        actions = [np.random.choice(range(choice_num), p=probs) for _ in range(len(hybrid_layers))]
        all_stage_range = gen_stage_range(actions, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes)
        if require_mem_limit:
            mem_actions = [np.random.choice(range(2), p=mem_probs) for _ in range(len(all_stage_range))]
            all_stage_range = modify_stage_range(all_stage_range, mem_actions)

        plan = get_plan(info_graph, hybrid_layers, all_stage_range)
        if plan:
            latency, cost = plan.cal_latency(), plan.cal_cost()
            if latency < lat_threshold and cost < opt_cost:
                opt_cost = cost
                opt_latency = latency
                opt_action = (actions, mem_actions) if require_mem_limit else actions
    return opt_latency, opt_cost, opt_action
'''

def baseline_solution(info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes, lat_threshold):
    start = time.time()

    baseline_func = traverse_all
    opt_latency, opt_cost, opt_action = baseline_func(info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes, lat_threshold)

    duration = int((time.time() - start) * 1000)
    print(f'opt cost: {opt_cost} opt latency: {opt_latency} duration: {duration}')
    print("opt action: ", opt_action)
    return opt_latency, opt_cost, opt_action

def test_benchmark(benchmarks):
    global all_records
    all_records = []

    res = {}
    for name, lat_thresholds in benchmarks.items():
        global layer_size
        _, info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes = get_graph_metrics(name)
        layer_size = len(hybrid_layers)
        for lat_threshold in lat_thresholds:
            opt_latency, opt_cost, opt_action = baseline_solution(info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes, lat_threshold)
            res[(name, lat_threshold)] = (opt_latency, opt_cost, opt_action)
    
    for key, re in res.items():
        print(f'{key[0]:15} {key[1]:20} {re[0]:20} {re[1]:20}')
        # print(f'{key[0]}  {key[1]}  {re[0]}  {re[1]}  {re[2]}')
    return res

if __name__ == "__main__":
    # test
    test_benchmark({'vgg11.onnx': [500]})