import os
import onnx
from onnx import helper
import math
import copy
from parallel_util import *
import time
from functools import reduce

lat_interval = 0.001

def gen_execution_plan(phy_stages):
    """
    1. partition each physical stage, get execution stages
    2. generate execution plan
    """
    func_limit = hyper_params['func_limit']
    step = hyper_params['step']
    def gen_opt_plan_recursive(stages, main_func_limit):
        if len(stages) == 0:
            return []

        required_mem = sum([ s.get_graph_size() for s in stages]) 
        if required_mem <= main_func_limit:
            # w/o limit for main function
            exe_stages = []
            for s in stages:
                exe_stages += copy.deepcopy(partition_stage_node(s, main_func_limit))
            return concat_exe_stages(exe_stages)
            return exe_stages
        else:
            test_stage = stages[-1]
            opt_lat = float("inf")
            opt_cost = float("inf")
            opt_stages = []

            for main_size in range(0, main_func_limit + 1, step):
                if main_size >= test_stage.get_graph_size() + step:
                    break
                exe_stages = gen_opt_plan_recursive(stages[:-1], main_func_limit - main_size) + copy.deepcopy(partition_stage_node(test_stage, main_size))
                exe_stages = concat_exe_stages(exe_stages)
                plan = ExecutionPlan(exe_stages)
                test_lat, test_cost = plan.cal_latency(), plan.cal_cost()
                if ( abs(test_lat - opt_lat) < lat_interval and test_cost < opt_cost ) or ( test_lat < opt_lat ):
                    opt_stages = exe_stages
                    opt_lat = test_lat
                    opt_cost = test_cost

            return opt_stages
    exe_stages = gen_opt_plan_recursive(phy_stages, func_limit)
    return ExecutionPlan(exe_stages)

stage_opt_notebook = {}
def partition_stage_node(phy_stage, main_func_limit):
    key = (phy_stage.id, main_func_limit)
    if key in stage_opt_notebook:
        return stage_opt_notebook[key]
    
    if len(phy_stage.graph.get_layers()) > 1: 
        # inter-layer
        opt_res = inter_layer_parallel(phy_stage, main_func_limit)
    else: 
        # intra-layer
        opt_res = intra_layer_parallel(phy_stage, main_func_limit)
    stage_opt_notebook[key] = opt_res
    return opt_res

def compare_with_opt(opt_lat, opt_cost, opt_choice, choice, plan):
    lat, cost = plan.cal_latency(), plan.cal_cost()
    if (abs(lat - opt_lat) < lat_interval and cost < opt_cost) or (lat < opt_lat):
        opt_lat = lat
        opt_cost = cost
        opt_choice = choice
    return opt_lat, opt_cost, opt_choice

param_opt_notebook = {}
def get_param_par_from_notebook(layer, main_func_limit):
    key = (layer.name, main_func_limit)
    if key in param_opt_notebook:
        # TODO: if the value is None
        return param_opt_notebook[key]
        
    opt_lat = float("inf")
    opt_cost = float("inf")
    opt_choice = None
    for c in num_choice:
        tmp_stage = param_partition(layer, c, main_func_limit)
        if tmp_stage:
            opt_lat, opt_cost, opt_choice = compare_with_opt(opt_lat, opt_cost, opt_choice, c, ExecutionPlan([tmp_stage]))
    
    param_opt_notebook[key] = (opt_lat, opt_cost, opt_choice)
    return (opt_lat, opt_cost, opt_choice)

def intra_layer_parallel(stage, main_func_limit):
    layer = stage.graph.get_layers()[0]
    mem_size = float("inf") if main_func_limit >= layer.size else main_func_limit

    if layer.size == 0: # ignore layers that have 0 size and 0 latency
        new_layer_graph = LayerInfoGraph(copy.deepcopy(layer.layer_nodes))
        return [ExecutionStage([layer.name, layer.name], 1, [new_layer_graph], [0], [0], mem_size, stage_type_param)]

    opt_lat = float("inf")
    if True in layer.dimension: # attribute partition
        opt_lat, opt_cost, opt_choice = get_attr_par_from_notebook(stage.graph, [layer], mem_size)

    opt_param_lat, opt_param_cost, opt_param_choice = get_param_par_from_notebook(layer, mem_size)
    exe_stage = param_partition(layer, opt_param_choice, mem_size) if opt_param_lat < opt_lat else attr_partition(stage.graph, [layer], opt_choice, mem_size)
    return [exe_stage]

attr_opt_notebook = {}
def get_attr_par_from_notebook(graph, layers, main_func_limit):
    key = (layers[0].name, layers[-1].name, main_func_limit)
    if key in attr_opt_notebook:
        return attr_opt_notebook[key]

    opt_lat = float("inf")
    opt_cost = float("inf")
    opt_choice = None
    for c in shape_choice:
        tmp_stage = attr_partition(graph, layers, c, main_func_limit)
        if tmp_stage:
            opt_lat, opt_cost, opt_choice = compare_with_opt(opt_lat, opt_cost, opt_choice, c, ExecutionPlan([tmp_stage]))

    attr_opt_notebook[key] = (opt_lat, opt_cost, opt_choice)
    return (opt_lat, opt_cost, opt_choice)

layer_opt_note = {}
def inter_layer_parallel(stage, main_func_limit):
    graph = stage.graph
    layers = graph.get_layers()
    
    step = hyper_params['step']

    def get_opt_plan(layer_i, layer_j, all_main_size):
        note_key = (layer_i, layer_j, all_main_size)
        if note_key in layer_opt_note:
            return layer_opt_note[note_key]

        model_size = sum([ l.size for l in layers[layer_i : layer_j+1]])
        if layer_i == layer_j:
            mem_size = float("inf") if all_main_size >= model_size else all_main_size
            opt_lat, opt_cost, opt_choice = get_attr_par_from_notebook(graph, [layers[layer_i]], mem_size)
            res = (opt_lat, opt_cost, [(layer_i, layer_j, opt_choice, mem_size)])
            layer_opt_note[note_key] = res
            return res

        elif layer_i < layer_j:
            final_opt_lat = float("inf")
            final_opt_cost = float("inf")
            final_plan = None
            for k in range(layer_i, layer_j + 1):
                if all_main_size >= model_size:
                    fuse_res = get_attr_par_from_notebook(graph, layers[layer_i : k + 1], float("inf"))
                    other_opt_res = get_opt_plan(k + 1, layer_j, float("inf"))
                    total_lat, total_cost = fuse_res[0] + other_opt_res[0], fuse_res[1] + other_opt_res[1]
                    if (abs(total_lat - final_opt_lat) < lat_interval and total_cost < final_opt_cost) or ( total_lat < final_opt_lat):
                        final_opt_lat = total_lat
                        final_opt_cost = total_cost
                        final_plan = [(layer_i, k, fuse_res[2], float("inf"))] + other_opt_res[2]
                else:
                    for fuse_mem_size in range(0, all_main_size + 1, step):
                        fuse_res = get_attr_par_from_notebook(graph, layers[layer_i : k + 1], fuse_mem_size)
                        other_opt_res = get_opt_plan(k + 1, layer_j, all_main_size - fuse_mem_size)
                        total_lat, total_cost = fuse_res[0] + other_opt_res[0], fuse_res[1] + other_opt_res[1]
                        if (abs(total_lat - final_opt_lat) < lat_interval and total_cost < final_opt_cost) or ( total_lat < final_opt_lat):
                            final_opt_lat = total_lat
                            final_opt_cost = total_cost
                            final_plan = [(layer_i, k, fuse_res[2], fuse_mem_size)] + other_opt_res[2]

            layer_opt_note[note_key] = (final_opt_lat, final_opt_cost, final_plan)
            return (final_opt_lat, final_opt_cost, final_plan)
        else:
            return (0, 0, [])

    res = get_opt_plan(0, len(layers) - 1, main_func_limit)
    logging.info(f'parallelize layers from 0 to {len(layers) - 1}. mem limit: {main_func_limit}, results: {res}')

    stage_list = []
    for i in res[2]:
        stage_list.append(attr_partition(graph, layers[i[0]: i[1] + 1], i[2], i[3]))

    return concat_exe_stages(stage_list)
    # return stage_list

def gen_plan(info_graph):
    start = time.time()

    phy_stages = gen_physical_stage_dag(info_graph)
    plan = gen_execution_plan(phy_stages)

    duration = int((time.time() - start) * 1000)
    logging.info(f'Solution generated. duration: {duration}')
    logging.info(f'plan latency: {plan.cal_latency()}, plan cost: {plan.cal_cost()}. plan: {plan}')

    stage_latencies = []
    aggre_lat = 0
    for s in plan.stage_list:
        lat = ExecutionPlan([s]).cal_latency()
        aggre_lat += lat
        stage_latencies.append(aggre_lat)
        if backend_no == 1: # KNIX backend
            if len(s.func_ids) > 1 and s.func_ids[0] == 0:
                s.func_ids = [ i + 1 for i in s.func_ids]

    logging.info(f'stage latency: {stage_latencies}')
    return plan

def test_model(name):
    info_graph = get_info_graph(name)
    plan = gen_plan(info_graph)
    return plan