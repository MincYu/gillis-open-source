import os
import onnx
from onnx import helper
import math
import copy
import mxnet.contrib.onnx as onnx_mxnet
from parallel_util import *

def partition_plan(plan, graph, dir_name, export_mxnet_params=False):

    for stage in plan.stage_list:
        start_i, end_i = stage.name[0], stage.name[1]
        shape = stage.partition_shape
        stage_type = stage.stage_type
        ids = stage.func_ids
        if stage_type == stage_type_attr:
            partition_graphs = partition_parallel_layers(graph, (start_i, end_i), shape)
        elif stage_type == stage_type_param:
            if shape == 1:
                partition_graphs = partition_parallel_layers(graph, (start_i, end_i), (1, 1))
            else:
                partition_graphs = partition_single_layer(graph, start_i, shape)
        for m_id, m in zip(ids, partition_graphs):
            m_name = '{}_{}_{}'.format(start_i, end_i, m_id)
            m_input = m.first_layer.inputs[0]
            # print(f'exporting {m_name} ...')
            print(f'\'{m_name}\': ({get_shape(m_input)}, \'{m_input.name}\'),')
            onnx_model_file = f'{dir_name}/{m_name}.onnx'
            onnx.save(m.to_onnx_model(), onnx_model_file)
            sym, arg, aux = onnx_mxnet.import_model(onnx_model_file)
            sym.save(f'{dir_name}/{m_name}.json')
            if export_mxnet_params:
                mod = mx.mod.Module(symbol=sym, context=mx.cpu(), data_names=[m_input.name], label_names=None)
                mod.bind(for_training=False, data_shapes=[(m_input.name, get_shape(m_input))], label_shapes=None)
                mod.set_params(arg_params=arg, aux_params=aux, allow_missing=True, allow_extra=True)
                mod.save_params(f'{dir_name}/{m_name}.params')

            # os.remove(onnx_model_file)

def partition_single_layer(graph, layer, partition_number):
    layers = graph.get_hybrid_nodes(True)
    partitoned_layer = layers[layer]
    partition_graphs = []

    res_layers = parallel_single_layer(partitoned_layer, partition_number, True)
    for new_layers, _ in res_layers:
        partition_graph = LayerGraph(new_layers)
        partition_graphs.append(partition_graph)

    return partition_graphs

def partition_parallel_layers(graph, partition_range, partition_shape):
    layers = graph.get_hybrid_nodes(True)
    start_i, end_i = partition_range[0], partition_range[1]

    partition_layers = layers[start_i:end_i + 1]
    partition_graphs = []

    res_layers = parallel_layers(graph, partition_layers, partition_shape, True)
    for sub_layers, _ in res_layers:
        partition_graph = LayerGraph(sub_layers)
        partition_graphs.append(partition_graph)

    return partition_graphs

def model_partition(name, plan, work_dir):
    model_path = f'{model_dir}/{name}'
    graph = load_onnx_model(model_path)

    partition_dir = f'{work_dir}/models'
    if create_dir(partition_dir):
        partition_plan(plan, graph, partition_dir, export_mxnet_params=True)