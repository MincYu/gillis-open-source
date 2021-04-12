import os
import onnx
from onnx import helper, TensorProto
import math
import copy
import json
from layer_graph import *
from functools import reduce
from mxnet import ndarray as F

hyper_params = {
    'func_limit': 1400,
    'step': 100,
}

# shape_choice = [(1, 1)] # for rnn models
shape_choice = [(1, 1), (1, 2), (2, 2)] # for simple cases or brute force
# shape_choice = [(1, 1), (1, 2), (1, 4), (2, 2), (2, 4)] # for general case
# shape_choice = [(1, 1), (1, 2), (1, 4), (2, 2)] # for google cloud quick test
num_choice = [1, 2, 4]
# num_choice = [1, 2, 4, 8]

def gen_plan_with_action(name, action):
    _, info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes = get_graph_metrics(name)
    if isinstance(action, tuple): # require mem limit
        stage_range = gen_stage_range(action[0], phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes)
        stage_range = modify_stage_range(stage_range, action[1])
    else:
        stage_range = gen_stage_range(action, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes)
    plan = get_plan(info_graph, hybrid_layers, stage_range)
    return plan

def get_info_graph(name):
    model_path = f'{model_dir}/{name}'

    # handle rnn with mock model graph
    if 'lstm' in name or 'rnn' in name:
        # rnn name format: lstm-{layer size}-{hidden size}
        layer_size = int(name.split('-')[1])
        hidden_size = int(name.split('-')[2])
        info_graph = RNNLayerInfoGraph(layer_size, hidden_size)
    else:
        graph = load_onnx_model(model_path)
        info_graph = LayerInfoGraph(graph)
    
    logging.info(f'{name} latency {model_latency(info_graph)}, size {info_graph.get_model_size()}')
    return info_graph

# SLO-aware solution utils
def get_graph_metrics(name):
    info_graph = get_info_graph(name)
    hybrid_layers = info_graph.get_hybrid_nodes()
    phy_stages = gen_physical_stage_dag(info_graph)
    embeddings, stage_layer_sizes = gen_layer_embedding(phy_stages)
    embeddings = embeddings.expand_dims(0).swapaxes(0, 1)
    aggre_stage_layer_sizes = []
    aggre_stage_size = 0
    for s in stage_layer_sizes:
        aggre_stage_size += s
        aggre_stage_layer_sizes.append(aggre_stage_size)

    aggre_layer_sizes = []
    aggre_layer_size = 0
    for i, l in enumerate(hybrid_layers):
        aggre_layer_size += l.size
        aggre_layer_sizes.append(aggre_layer_size)
    return embeddings, info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes
    
def gen_layer_embedding(stages):
    """
    layer embedding: stage id | layer size | one-hot
    """
    layer_embeddings = []
    stage_layer_sizes = [ len(s.graph.get_layers()) for s in stages]
    layer_size = sum(stage_layer_sizes)

    encode_stage = []
    encode_size = []
    encode_index = []
    encode_one_hot = []
    layer_index = 0
    for stage in stages:
        hybrid_nodes = stage.graph.get_layers()
        for node in hybrid_nodes:
            encode_stage.append(F.array([stage.id]))
            encode_size.append(F.array([node.size]))
            encode_index.append(F.array([layer_index]))
            layer_index += 1
    
    for i in encode_index:
        encode_one_hot.append(F.one_hot(i, layer_size))

    for stage, size, one_hot in zip(encode_stage, encode_size, encode_one_hot):
        layer_embeddings.append(F.concat(stage, size, one_hot[0], dim=0).expand_dims(0))
    return F.concat(*layer_embeddings, dim=0), stage_layer_sizes

def gen_stage_embedding(stage_ranges, layer_size):
    """
    stage embedding: stage id | partition index | stage size | belonging layers
    """
    stage_indexes = []
    partion_indexes = []
    stage_sizes = []

    layers_vectors = []
    for i, stage_range in enumerate(stage_ranges):
        stage_indexes.append(F.array([i]))
        partion_indexes.append(F.array([stage_range[2]]))

        stage_sizes.append(F.array([stage_range[3]]))

        layers_vector = [0] * stage_range[0] + [1] * (stage_range[1] - stage_range[0] + 1) + [0] * (layer_size - stage_range[1] - 1)
        layers_vectors.append(F.array(layers_vector))

    stage_embeddings = []
    for stage_index, partition_index, stage_size, layers_vector in zip(stage_indexes, partion_indexes, stage_sizes, layers_vectors):
        stage_embeddings.append(F.concat(stage_index, partition_index, stage_size, layers_vector, dim=0).expand_dims(0))
    return F.concat(*stage_embeddings, dim=0)

def modify_stage_range(all_range_actions, mem_actions):
    for i in range(len(all_range_actions)):
        if mem_actions[i] == 1: # whether outsourcing all computation to worker functions
            all_range_actions[i][-1] = 0
    return all_range_actions

def gen_stage_range(actions, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes):
    partition_plans = []
    
    if actions[-1] == 0: # the last layer
        actions[-1] = 1
    for i, a in enumerate(actions):
        if a > 0:
            partition_plans.append((i, a - 1))

    stage_idx = 0
    partition_idx = 0
    range_start_idx = 0
    all_range_actions = []

    mid_stages = False
    while partition_idx < len(partition_plans):
        
        if partition_plans[partition_idx][0] < aggre_stage_layer_sizes[stage_idx]:
            stage_type = stage_type_attr if len(phy_stages[stage_idx].graph.get_layers()) > 1 else stage_type_param
            action = partition_plans[partition_idx][1]
            if mid_stages and action > 0:
                pre_size = 0 if range_start_idx == 0 else aggre_layer_sizes[range_start_idx - 1]
                stage_size = aggre_layer_sizes[partition_plans[partition_idx][0] - 1] - pre_size
                all_range_actions.append([range_start_idx, partition_plans[partition_idx][0] - 1, 0, stage_size, stage_type, float("inf")])
                range_start_idx =  partition_plans[partition_idx][0]
                # action = 0
                # stage_type = stage_type_attr
            
            pre_size = 0 if range_start_idx == 0 else aggre_layer_sizes[range_start_idx - 1]
            stage_size = aggre_layer_sizes[partition_plans[partition_idx][0]] - pre_size

            all_range_actions.append([range_start_idx, partition_plans[partition_idx][0], action, stage_size, stage_type, float("inf")])
            range_start_idx = partition_plans[partition_idx][0] + 1
            partition_idx += 1
        elif partition_plans[partition_idx][0] >= aggre_stage_layer_sizes[stage_idx]:
            mid_stages = range_start_idx < aggre_stage_layer_sizes[stage_idx]
            stage_idx += 1
    return all_range_actions

def get_plan(info_graph, hybrid_layers, all_range_actions):
    
    exe_stages = []
    aggre_stage_mem_size = 0
    for begin_i, end_i, choice, stage_size, stage_type, mem_limit in all_range_actions:

        # manually adjust the limit based on the current mem used
        expect_mem = stage_size if stage_type == stage_type_attr else stage_size / num_choice[choice]
        aggre_stage_mem_size += stage_size
        if aggre_stage_mem_size > hyper_params['func_limit']:
            mem_limit = 0

        layers = hybrid_layers[begin_i : end_i + 1]
        if choice == 0: # no parallel
            exe_stage = attr_partition(info_graph, layers, shape_choice[choice], mem_limit)
        else:
            if stage_type == stage_type_attr:
                exe_stage = attr_partition(info_graph, layers, shape_choice[choice], mem_limit)
            else:
                exe_stage = param_partition(layers[0], num_choice[choice], mem_limit)
        if not exe_stage: # TODO exe_stage is none
            return None
        exe_stages.append(exe_stage)
    exe_stages = concat_exe_stages(exe_stages)
    plan = ExecutionPlan(exe_stages)
    return plan

def concat_exe_stages(stage_list):
    merged_stage_list = []
    stage = stage_list.pop(0)
    while len(stage_list) > 0:
        next_stage = stage_list.pop(0)
        if stage.able_to_merge(next_stage):
            stage.merge_stage(next_stage)
        else:
            merged_stage_list.append(stage)
            stage = next_stage
    merged_stage_list.append(stage)
    return merged_stage_list

def gen_physical_stage_dag(model_graph):
    # cur_node = model_graph.first_layer

    def compare_dim(pre_dim, cur_dim):
        if len(pre_dim) == len(cur_dim):
            return [ p and c for p, c in zip(pre_dim, cur_dim) ]
        else:
            return [False for _ in range(len(cur_dim))]

    hybrid_nodes = model_graph.get_hybrid_nodes()
    stages = []
    dim = hybrid_nodes[0].dimension
    layer_nodes = [hybrid_nodes[0]]
    count = 0
    for cur_node in hybrid_nodes[1:]:
        new_dim = compare_dim(dim, cur_node.dimension)
        if any(new_dim):
            layer_nodes.append(cur_node)
            dim = new_dim
        else:
            stages.append(PhysicalStage(count, LayerInfoGraph(layer_nodes), dim))
            layer_nodes = [cur_node]
            dim = cur_node.dimension
            count += 1
    if len(layer_nodes) > 0:
        stages.append(PhysicalStage(count, LayerInfoGraph(layer_nodes), dim))
    return stages

class PhysicalStage():
    def __init__(self, id, graph, dimension, inputs=None, outputs=None):
        self.id = id
        # the graph node is hybrid layer
        self.graph = graph
        self.dimension = dimension
        self.inputs = inputs
        self.outputs = outputs
    
    def get_graph_size(self):
        return self.graph.get_model_size()

    def __repr__(self):
        return f'Stage_{self.id}(dim: {self.dimension}, graph: {self.graph})'
        # return f'Stage_{self.id}'

    def __str__(self):
        return self.__repr__()

stage_type_attr = 1
stage_type_param = 2
class ExecutionStage():
    def __init__(self, name, partition_shape, models, locs, func_ids, mem_limit, stage_type):
        self.name = name
        self.partition_shape = partition_shape
        self.models = models
        self.locs = locs
        self.func_ids = func_ids
        self.mem_limit = mem_limit
        self.stage_type = stage_type
    
    def __repr__(self):
        return f'ExecutionStage(name: {self.name[0]}_{self.name[1]}; shape: {self.partition_shape}; limit: {self.mem_limit}; stage_type: {self.stage_type}; models: {len(self.models)}; ids: {self.func_ids})'

    def __str__(self):
        return self.__repr__()

    def able_to_merge(self, suc_stage):
        if len(self.models) == 1 and len(suc_stage.models) == 1:
            if self.func_ids[0] != suc_stage.func_ids[0]:
                return False
            elif self.func_ids[0] == 0:
                return True
            elif self.func_ids[0] == 1:
                func_limit = hyper_params['func_limit']
                return (self.models[0].get_model_size() + suc_stage.models[0].get_model_size()) <= func_limit
        return False

    def merge_stage(self, suc_stage):
        if not self.able_to_merge(suc_stage):
            print('Unable to merge')
            return
        # print('merge: ', self, suc_stage)
        self.models[0].merge_graph(suc_stage.models[0])
        self.mem_limit += suc_stage.mem_limit
        self.name[1] = suc_stage.name[1]
    
    def cal_size(self):
        return self.models[0].get_model_size() if self.func_ids[0] == 0 else 0

class ExecutionPlan():
    def __init__(self, stage_list=None):
        if stage_list:
            self.stage_list = stage_list
            self.stage_num = len(stage_list)

    def cal_latency(self):
        return predictor.plan_latency(self)
    
    def cal_size(self):
        return sum([ s.cal_size() for s in self.stage_list])
    
    def lat_breakdown(self):
        return predictor.plan_latency_breakdown(self)

    def cal_cost(self):
        return predictor.plan_cost(self)

    def to_json(self, file_name='data.json'):
        json_stage_list = []
        for stage in self.stage_list:
            json_stage = {}
            stage_name = f'{stage.name[0]}_{stage.name[1]}'
            json_stage['name'] = stage_name
            json_stage['coordinate'] = stage_name
            json_stage['partition_shape'] = stage.partition_shape
            json_stage['models'] = []

            for m, l, f in zip(stage.models, stage.locs, stage.func_ids):
                json_model = {
                    'input_shape': m.get_in_shape(),
                    'input_name': m.first_layer.inputs[0].name,
                    'location': l,
                    'function_id': f
                }
                json_stage['models'].append(json_model)
            json_stage_list.append(json_stage)

        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump({'stage_list': json_stage_list}, f, ensure_ascii=False, indent=2)

    def from_json(self, file_name):
        with open(file_name, 'r') as json_file:
            stage_dict = json.load(json_file)

        stage_list = []
        for idx, stage in enumerate(stage_dict['stage_list']):
            stage_name = [ int(i) for i in stage['name'].split('_')]
            partition_shape = stage['partition_shape']
            stage_type = stage_type_param if isinstance(partition_shape, int) else stage_type_attr

            # not required for partition use
            mem_limit = -1 
            models = []
            locs = []

            func_ids = []
            for model in stage['models']:
                fid = model['function_id']
                func_ids.append(fid)
            exe_stage = ExecutionStage(stage_name, partition_shape, models, locs, func_ids, mem_limit, stage_type)
            stage_list.append(exe_stage)
        self.stage_list = stage_list
        self.stage_num = len(stage_list) 
                
    def __repr__(self):
        return f'stage_num: {self.stage_num}; stage_list: {self.stage_list}'

    def __str__(self):
        return self.__repr__() 

def param_partition(layer, parallel_num, main_func_limit):
    models = []
    locs = []
    size_per_model = layer.size
    func_limit = hyper_params['func_limit']
    if size_per_model / parallel_num > func_limit:
        return None

    res_layers = parallel_single_layer(layer, parallel_num)
    for new_layers, loc in res_layers:
        partition_graph = LayerInfoGraph(new_layers)
        models.append(partition_graph)
        locs.append(loc)
    
    stage_name = [layer.name, layer.name]
    if size_per_model / parallel_num <= main_func_limit:
        # put into main function to avoid double billing
        return ExecutionStage(stage_name, parallel_num, models, locs, range(len(models)), main_func_limit, stage_type_param)
    else:
        return ExecutionStage(stage_name, parallel_num, models, locs, range(1, len(models) + 1), main_func_limit, stage_type_param)
        
attr_shape_note = {}
def attr_partition(graph, layers, shape, main_func_limit):
    func_limit = hyper_params['func_limit']

    if sum([ l.size for l in layers]) > func_limit:
        return None

    key = (layers[0].name, layers[-1].name, shape)

    if key in attr_shape_note:
        res_layers = attr_shape_note[key]
    else:
        res_layers = parallel_layers(graph, layers, shape)
        attr_shape_note[key] = res_layers
    models = [ LayerInfoGraph(layers) for layers, loc in res_layers ]
    locs = [ loc for layers, loc in res_layers ]

    stage_name = [layers[0].name, layers[-1].name]
    if models[0].get_model_size() <= main_func_limit:
        return ExecutionStage(stage_name, shape, models, locs, range(len(models)), main_func_limit, stage_type_attr)
    else:
        return ExecutionStage(stage_name, shape, models, locs, range(1, len(models) + 1), main_func_limit, stage_type_attr)

def parallel_single_layer(layer, parallel_num, require_partition=False):
    # TODO multiple dimensions
    out_shape = get_shape(layer.outputs[0])
    new_out_num = int(math.ceil(out_shape[-1] / parallel_num))
    out_shape = out_shape[:-1] + [new_out_num]

    res_layers = []
    for i in range(parallel_num):
        new_layers = []
        for l in layer.layer_nodes:
            new_layer = copy.deepcopy(l)
            output = new_layer.outputs[0]
            set_shape(output, out_shape)
            if require_partition:
                for i in range(len(new_layer.values)):
                    new_dim = new_layer.values[i].dims
                    new_dim[0] = new_out_num
                    new_value = helper.make_tensor(new_layer.values[i].name, 1, new_dim, [1] * reduce(lambda a, b: a*b, new_dim))
                    new_layer.values[i] = new_value

                    for param_i in range(len(new_layer.params)):
                        if new_layer.params[param_i].name == new_layer.values[i].name:
                            set_shape(new_layer.params[param_i], new_dim)
            new_layers.append(new_layer)

        res_layers.append((new_layers, i))
    return res_layers

def parallel_layers(graph, layers, shape, par_required=False):
    '''
    Partitioning the given layers based on output shape.
        - determine the new shape of last layer
        - update all layers from back to head
    Return: all partitioned layers
    '''
    res_layers = []
    parallel_h, parallel_w = shape[0], shape[1] # TODO other dimensions

    output_tensors = divide_tensor(layers[-1].outputs[0], parallel_h, parallel_w)
    use_origin_layer = (parallel_h, parallel_w) == (1, 1)

    for output in output_tensors:
        output_layers = []
        output_tensor, partition_type = output[0], output[1]
        
        for i in range(len(layers) - 1, -1, -1):
            cur_layer = layers[i]
            if cur_layer.hybrid_type == hybrid_type_branch:
                branch_tensor_shapes = []
                branch_nodes, tail_node = cur_layer.layer_nodes

                new_tail_node = copy.deepcopy(tail_node)
                new_tail_node.outputs = [output_tensor]
                op = get_op_object(new_tail_node.type)
                if use_origin_layer:
                    new_tail_node, branch_outputs = new_tail_node, new_tail_node.inputs
                else:
                    new_tail_node, branch_outputs = op.parallel_input(new_tail_node, output_tensor, partition_type)
                output_layers.append(new_tail_node)

                # suc_layer = output_layers[-1] if len(output_layers) > 0 else None
                for branch, branch_output_tensor in zip(branch_nodes, branch_outputs):
                    suc_layer_for_slice = new_tail_node
                    for j in range(len(branch) - 1, -1, -1):
                        new_layer = copy.deepcopy(branch[j])
                        new_layer.outputs = [branch_output_tensor]
                        op = get_op_object(new_layer.type)
                        if use_origin_layer:
                            new_layer, input = new_layer, new_layer.inputs[0]
                        else:
                            new_layer, input = op.parallel_input(new_layer, branch_output_tensor, partition_type)
                        output_layers.append(new_layer)
                        suc_layer_for_slice = new_layer
                        branch_output_tensor = input
                    branch_tensor_shapes.append((get_shape(branch_output_tensor), suc_layer_for_slice))

                branch_shapes = [s for s, _ in branch_tensor_shapes]
                output_tensor_shape = [ max(s) for s in zip(*branch_shapes)]
                output_tensor = copy.deepcopy(cur_layer.inputs[0])
                set_shape(output_tensor, output_tensor_shape)
                for s, l in branch_tensor_shapes:
                    # print(l, partition_type, s, output_tensor_shape)
                    if s != output_tensor_shape:
                        slice_layers = add_slice_layer(output_tensor, l, partition_type)
                        if not par_required:
                            slice_layers = [ LayerInfoNode(l) for l in slice_layers ]
                        output_layers += slice_layers
            
            elif cur_layer.hybrid_type == hybrid_type_fuse:
                for j in range(len(cur_layer.layer_nodes) - 1, -1, -1):
                    new_layer = copy.deepcopy(cur_layer.layer_nodes[j])
                    new_layer.outputs = [output_tensor]
                    op = get_op_object(new_layer.type)
                    if use_origin_layer:
                        new_layer, input = new_layer, new_layer.inputs[0]
                    else:
                        new_layer, input = op.parallel_input(new_layer, output_tensor, partition_type)
                    output_layers.append(new_layer)
                    output_tensor = input

        output_layers.reverse()
        res_layers.append((output_layers, partition_type))
    return res_layers

def add_slice_layer(input_node, output_node, pads):
    """
    Add two slice nodes. (As mxnet only supports one-dimension slicing)
    """
    input_tensor = input_node
    if isinstance(input_node, LayerNode):
        input_tensor = input_node.outputs[0]
    
    input_shape = get_shape(input_tensor)
    input_name = input_tensor.name

    required_input = [ i for i in output_node.inputs if i.name == input_name ][0]
    required_shape = get_shape(required_input)
    
    prefix_name = input_name + '_' + output_node.name

    def get_new_range(pads, old_s, required_s):
        new_range = (0, required_s)
        if pads == [0, 0]:
            bias = int((old_s - required_s) / 2)
            new_range = (bias, old_s - bias)
        elif pads == [0, 1]:
            new_range = (old_s - required_s, old_s)
        return new_range

    slice_layers = []
    slice_input = input_tensor
    in_name = input_name
    for slice_num in range(2):
        starts_name = prefix_name + f'_starts_{slice_num}'
        ends_name = prefix_name + f'_ends_{slice_num}'
        axes_name = prefix_name + f'_axes_{slice_num}'
        out_name = prefix_name + f'_slice_{slice_num}'

        new_range = get_new_range([pads[slice_num], pads[slice_num + 2]], input_shape[slice_num + 2], required_shape[slice_num + 2])
        starts = helper.make_tensor(starts_name, TensorProto.INT64, [1], [new_range[0]])
        ends = helper.make_tensor(ends_name, TensorProto.INT64, [1], [new_range[1]])
        axes = helper.make_tensor(axes_name, TensorProto.INT64, [1], [slice_num + 2])

        starts_info = helper.make_tensor_value_info(starts_name, TensorProto.INT64, [1])
        ends_info = helper.make_tensor_value_info(ends_name, TensorProto.INT64, [1])
        axes_info = helper.make_tensor_value_info(axes_name, TensorProto.INT64, [1])
        out_shape = [input_shape[0], input_shape[1], required_shape[2], input_shape[3]] if slice_num == 1 else [input_shape[0], input_shape[1], required_shape[2], required_shape[3]]
        out_tensor = helper.make_tensor_value_info(out_name, slice_input.type.tensor_type.elem_type, out_shape)

        node = onnx.helper.make_node(
            'Slice',
            name=out_name,
            inputs=[in_name, starts_name, ends_name, axes_name],
            outputs=[out_name],
            starts=[new_range[0]],
            ends=[new_range[1]],
            axes=[slice_num + 2],
        )

        slice_layer = LayerNode(node, [starts_info, ends_info, axes_info], [starts, ends, axes], [slice_input], [out_tensor])
        slice_input = out_tensor
        in_name = out_name
        slice_layers.append(slice_layer)

    for i in range(len(output_node.node.input)):
        if output_node.node.input[i] == input_name:
            output_node.node.input[i] = out_name
    required_input.name = out_name
    slice_layers.reverse()
    return slice_layers