import os
import numpy as np
import mxnet as mx
import onnx
import glob
import networkx as nx
from onnx import helper, shape_inference
from utils import *
import math
import copy
import json
from functools import reduce

class BaseGraph():
    def __init__(self, layers=None, require_append=False):
        self.graph = nx.DiGraph()
        self.first_layer = None
        self.tail_layer = None
        self.output_to_layer = {}

        if layers:
            self.first_layer = layers[0]
            if require_append:
                [ self.append_layer(l) for l in layers ]
            else:
                self.gen_graph(layers)

    def __repr__(self):
        return self.graph.nodes().__repr__()

    def __str__(self):
        return self.graph.nodes().__str__()

    def gen_graph(self, layers, get_name=lambda obj: obj.name):
        for l in layers:
            self.graph.add_node(l)
            # update output dict
            for out in l.outputs:
                if get_name(out) in self.output_to_layer:
                    logging.warn('Duplicated!')
                    print(self.output_to_layer[get_name(out)], l)
                else:
                    self.output_to_layer[get_name(out)] = l
            # traverse input to add edge        
            for inp in l.inputs:
                if get_name(inp) in self.output_to_layer:
                    self.graph.add_edge(self.output_to_layer[get_name(inp)], l)
            self.tail_layer = l

    def merge_graph(self, other_g):
        self.gen_graph(other_g.get_layers())

    def get_hybrid_nodes(self, partition_layer_node=False):
        next_layers = [self.first_layer]
        # cur_one_node = self.first_layer
        layer_nodes = []
        hybrid_nodes = []
        hybrid_index = 0

        while next_layers:
            if len(next_layers) == 1:
                cur_one_node = next_layers[0]

                if len(layer_nodes) == 0 or is_ele_op(cur_one_node.type):
                    layer_nodes.append(cur_one_node)
                else:
                    hybrid_nodes.append(HybridLayerInfoNode(layer_nodes, hybrid_index, partition_layer_node=partition_layer_node))
                    hybrid_index += 1
                    layer_nodes = [cur_one_node]
                next_layers = list(self.graph.successors(cur_one_node))

            elif len(next_layers) > 1:
                if len(layer_nodes) > 0:
                    hybrid_nodes.append(HybridLayerInfoNode(layer_nodes, hybrid_index, partition_layer_node=partition_layer_node))
                    hybrid_index += 1
                    layer_nodes = []
                traversed_branches = [ self.traverse_branch(l, 1, False) for l in next_layers ]
                # traversed_branches.sort(key=lambda e: len(e))

                tail_node = traversed_branches[0][-1]

                hybrid_nodes.append(HybridLayerInfoNode(([ branch[:-1] for branch in traversed_branches], tail_node), hybrid_index, hybrid_type=hybrid_type_branch, partition_layer_node=partition_layer_node))
                hybrid_index += 1
                next_layers = list(self.graph.successors(tail_node))
            else:
                next_layers = None
        if len(layer_nodes) > 0:
            hybrid_nodes.append(HybridLayerInfoNode(layer_nodes, hybrid_index, partition_layer_node=partition_layer_node))
        return hybrid_nodes

    def traverse_graph(self):
        cur_node = self.first_layer
        last_layer = self.traverse_branch(cur_node)

    def modify_input_name(self, new_name):
        old_name = self.first_layer.inputs[0].name
        print(f'modify old name {old_name}')
        for l in self.get_layers():
            for i in range(len(l.node.input)):
                if l.node.input[i] == old_name:
                    l.node.input[i] = new_name
                    l.inputs[i].name = new_name

    def traverse_branch(self, cur_node, blank=0, print_out=True):
        tab = blank * '    '
        branch_nodes = []
        while self.graph.has_node(cur_node):
            branch_nodes.append(cur_node)
            if blank and len(list(self.graph.predecessors(cur_node))) > 1:
                if print_out:
                    print(f'{tab}---------')
                # branch return the traversed node
                return branch_nodes
            if print_out:
                print(f'{tab}{cur_node}')
            next_layers = list(self.graph.successors(cur_node))
            if next_layers and len(next_layers) > 0:
                if len(next_layers) == 1:
                    cur_node = next_layers[0]
                else:
                    join_node = set([ self.traverse_branch(l, blank + 1)[-1] for l in next_layers ])
                    if len(join_node) == 1: # TODO: complicated branch
                        cur_node = list(join_node)[0]
                    else:
                        break
            else:
                break
        # root return the last node
        return cur_node
        
    def append_layer(self, layer): # for linear model
        self.graph.add_node(layer)
        if not self.first_layer:
            self.first_layer = layer
        if self.tail_layer:
            self.graph.add_edge(self.tail_layer, layer)
        self.tail_layer = layer

    def get_layers(self):
        return list(self.graph.nodes())

class LayerGraph(BaseGraph):

    def to_onnx_model(self):
        if not self.tail_layer:
            return None
        layers = []
        inputs = self.first_layer.inputs
        outputs = self.tail_layer.outputs
        initializers = []

        for l in self.get_layers():
            layers.append(l.node)
            inputs += [ i for i in l.params if i not in inputs]
            initializers += [ i for i in l.values if i not in initializers]
        
        graph_def = helper.make_graph(layers, 'test', inputs=inputs, outputs=outputs, initializer=initializers)
        model_def = helper.make_model(graph_def, producer_name='onnx-test')
        return model_def

class LayerNode():
    def __init__(self, onnx_node, input_info, parameters, inputs=None, outputs=None):
        self.node = onnx_node
        self.params = input_info # infer output 
        self.values = parameters
        self.type = self.node.op_type

        self.name = onnx_node.name if onnx_node.name else onnx_node.op_type + '_' + onnx_node.output[0]
        self.inputs = inputs
        self.outputs = outputs
        self.dimension = get_op_object(self.type).get_parallel_dim()
    
    def __repr__(self):
        return f'Layer({self.name})'

    def __str__(self):
        return self.__repr__()

def gen_layer_graph(model):

    onnx_model = shape_inference.infer_shapes(model)

    model_parameters = { input_info.name: input_info for input_info in onnx_model.graph.input }
    model_values = { params_input.name: params_input for params_input in onnx_model.graph.initializer }
    model_value_info = { value_i.name: value_i for value_i in onnx_model.graph.value_info }

    layer_nodes = []
    for n in onnx_model.graph.node:
        parameters = [ model_parameters[i] for i in n.input if i in model_values] # avoid include 'data'
        values = [ model_values[i] for i in n.input if i in model_values ]
        inputs = [ model_value_info[i] for i in n.input if i in model_value_info ] 
        if onnx_model.graph.input[0].name in n.input:
            inputs.append(onnx_model.graph.input[0])

        outputs = [ model_value_info[i] for i in n.output if i in model_value_info ]
        if onnx_model.graph.output[0].name in n.output:
            outputs.append(onnx_model.graph.output[0])
        layer_nodes.append(LayerNode(n, parameters, values, inputs, outputs))

    layer_graph = LayerGraph(layer_nodes)

    # layer_graph.traverse_graph()
    return layer_graph

def load_onnx_model(model_path):
    if os.path.isdir(model_path): # multiple sub models
        model_pathes = glob.glob(f'{model_path}/*.onnx')
        model_pathes.sort()
        
        base_model = onnx.load(model_pathes[0])
        graph = gen_layer_graph(base_model)
        input_name = graph.tail_layer.outputs[0].name

        for path in model_pathes[1:]:
            print(f'loading model: {path}')
            sub_model = onnx.load(path)
            sub_graph = gen_layer_graph(sub_model)
            sub_graph.modify_input_name(input_name)
            graph.merge_graph(sub_graph)

            input_name = sub_graph.tail_layer.outputs[0].name

    elif os.path.isfile(model_path):
        model = onnx.load(model_path)
        graph = gen_layer_graph(model)
    else:
        print('Incorrect model path.')
        graph = None
    return graph

class LayerInfoGraph(BaseGraph):
    def __init__(self, arg=None):
        layers = arg
        require_append = False
        if isinstance(layers, LayerGraph):
            layers = [ LayerInfoNode(l) for l in layers.get_layers()]
        elif isinstance(layers[0], HybridLayerInfoNode):
            require_append = True

        BaseGraph.__init__(self, layers, require_append)
    
    def get_model_size(self):
        size = sum([l.size for l in self.get_layers()])
        return size
    
    def get_in_shape(self):
        return get_shape(self.first_layer.inputs[0])

    def get_out_shape(self):
        return get_shape(self.tail_layer.outputs[0])

    def get_in_out_shape(self):
        return (self.get_in_shape(), self.get_out_shape())

def cal_tensor_size(tensor):
    shape = get_shape(tensor)
    num = reduce(lambda a, b: a*b, shape)
    return num * 4  / 1024 / 1024

class LayerInfoNode():
    def __init__(self, layer_node):
        self.node = layer_node.node
        self.type = layer_node.node.op_type
        self.inputs = layer_node.inputs
        self.outputs = layer_node.outputs
        self.name = layer_node.name
        self.dimension = layer_node.dimension
        self.size = self.get_param_size(layer_node.params)
    
    def get_param_size(self, params):
        # return size in MB
        times = 4 # float
        size = 0
        for p in params: # params size
            size += cal_tensor_size(p)
        
        # for o in self.outputs: # outputs size
        #     size += cal_tensor_size(o)
        return size

    def __repr__(self):
        return f'LayerInfo({self.name})'

    def __str__(self):
        return self.__repr__()

    def get_out_shape(self):
        return get_shape(self.outputs[0])

hybrid_type_branch = 0
hybrid_type_fuse = 1
class HybridLayerInfoNode():
    def __init__(self, layer_nodes, name, hybrid_type=hybrid_type_fuse, partition_layer_node=False):
        self.layer_nodes = layer_nodes
        self.name = name
        self.hybrid_type = hybrid_type

        if hybrid_type == hybrid_type_fuse:
            self.inputs = layer_nodes[0].inputs
            self.dimension = layer_nodes[-1].dimension
            self.outputs = layer_nodes[-1].outputs
            if not partition_layer_node:
                self.size = sum([ l.size for l in layer_nodes])

        elif hybrid_type == hybrid_type_branch:
            branch_nodes, tail_node = layer_nodes
            self.inputs = branch_nodes[0][0].inputs
            self.dimension = tail_node.dimension
            self.outputs = tail_node.outputs
            if not partition_layer_node:
                self.size = 0 
                for branch in branch_nodes:
                    self.size += sum([ l.size for l in branch])
                self.size += tail_node.size

    def __repr__(self):
        return f'HybridLayerInfo({self.name})'

    def __str__(self):
        return self.__repr__()

### For LSTM
class RNNLayerInfoNode():
    def __init__(self, idx, hidden_size):
        self.node = None
        self.type = 'LSTM'
        self.hidden_size = hidden_size
        self.name = f'lstm_{idx}'
        self.inputs = [helper.make_tensor_value_info(f'output_{idx-1}', 1, [1, 1, hidden_size * 3])] # including the output and state
        self.outputs = [helper.make_tensor_value_info(f'output_{idx}', 1, [1, 1, hidden_size * 3])]
        self.dimension = get_op_object(self.type).get_parallel_dim()
        self.size = self.get_param_size(hidden_size)
    
    def get_param_size(self, hidden_size):
        # empirical results
        return 8 * hidden_size * hidden_size * 4 / 1024 / 1024

    def __repr__(self):
        return f'RNNLayerInfo({self.name})'

    def __str__(self):
        return self.__repr__()

    def get_out_shape(self):
        return get_shape(self.outputs[0])

class RNNLayerInfoGraph(BaseGraph):
    def __init__(self, layer_size, hidden_size=2048):
        layers = [ RNNLayerInfoNode(i, hidden_size) for i in range(layer_size) ]
        BaseGraph.__init__(self, layers, require_append=True)
    
    def get_model_size(self):
        size = sum([l.size for l in self.get_layers()])
        return size
    
    def get_in_shape(self):
        return get_shape(self.first_layer.inputs[0])

    def get_out_shape(self):
        return get_shape(self.tail_layer.outputs[0])

    def get_in_out_shape(self):
        return (self.get_in_shape(), self.get_out_shape())

### latency predictor

get_cost = lambda lat: math.ceil(lat / 100) * 100

class MeanPredictor():

    def plan_latency(self, plan):
        latency = 0
        for s in plan.stage_list:
            latency += self._stage_latency(s)
        return latency
    
    def _stage_latency(self, stage):
        latency = 0
        worker_models = [ m for m, id in zip(stage.models, stage.func_ids) if id != 0]
        if len(worker_models) > 0:
            latency += sum([ cal_en_overhead(m.get_in_shape()) for m in worker_models ])
            latency += cal_wait_time(worker_models)
            latency += sum([ cal_de_overhead(m.get_out_shape()) for m in worker_models ])
        else: # only have master function
            latency += model_latency(stage.models[0])
        return latency

    def plan_latency_breakdown(self, plan):
        pass

    def plan_cost(self, plan):
        cost = 0
        latency = 0
        for s in plan.stage_list:
            cost += self._stage_cost(s)
            latency += self._stage_latency(s)
        cost += get_cost(latency)
        return cost
    
    # get cost for worker functions
    def _stage_cost(self, stage):
        worker_models = [ m for m, id in zip(stage.models, stage.func_ids) if id != 0]
        return sum([ get_cost( cal_de_overhead(m.get_in_shape()) + model_latency(m) + cal_en_overhead(m.get_out_shape())) for m in worker_models ])
    

### For prediction use
from scipy.stats import exponnorm
from scipy import signal
class LambdaCommLatency():
    def __init__(self, dist="exponnorm", delta=1e-3, bound=300):
        super(LambdaCommLatency, self, ).__init__()
        
        supported_dist = ["exponnorm"]
        if dist in supported_dist:
            self.dist = dist
        else:
            exit('[Error] LambdaCommLatency: unintended distribution type!')
            
        self.delta = delta
        self.bound = bound
        self.big_grid = np.arange(-bound, bound, delta)

    def a_call_b_dist(self, x):
        rate = 6.629996127305979e-07 * x + 2.765206182220345
        mean = 7.4112946016258187e-05 * x + 28.832586318285166
        std = 2.1693192919678337e-06 * x + 0.9965843281779094
        return exponnorm(rate, mean, std)

    def b_return_a_dist(self, x):
        rate = -6.376711465017302e-09 * x + 2.7652061637888035
        mean = 5.5936456142808507e-05 * x + 28.832586317657142
        std = 1.054867186790225e-06 * x + 0.9965845859530119
        return exponnorm(rate, mean, std)
    
    def getLat(self, n, d1, d2, per):
        pmf1 = self.a_call_b_dist(d1).pdf(self.big_grid) * self.delta
        pmf2 = self.b_return_a_dist(d2).pdf(self.big_grid) * self.delta
        pmf3 = np.flip(self.a_call_b_dist(0).pdf(self.big_grid) * self.delta)
        conv_pmf = signal.fftconvolve(signal.fftconvolve(pmf1, pmf2, 'same'), pmf3, 'same')
        
        conv_pdf = conv_pmf / self.delta
        conv_cdf = conv_pmf
        count = 0
        for i in range(len(conv_cdf)):
            temp = conv_cdf[i]
            conv_cdf[i] += count
            count += temp
       
        max_pdf = n * np.ones(len(conv_pdf))
        for i in range(len(max_pdf)):
            max_pdf[i] *= conv_pdf[i] * conv_cdf[i] ** (n - 1)

        max_pmf = max_pdf * self.delta
        max_cdf = max_pmf
        count = 0
        for i in range(len(max_cdf)):
            temp = max_cdf[i]
            max_cdf[i] += count
            count += temp
            
        idx = np.searchsorted(max_cdf, per / 100)
        latency = - self.bound + idx * self.delta
        return latency
    
    def getExp(self, n, d1, d2):
        pmf1 = self.a_call_b_dist(d1).pdf(self.big_grid) * self.delta
        pmf2 = self.b_return_a_dist(d2).pdf(self.big_grid) * self.delta
        pmf3 = np.flip(self.a_call_b_dist(0).pdf(self.big_grid) * self.delta)
        conv_pmf = signal.fftconvolve(signal.fftconvolve(pmf1, pmf2, 'same'), pmf3, 'same')
        
        conv_pdf = conv_pmf / self.delta
        conv_cdf = conv_pmf
        count = 0
        for i in range(len(conv_cdf)):
            temp = conv_cdf[i]
            conv_cdf[i] += count
            count += temp
       
        max_pdf = n * np.ones(len(conv_pdf))
        for i in range(len(max_pdf)):
            max_pdf[i] *= conv_pdf[i] * conv_cdf[i] ** (n - 1)
        
        exp = 0
        for a, b in zip(self.big_grid, max_pdf):
            exp += a * b
        return exp * self.delta

class KnixCommLatency():
    def getExp(self, n, d1, d2):
        read_redis_rate = 64
        write_redis_rate = 64
        io_amount = d1 + d2
        comm_lat = io_amount / read_redis_rate + io_amount / write_redis_rate + 12
        return comm_lat

class GoogleCommLatency():
    def getExp(self, n, d1, d2):
        send_k = 0.1204
        receive_k = 0.135
        comm_lat = send_k * d1 + receive_k * d2 + 40
        return comm_lat

def model_latency(model):
    latency = 0
    layers = model.graph.nodes()
    for l in layers:
        latency += layer_latency(l)
    return latency

def layer_latency(layer):
    return get_op_object(layer.type).predict(layer)

def cal_wait_time(worker_models):
    model_time = max([ cal_de_overhead(m.get_in_shape()) + model_latency(m) + cal_en_overhead(m.get_out_shape()) for m in worker_models ])
    comm_time = cal_comm_overhead(len(worker_models), *worker_models[0].get_in_out_shape())
    return  model_time + comm_time

comm_predictors = [LambdaCommLatency(delta=2), KnixCommLatency(), GoogleCommLatency()]
comm_latency = comm_predictors[backend_no]
def cal_comm_overhead(worker, in_shape, out_shape, m_a=3008, m_b=3008):
    in_num = reduce(lambda a, b: a*b, in_shape) / 1024
    out_num = reduce(lambda a, b: a*b, out_shape) / 1024
    # if use_direct_call:
    #     a = 0.12016927
    #     b = 0.09227865
    #     l_inv = 21.51555554
    #     return l_inv + a * in_num + b * out_num
    # else:
    #     r_w = 0.02
    #     return 18 + r_w * (in_num + out_num)
    return comm_latency.getExp(worker, in_num, out_num)

def cal_en_overhead(shape, mem=3008):
    num = reduce(lambda a, b: a*b, shape) / 1024
    a = encode_oh_a
    return a * num

def cal_de_overhead(shape, mem=3008):
    num = reduce(lambda a, b: a*b, shape) / 1024
    a = decode_oh_a
    return a * num

predictor = MeanPredictor()

### The statistics of popular layers
class Operator():
    def predict(self, layer, m=3008):
        return 0
    
    def get_parallel_dim(self):
        return [False, False]

    def parallel_input(self, new_layer, new_output, new_pads):
        input = new_layer.inputs[0]
        return (new_layer, input)

    def parallel_params(self, new_layer, new_output, new_pads):
        pass

def is_ele_op(layer_type):
    return layer_type in ['Flatten', 'Relu', 'BatchNormalization', 'Dropout']

class OpFlatten(Operator):
    def get_parallel_dim(self):
        return [False, False]

class OpRelu(Operator):
    def predict(self, layer, m=3008):
        shape = get_shape(layer.inputs[0])
        op_num = reduce(lambda a, b: a*b, shape)
        a = relu_a
        b = relu_b
        return max(0, linear_res(a, op_num/1024, b))

    def get_parallel_dim(self):
        return [False, True, True, True]

    def parallel_input(self, new_layer, new_output, new_pads):
        input = new_layer.inputs[0]
        set_shape(input, get_shape(new_output))
        return (new_layer, input)

class OpMaxPool(Operator): # 2D
    def predict(self, layer, m=3008):
        # TODO stride > 2
        shape = get_shape(layer.inputs[0])
        op_num = reduce(lambda a, b: a*b, shape)
        a = pool_a
        b = pool_b
        return max(0, linear_res(a, op_num/1024, b))

    def get_parallel_dim(self):
        return [False, True, True, True]

    def parallel_input(self, new_layer, new_output, new_pads):
        attrs = get_attrs(new_layer.node)
        kernel_shape = attrs['kernel_shape'][0]
        strides = attrs['strides'][0]
        pads = attrs['pads'][0]

        new_pads = [ t*p for t, p in zip(new_pads, pads)]
        update_pads(new_layer, new_pads)

        # shape_factor = [ k + (k - s) for k, s in zip(kernel_shape, strides)]
        shape_factor = strides
        out_shape = get_shape(new_output)
        # in_shape = out_shape[:2] + [ o * f for o, f in zip(out_shape[2:], shape_factor)]

        in_hw_shape = out_shape[2:]
        in_hw_shape[0] = (in_hw_shape[0] - 1) * strides[0] + kernel_shape[0] - (new_pads[0] + new_pads[2])
        in_hw_shape[1] = (in_hw_shape[1] - 1) * strides[1] + kernel_shape[1] - (new_pads[1] + new_pads[3])
        in_shape = out_shape[:2] + in_hw_shape

        input = new_layer.inputs[0]
        set_shape(input, in_shape)
        return (new_layer, input)         

class OpConv(Operator):
    def predict(self, layer, m=3008):
        attrs = get_attrs(layer.node)
        kernel_shape = attrs['kernel_shape'][0]
        strides = attrs['strides'][0]
        pads = attrs['pads'][0]

        in_shape = get_shape(layer.inputs[0])
        out_shape = get_shape(layer.outputs[0])
        c, h, w, k, r, s = in_shape[1], in_shape[2], in_shape[3], out_shape[1], kernel_shape[0], kernel_shape[1]
        matrix_mul_times = k * ((h + pads[0] + pads[2] - r) / strides[0] + 1 ) * ((w + pads[1] + pads[3] - s) / strides[1] + 1) # TODO numbers of output
        matrix_ops = c * r * s

        all_ops = matrix_mul_times * matrix_ops
        a = conv_a
        b = conv_b
        return max(0, linear_res(a, all_ops / 1024, b))

    def get_parallel_dim(self):
        return [False, False, True, True]

    def parallel_input(self, new_layer, new_output, new_pads):
        attrs = get_attrs(new_layer.node)
        dilations = attrs['dilations'][0]
        kernel_shape = attrs['kernel_shape'][0]
        strides = attrs['strides'][0]
        pads = attrs['pads'][0]

        new_pads = [ t*p for t, p in zip(new_pads, pads)]
        update_pads(new_layer, new_pads)

        out_shape = get_shape(new_output)
        # in_hw_shape = [ o * f for o, f in zip(out_shape[2:], strides)] # TODO
        # in_hw_shape[0] = in_hw_shape[0] + (kernel_shape[0] - (new_pads[0] + new_pads[2]) - 1)
        # in_hw_shape[1] = in_hw_shape[1] + (kernel_shape[1] - (new_pads[1] + new_pads[3]) - 1)

        in_hw_shape = out_shape[2:]
        in_hw_shape[0] = (in_hw_shape[0] - 1) * strides[0] + kernel_shape[0] - (new_pads[0] + new_pads[2])
        in_hw_shape[1] = (in_hw_shape[1] - 1) * strides[1] + kernel_shape[1] - (new_pads[1] + new_pads[3])
        in_shape = get_shape(new_layer.inputs[0])[:2] + in_hw_shape
        
        input = new_layer.inputs[0]
        set_shape(input, in_shape)
        return (new_layer, input)

class OpGemm(Operator):
    def predict(self, layer, m=3008):
        in_shape = get_shape(layer.inputs[0])
        out_shape = get_shape(layer.outputs[0])
        param_num = reduce(lambda a, b: a*b, in_shape) * reduce(lambda a, b: a*b, out_shape)
        # a = 0.001185219
        a = gemm_a
        b = gemm_b
        return max(0, linear_res(a, param_num/1024, b))

    def get_parallel_dim(self):
        return [False, False]
    
    def parallel_params(self, new_layer, new_output, new_pads):
        pass

class OpBatchNormalization(Operator):
    def predict(self, layer, m=3008):
        shape = get_shape(layer.inputs[0])
        op_num = reduce(lambda a, b: a*b, shape)
        a = bn_a
        b = bn_b
        return max(0, linear_res(a, op_num/1024, b))

    def get_parallel_dim(self):
        return [False, True, True, True]
    
    def parallel_input(self, new_layer, new_output, new_pads):
        input = new_layer.inputs[0]
        set_shape(input, get_shape(new_output))
        return (new_layer, input)

class OpAdd(Operator):
    def predict(self, layer, m=3008):
        shape = get_shape(layer.inputs[0])
        op_num = reduce(lambda a, b: a*b, shape)
        a = add_a
        b = add_b
        return max(0, linear_res(a, op_num/1024, b)) 

    def get_parallel_dim(self):
        return [False, True, True, True]

    def parallel_input(self, new_layer, new_output, new_pads):
        out_shape = get_shape(new_output)
        
        input = []
        for i in new_layer.inputs:
            set_shape(i, out_shape)
            input.append(i)
        return (new_layer, input)  

class OpGlobalAveragePool(Operator):
    def get_parallel_dim(self):
        return [False, True, False, False]

class OpConcat(Operator):
    def predict(self, layer, m=3008):
        return 0

    def get_parallel_dim(self):
        return [False, False, True, True]

    def parallel_input(self, new_layer, new_output, new_pads):
        out_shape = get_shape(new_output)
        
        input = []
        for i in new_layer.inputs:
            set_shape(i, out_shape)
            input.append(i)
        return (new_layer, input)  

class OpLSTM(Operator):
    def predict(self, layer, m=3008):
        if layer.hidden_size == 2048:
            return lstm_lat
        else: # TODO
            return

    def get_parallel_dim(self):
        return [False, True, True]

def linear_res(a, x, b):
    return a * x + b

op_types = {
    'Default': Operator(),
    'Flatten': OpFlatten(),
    'Relu': OpRelu(),
    'Conv': OpConv(),
    'MaxPool': OpMaxPool(),
    'Gemm': OpGemm(),
    'BatchNormalization': OpBatchNormalization(),
    'Add': OpAdd(),
    'GlobalAveragePool': OpGlobalAveragePool(),
    'AveragePool': OpMaxPool(),
    'Concat': OpConcat(), 

    'LSTM': OpLSTM(),
}    

get_op_object = lambda op_type: op_types[op_type] if op_type in op_types else op_types['Default']