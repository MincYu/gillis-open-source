import os
import onnx
from onnx import helper, TensorProto
import math
import copy
from functools import reduce

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

backends = ['lambda', 'knix', 'google']
backend_no = 1
selected_backend =backends[backend_no]

model_dir = os.path.dirname(__file__) + '/models'
partition_results_dir = os.path.dirname(os.path.dirname(__file__))



# load runtime configuration
import configparser

config = configparser.ConfigParser()
config.read('layer_runtime.cfg')

encode_oh_a = config.getfloat(selected_backend, 'encode_oh_a')
decode_oh_a = config.getfloat(selected_backend, 'decode_oh_a')

relu_a = config.getfloat(selected_backend, 'relu_a')
relu_b = config.getfloat(selected_backend, 'relu_b')
pool_a = config.getfloat(selected_backend, 'pool_a')
pool_b = config.getfloat(selected_backend, 'pool_b')
conv_a = config.getfloat(selected_backend, 'conv_a')
conv_b = config.getfloat(selected_backend, 'conv_b')
gemm_a = config.getfloat(selected_backend, 'gemm_a')
gemm_b = config.getfloat(selected_backend, 'gemm_b')
bn_a = config.getfloat(selected_backend, 'bn_a')
bn_b = config.getfloat(selected_backend, 'bn_b')
add_a = config.getfloat(selected_backend, 'add_a')
add_b = config.getfloat(selected_backend, 'add_b')
lstm_lat = config.getfloat(selected_backend, 'lstm_lat')

def create_dir(newdir):
    """
    Return True if the creation is successful, False if the dir exists
    """
    if type(newdir) is not str:
        newdir = str(newdir)
    if os.path.isdir(newdir):
        print("The new dir " + newdir + " exists ")
        return False
    elif os.path.isfile(newdir):
        raise OSError("a file with the same name as the desired " \
                      "dir, '%s', already exists." % newdir)
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            create_dir(head)
        if tail:
            os.mkdir(newdir)
        return True

def get_shape(tensor):
    return [ d.dim_value for d in tensor.type.tensor_type.shape.dim]

def set_shape(tensor, new_shape):
    for i in range(len(new_shape)):
        tensor.type.tensor_type.shape.dim[i].dim_value = new_shape[i]

def get_attrs(node):
    return { a.name: (a.ints, a.type) for a in node.attribute}

def divide_tensor(tensor, h, w):
    ele_type = tensor.type.tensor_type.elem_type
    shape = get_shape(tensor)

    if len(shape) != 4:
        return [(tensor, [1, 1, 1, 1])]

    # TODO: only support 2 dimension now
    new_shape = [shape[0], shape[1], int(math.ceil(shape[2] / h)), int(math.ceil(shape[3] / w))]

    tensors = []
    tensor = helper.make_tensor_value_info(tensor.name, ele_type, new_shape)
    for i in range(h):
        for j in range(w):
            pad_type = get_pad_type(i, j, h, w)
            tensors.append((tensor, pad_type))
    return tensors

def get_pad_type(h_i, w_j, h, w):
    h_pads = check_one_dimen(h_i, h)
    w_pads = check_one_dimen(w_j, w)
    return (h_pads[0], w_pads[0], h_pads[1], w_pads[1])

def check_one_dimen(i, d):
    if d == 1:
        return (1, 1)
    elif d > 1:
        if i == 0:
            return (1, 0)
        elif i == d - 1:
            return (0, 1)
        else:
            return (0, 0)

def update_pads(new_layer, new_pads):
    old_attr_pad = [ a for a in new_layer.node.attribute if a.name == 'pads'][0]
    old_ints = [i for i in old_attr_pad.ints]
    [ old_attr_pad.ints.remove(i) for i in old_ints]
    [ old_attr_pad.ints.insert(i, new_pads[i]) for i in range(len(new_pads))]