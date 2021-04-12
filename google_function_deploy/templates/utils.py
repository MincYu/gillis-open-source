import time
import os
import urllib
import json
import numpy as np
import mxnet as mx
import struct
import base64
from collections import namedtuple
from functools import reduce

Batch = namedtuple('Batch', ['data'])

def encode_array(tensor_data):
    shape = tensor_data.shape
    byte_data = struct.pack("<%df" % tensor_data.size, *tensor_data.flatten())
    base64_data = base64.b64encode(byte_data)
    string_data = str(base64_data)[2:-1]  # throw "b'" and "'"
    # result_json = json.dumps(string_data)
    return shape, string_data


def decode_array(shape, string_data):
    all_len = reduce(lambda a, b: a * b, shape)
    # string_data = json.loads(result_json)
    base64_data = bytes(string_data, 'utf-8')
    byte_data = base64.b64decode(base64_data)
    new_list = list(struct.unpack("<%df" % all_len, byte_data))
    return np.array(new_list).reshape(shape)

def get_splits(input, partition_shape, model_dict):
    
    sorted_keys = sorted(model_dict.keys())
    
    # partition Dense
    if isinstance(partition_shape, int):
        input_shape = model_dict[sorted_keys[0]]['input_shape'][1]
        return [input[:, :input_shape]] * partition_shape

    # filter non-4D tensor
    if len(input.shape) < 4:
        input_shape = model_dict[sorted_keys[0]]['input_shape'][1]
        return [input[:, :input_shape]]

    return_list = []
    dim2_size = input.shape[2]
    dim3_size = input.shape[3]
    dim2_num = partition_shape[0]
    dim3_num = partition_shape[1]
    for i in range(dim2_num):
        for j in range(dim3_num):
            idx = i * dim3_num + j
            dim2_delta = model_dict[sorted_keys[idx]]['input_shape'][2]
            dim3_delta = model_dict[sorted_keys[idx]]['input_shape'][3]

            if dim2_num < 2:
                dim2_s = 0
            else:
                dim2_s = i * ((dim2_size - dim2_delta) // (dim2_num - 1))
            if dim3_num < 2:
                dim3_s = 0
            else:
                dim3_s = j * ((dim3_size -  dim3_delta) // (dim3_num - 1))

            dim2_e = dim2_s + dim2_delta
            dim3_e = dim3_s + dim3_delta
            return_list.append(input[:, :, dim2_s:dim2_e, dim3_s:dim3_e])
    
    return return_list

def get_model(model_name, shape, name='data'):
    mod = mx.mod.Module(mx.symbol.load("./{}.json".format(model_name)), data_names=[name], label_names=None)
    mod.bind(for_training=False, data_shapes=[(name, shape)], label_shapes=None)
    mod.init_params()
    return mod

def execute_model(request, mod):

    # start_clock = event['clock']
    start_clock = time.time()
    request_json = request.get_json(silent=True)
    in_shape = request_json['input_shape']
    json_data = request_json['input_data']

    input = decode_array(in_shape, json_data)
    data = mx.nd.array(input)

    mod.forward(Batch([data]), is_train=False)
    output = mod.get_outputs()[0]
    np_output = output.asnumpy()

    # json_ret = json.dumps(np_output, cls=NumpyEncoder)    
    out_shape, json_ret = encode_array(np_output)
    latency = int((time.time() - start_clock) * 1000)

    response = {
        "output": json_ret,
        "shape": out_shape,
        "latency": latency,
        # "latency": [decode_clock, exe_clock, encode_clock],
    }

    return response