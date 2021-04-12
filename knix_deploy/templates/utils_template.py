import time
import os
import redis
import urllib
import json
import pickle
import numpy as np
import mxnet as mx
import struct
import base64
from collections import namedtuple
from functools import reduce

Batch = namedtuple('Batch', ['data'])

r = redis.Redis(host='<private_ip>', port='6379')

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

def execute_model(event, mod, local_key):

    # start_clock = event['clock']
    start_clock = time.time()
    data = pickle.loads(r.get(event['key']))

    mod.forward(Batch([data]), is_train=False)
    output = mod.get_outputs()[0]

    pickled_output = pickle.dumps(output)
    r.set(local_key, pickled_output)
    local_latency = int((time.time() - start_clock) * 1000)

    response = {
        "key": local_key,
        "latency": local_latency,
    }

    return response