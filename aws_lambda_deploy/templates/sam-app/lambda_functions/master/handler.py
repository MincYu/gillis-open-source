try:
    import unzip_requirements
except ImportError:
    pass
import time
import os
import urllib
import json
import numpy as np
import mxnet as mx
import boto3
from collections import namedtuple
from mxnet.contrib import onnx as onnx_mxnet
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from utils import *
from functools import reduce
from collections import defaultdict

start = time.time()

urllib.request.urlretrieve('https://raw.githubusercontent.com/multimedia-berkeley/tutorials/master/grids.txt',
                           '/tmp/grids.txt')
grids = []
with open('/tmp/grids.txt', 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        lat = float(line[1])
        lng = float(line[2])
        grids.append((lat, lng))

img_url = 'https://farm5.staticflickr.com/4275/34103081894_f7c9bfa86c_k_d.jpg'
image_file = '/tmp/img.jpg'
urllib.request.urlretrieve(img_url, image_file)

mydict = lambda: defaultdict(mydict)

with open(os.path.join(os.getcwd(), 'structure.json'), 'r') as json_file:
    stage_dict = json.load(json_file)

stage_info = []
master_models = {}
for idx, stage in enumerate(stage_dict['stage_list']):
    stage_coordinate = stage['coordinate']

    partition_shape = stage['partition_shape']
    model_dict = mydict()
    for model in stage['models']:
        fid = model['function_id']
        model_dict[fid]['input_shape'] = model['input_shape']
        # model_dict[fid]['input_name'] = model['input_name']
        # model_dict[fid]['location'] = model['location']
        model_name = stage_coordinate + "_" + str(fid)
        model_dict[fid]['model_name'] = model_name
        if fid == 0:
            master_model = get_model(model_name, model['input_shape'], model['input_name'])
            master_models[model_name] = master_model

    stage_info.append((partition_shape, model_dict))

print('load after ' + str(int((time.time() - start) * 1000)))

client = boto3.client('lambda')
record_time = False

def worker(func_dict, mid_clock=None):
    split_name = func_dict['model_name'].split('_')
    response = client.invoke(
        FunctionName='from{}To{}Worker{}'.format(
            split_name[0], split_name[1], split_name[2]),
        InvocationType="RequestResponse",
        Payload=json.dumps(func_dict)
    )

    if record_time:
        network_time = int((time.time() - mid_clock) * 1000)

    json_result = json.loads(response['Payload'].read())
    np_output = decode_array(json_result['shape'], json_result['output'])
    nd_output = mx.nd.array(np_output)

    lat = json_result['latency']
    if record_time:
        decode_time = int((time.time() - mid_clock) * 1000)
        return (nd_output, lat, network_time, decode_time)
    else:
        return (nd_output, lat)

def do_parallel(model_dict, mid_clock):
    outputs = []
    futures = []
    lats = []

    thread_num = 0
    master_flag = 0
    master_input = None
    master_model = None
    for fid, func_dict in model_dict.items():
        if not fid == 0:
            _, input_str = encode_array(func_dict['input_data'].asnumpy())
            func_dict['input_data'] = input_str
            thread_num += 1
        else:
            master_flag = 1
            master_input = func_dict['input_data']
            master_model = master_models[func_dict['model_name']]

    encode_clock = int((time.time() - mid_clock) * 1000)

    if not thread_num == 0:
        pool = ThreadPoolExecutor(max_workers=thread_num)
        for fid, func_dict in model_dict.items():
            if not fid == 0:
                fu = pool.submit(worker, func_dict, mid_clock)
                futures.append(fu)

    if master_flag == 1:
        master_model.forward(Batch([master_input]), is_train=False)
        master_res = master_model.get_outputs()[0].asnumpy()
        outputs.append(mx.nd.array(master_res))

    local_clock = int((time.time() - mid_clock) * 1000)

    if not thread_num == 0:
        for f in futures:
            fu_result = f.result()
            outputs.append(fu_result[0])
            if record_time:
                lats.append(fu_result[1:])
            else:
                lats.append(fu_result[1])

    return outputs, (lats, encode_clock, local_clock)

def load_data(size=(224, 224)):
    img = mx.image.imread(image_file)
    img = mx.image.imresize(img, size[0], size[1])
    img = img.transpose((2, 0, 1))
    img = img.expand_dims(axis=0)
    return img

def parallel():
    all_lats = []
    timer = []
    img = load_data()
    mid_clock = time.time()

    input = img
    for partition_shape, model_dict in stage_info:

        input_list = get_splits(input, partition_shape, model_dict)
        sorted_keys = sorted(model_dict.keys())
        for idx, fid in enumerate(sorted_keys):
            model_dict[fid]['input_data'] = input_list[idx]
        
        output, lats = do_parallel(model_dict, mid_clock)

        if isinstance(partition_shape, int):
             output = reduce(lambda x, y: mx.nd.concat(x, y, dim=1), output)
        else:
            dim2_num = partition_shape[0]
            dim3_num = partition_shape[1]
            dim2_list = []
            for i in range(dim2_num):
                dim2_list.append(reduce(lambda x, y: mx.nd.concat(x, y, dim=3),
                                        output[i*dim3_num : (i+1)*dim3_num]))
            output = reduce(lambda x, y: mx.nd.concat(x, y, dim=2),
                            dim2_list)
        
        input = output
        all_lats += lats[0]
        timer += lats[1:]
        timer.append(int((time.time() - mid_clock) * 1000))

    output = output.asnumpy()
    all_lats.append(timer)
    return output, all_lats

def lambda_handler(event, context):
    # start = time.time()

    ret = {}
    output, conv_lats = parallel()
    prob = output[0]
    
    pred = np.argsort(prob)[::-1]
    pred_loc = grids[int(pred[0])]

    ret['pred'] = pred_loc
    # ret['latency'] = int((time.time() - start) * 1000)
    ret['cost'] = conv_lats
    ret['latency'] = conv_lats[-1][-1]

    response = {
        "statusCode": 200,
        "body": json.dumps(ret)
    }

    return response