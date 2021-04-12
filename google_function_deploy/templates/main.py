import requests
import urllib
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

# worker_url_dict = json.loads('<worker_urls>')

def worker(func_name, json_data):
    # url = worker_url_dict[func_name]
    url = 'https://asia-east2-singular-style-240409.cloudfunctions.net/w{}'.format(func_name)
    response = requests.post(url, json=json_data)

    json_result = response.json()

    np_output = decode_array(json_result['shape'], json_result['output'])
    nd_output = mx.nd.array(np_output)

    lat = json_result['latency']
    return (nd_output, lat)

def do_parallel(model_dict, start_clock):
    outputs = []
    futures = []
    lats = []

    thread_num = 0
    master_flag = 0
    master_input = None
    master_model = None
    
    func_name_input = []
    for fid, func_dict in model_dict.items():
        if not fid == 0:
            func_name = func_dict['model_name']
            input_shape = func_dict['input_shape']
            _, input_str = encode_array(func_dict['input_data'].asnumpy())
            func_name_input.append((func_name, {'input_data': input_str, 'input_shape': input_shape}))
            thread_num += 1
        else:
            master_flag = 1
            master_input = func_dict['input_data']
            master_model = master_models[func_dict['model_name']]

    local_clock = int((time.time() - start_clock) * 1000)

    if not thread_num == 0:
        pool = ThreadPoolExecutor(max_workers=thread_num)
        for func_name, json_input in func_name_input:
            fu = pool.submit(worker, func_name, json_input)
            futures.append(fu)
    
    if master_flag == 1:
        master_model.forward(Batch([master_input]), is_train=False)
        master_res = master_model.get_outputs()[0].asnumpy()
        outputs.append(mx.nd.array(master_res))

    if not thread_num == 0:
        for f in futures:
            fu_result = f.result()
            outputs.append(fu_result[0])
            lats.append(fu_result[1])

    return outputs, local_clock, lats

def load_data(size=(224, 224)):
    img = mx.image.imread(image_file)
    img = mx.image.imresize(img, size[0], size[1])
    img = img.transpose((2, 0, 1))
    img = img.expand_dims(axis=0)
    return img

def parallel():
    timer = []
    img = load_data()
    start_clock = time.time()
    master_timer = []

    input = img
    for partition_shape, model_dict in stage_info:

        input_list = get_splits(input, partition_shape, model_dict)
        sorted_keys = sorted(model_dict.keys())

        for idx, fid in enumerate(sorted_keys):
            model_dict[fid]['input_data'] = input_list[idx]
        
        output, local_clock, lats = do_parallel(model_dict, start_clock)

        before_concat = int((time.time() - start_clock) * 1000)
        master_timer.append(before_concat - local_clock)
        if isinstance(partition_shape, int):
             output = reduce(lambda x, y: mx.nd.concat(x, y, dim=1), output)
        else:
            # dim2_num = partition_shape[0]
            # dim3_num = partition_shape[1]
            # dim2_list = []
            # for i in range(dim2_num):
            #     dim2_list.append(reduce(lambda x, y: mx.nd.concat(x, y, dim=3),
            #                             output[i*dim3_num : (i+1)*dim3_num]))
            # output = reduce(lambda x, y: mx.nd.concat(x, y, dim=2),
            #                 dim2_list)
            
            if partition_shape == [1, 1]:
                output = output[0]
            elif partition_shape == [1, 2]:
                output = mx.nd.concat(output[0], output[1], dim=3)
            elif partition_shape == [2, 2]:
                output = mx.nd.concat(mx.nd.concat(output[0], output[1], dim=3), mx.nd.concat(output[2], output[3], dim=3), dim=2)
            elif partition_shape == [2, 4]:
                output = mx.nd.concat(mx.nd.concat(*output[0:4], dim=3), mx.nd.concat(*output[4:8], dim=3), dim=2)
            elif partition_shape == [4, 4]:
                output = mx.nd.concat(mx.nd.concat(*output[0:4], dim=3), mx.nd.concat(*output[4:8], dim=3), mx.nd.concat(*output[8:12], dim=3), mx.nd.concat(*output[12:16], dim=3), dim=2)

        input = output
        timer.append(lats)

    output = output.asnumpy()
    latency = int((time.time() - start_clock) * 1000)
    return output, timer, master_timer, latency

def handle(request):
    # start = time.time()

    ret = {}
    output, timer, before_concats, latency = parallel()
    
    # prob = output[0]
    # pred = np.argsort(prob)[::-1]
    # pred_loc = grids[int(pred[0])]
    # ret['pred'] = pred_loc

    ret['timer'] = timer
    ret['concat'] = before_concats
    ret['latency'] = latency

    return ret