import os
import shutil
import subprocess
import json
from zipfile import ZipFile
from functools import reduce
import argparse
import concurrent
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

from mfn_sdk import MfnClient

host = 'ec2-54-210-85-145.compute-1.amazonaws.com'
mfn = MfnClient(
    mfn_url=f'http://{host}',
    mfn_user="test",
    mfn_password="test",
    # mfn_name="test",
    proxies={
        "http": f'http://{host}:80',
        "https": f'http://{host}:443'
    })

#########################

parser = argparse.ArgumentParser()
parser.add_argument('--job', type=str, default='vgg16_workspace', help="job to deploy")
# parser.add_argument('--ip', type=str, default='172.31.0.96', help="provate ip")
parser.add_argument('--ip', type=str, default='172.31.20.179', help="provate ip")
parser.add_argument('--undeploy', type=bool, default=False)
args = parser.parse_args()

job_name = args.job
private_ip = args.ip
undeploy_all = args.undeploy

def undeploy_func(wf):
    if wf.status == 'deployed':
        print(f'undeploying {wf.name}')
        wf.undeploy()
    mfn.delete_workflow(wf)

if undeploy_all:
    pool = ThreadPoolExecutor(max_workers=8)
    for wf in mfn.workflows:
        fu = pool.submit(undeploy_func, wf)
    pool.shutdown(wait=True)
    exit(0)

# input file path
current_path = os.getcwd()
job_path = os.path.join(current_path, job_name)
input_path = os.path.join(job_path, "input")
models_path = os.path.join(input_path, "models")
structure_json_path = os.path.join(input_path, 'structure.json')

# template file path
template_path = os.path.join(current_path, "templates")

# output file path
output_path = os.path.join(job_path, "output")

###########################

def check_and_new_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

def copy_files(src, dest, name_lists):
    for name in name_lists:
        shutil.copyfile(os.path.join(src, name), os.path.join(dest, name))

def copy_tree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def get_stage_info(path):
    with open(structure_json_path, 'r') as json_file:
        stage_dict = json.load(json_file)
    return stage_dict

def dump_line_list(line_list, file_obj):
    for line in line_list:
        file_obj.writelines(line)
    file_obj.writelines("\n")

def change_variable_in_file(tem_path, file_path, variable_identity, value):
    identity = '<{}>'.format(variable_identity)
    data = []
    with open(tem_path, 'r') as f:
        for l in f:
            if identity in l:
                l = l.replace(identity, value)
            data.append(l)

    with open(file_path, 'w') as f:
        f.writelines(data)

stage_dict = get_stage_info(input_path)
check_and_new_folder(output_path)

############################

output_functions_path = output_path
master_path = os.path.join(output_functions_path, 'master')
os.mkdir(master_path)

shutil.copy2(structure_json_path, master_path)

worker_func_names = []
for idx, stage in enumerate(stage_dict['stage_list']):
    stage_coordinate = stage['coordinate']

    partition_shape = stage['partition_shape']
    for model in stage['models']:
        func_id = model['function_id']
        input_shape = model['input_shape']
        input_name = model['input_name']
        model_name = stage_coordinate + "_" + str(func_id)
        path = os.path.join(models_path, model_name + ".json")
        if int(func_id) == 0:
            shutil.copy2(path, master_path)
        else:
            split_name = model_name.split('_')
            worker_name = '{}_{}_{}'.format(split_name[0], split_name[1], split_name[2])
            worker_func_names.append(worker_name)

            worker_path = os.path.join(output_functions_path, worker_name)

            os.mkdir(worker_path)
            shutil.copy(path, worker_path)
            
            worker_lines = []
            worker_lines.append('from utils import *\n')
            worker_lines.append('model = get_model(\"{}\", {}, \"{}\")\n'.format(model_name, input_shape, input_name))
            worker_lines.append('local_key = \"{}\"\n'.format(worker_name))
            worker_lines.append('def handle(event, context):\n')
            worker_lines.append('    return execute_model(event, model, local_key)\n')

            woker_end_point_path = os.path.join(worker_path, '{}.py'.format(worker_name))
            output_obj = open(woker_end_point_path, 'w+')
            dump_line_list(worker_lines, output_obj)
            output_obj.close()

util_temp_path = os.path.join(template_path, 'utils_template.py')
util_write_path = os.path.join(template_path, 'utils.py')
change_variable_in_file(util_temp_path, util_write_path, 'private_ip', private_ip)

for name in os.listdir(output_functions_path):
    path = os.path.join(output_functions_path, name)
    if os.path.isdir(path):
        copy_files(template_path, path, ['requirements.txt', 'utils.py'])

print('output generated')
############################

wf_endpoints = {}

for worker_func_name in worker_func_names:
    zip_name = f'{worker_func_name}.zip'
    zip_path = os.path.join(output_functions_path, zip_name)
    if os.path.exists(zip_path):
        os.remove(zip_path)

    worker_path = os.path.join(output_functions_path, worker_func_name)
    os.chdir(worker_path)

    with ZipFile(zip_path,'w') as zf:
        for f in os.listdir(worker_path):
            zf.write(f)

def deploy_func(worker_func_name):
    knix_func = None
    for f in mfn.functions:
        if f.name == worker_func_name:
            knix_func = f
    
    if knix_func is None:
        knix_func = mfn.add_function(worker_func_name)
    zip_name = f'{worker_func_name}.zip'
    zip_path = os.path.join(output_functions_path, zip_name)

    knix_func.upload(zip_path)
    knix_func.requirements = 'redis==3.5.3\nmxnet==1.5.1\nnumpy==1.17.2\nrequests==2.22.0\nurllib3==1.25.6'

    wf_name = f'wf_{worker_func_name}'
    # wf = None
    for w in mfn.workflows:
        if w.name == wf_name:
            if w.status == 'deployed':
                w.undeploy()
            mfn.delete_workflow(w)
        # if w.name == wf_name:
        #     wf = w

    # if wf is None:
    wf = mfn.add_workflow(wf_name)

    wf.json = '''
{
    "Comment": "",
    "StartAt": "start",
    "States": {
        "start": {
            "Type": "Task",
            "Resource": "''' + worker_func_name + '''",
            "End": true
        }
    }
}'''

    wf.deploy(timeout=0)
    return wf.endpoint

pool = ThreadPoolExecutor(max_workers=4)

fs = {}
for idx, worker_func_name in enumerate(worker_func_names):
    print(f'Submit job {idx + 1}/{len(worker_func_names)}: {worker_func_name}')
    fu = pool.submit(deploy_func, worker_func_name)
    fs[fu] = worker_func_name

for future in concurrent.futures.as_completed(fs):
    worker_func_name = fs[future]
    wf_endpoints[worker_func_name] = future.result()

master_temp_path = os.path.join(template_path, 'master_template.py')
master_func_path = os.path.join(template_path, 'master.py')
change_variable_in_file(master_temp_path, master_func_path, 'worker_urls', json.dumps(wf_endpoints))
shutil.copy2(master_func_path, master_path)

zip_path = os.path.join(output_functions_path, 'master.zip')
if os.path.exists(zip_path):
    os.remove(zip_path)
os.chdir(master_path)
with ZipFile(zip_path,'w') as zf:
    for f in os.listdir(master_path):
        zf.write(f)

wf = deploy_func('master')