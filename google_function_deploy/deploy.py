import os
import shutil
import subprocess
import json
from zipfile import ZipFile
from functools import reduce
import argparse
import concurrent
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

#########################

parser = argparse.ArgumentParser()
parser.add_argument('--job', type=str, default='vgg16_workspace', help="job to deploy")
args = parser.parse_args()

job_name = args.job

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
            worker_lines.append('mod = get_model(\"{}\", {}, \"{}\")\n'.format(model_name, input_shape, input_name))
            worker_lines.append('local_key = \"{}\"\n'.format(worker_name))
            worker_lines.append('def handle(request):\n')
            worker_lines.append('    return execute_model(request, mod)\n')

            woker_end_point_path = os.path.join(worker_path, 'main.py')
            output_obj = open(woker_end_point_path, 'w+')
            dump_line_list(worker_lines, output_obj)
            output_obj.close()

for name in os.listdir(output_functions_path):
    path = os.path.join(output_functions_path, name)
    if os.path.isdir(path):
        copy_files(template_path, path, ['requirements.txt', 'utils.py'])

master_func_path = os.path.join(template_path, 'main.py')
shutil.copy2(master_func_path, master_path)

print('output generated')


def deploy_func(worker_func_name):
    cmd = ['gcloud', 'functions', 'deploy', 'w' + worker_func_name, '--region', 'asia-east2', '--source', os.path.join(output_path, worker_func_name), '--entry-point', 'handle', 
        '--runtime', 'python37', '--memory', '4096MB', '--trigger-http', '--allow-unauthenticated']

    subprocess.run(cmd)

pool = ThreadPoolExecutor(max_workers=8)

for worker_func_name in worker_func_names + ['master']:
    print(f'Deploying {worker_func_name}')
    pool.submit(deploy_func, worker_func_name)

pool.shutdown(wait=True)