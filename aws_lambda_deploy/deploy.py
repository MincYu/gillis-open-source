import os
import shutil
import subprocess
import json
from functools import reduce
import argparse

#########################

parser = argparse.ArgumentParser()
parser.add_argument('--job', type=str, default='wresnet50_2', help="job to deploy")
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
sam_app_path = os.path.join(template_path, "sam-app")

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

stage_dict = get_stage_info(input_path)
check_and_new_folder(output_path)
copy_tree(sam_app_path, output_path + "/")

############################

output_functions_path = os.path.join(output_path, 'lambda_functions')
worker_template_path = os.path.join(template_path, 'worker.py')

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
            shutil.copy2(path, os.path.join(output_functions_path, 'master'))
            shutil.copy2(structure_json_path, os.path.join(output_functions_path, 'master'))
        else:
            split_name = model_name.split('_')
            worker_path = os.path.join(output_functions_path,
                                      'from{}To{}Worker{}'.format(
                                      split_name[0], split_name[1], split_name[2]))
            os.mkdir(worker_path)
            shutil.copy(path, worker_path)
            
            worker_lines = []
            worker_lines.append('try:\n')
            worker_lines.append('  import unzip_requirements\n')
            worker_lines.append('except ImportError:\n')
            worker_lines.append('  pass\n')
            worker_lines.append('from utils import *\n')
            worker_lines.append('model = get_model(\"{}\", {}, \"{}\")\n'.format(model_name, input_shape, input_name))
            worker_lines.append('def lambda_handler(event, context):\n')
            worker_lines.append('  return execute_model(event, model)\n')

            woker_end_point_path = os.path.join(worker_path, 'worker.py')
            output_obj = open(woker_end_point_path, 'w+')
            dump_line_list(worker_lines, output_obj)
            output_obj.close()

for name in os.listdir(output_functions_path):
    path = os.path.join(output_functions_path, name)
    if os.path.isdir(path):
        copy_files(template_path, path, ['requirements.txt', '__init__.py', 'utils.py'])

############################
    
prefix_path = os.path.join(template_path, 'template_yaml_prefix')
postfix_path = os.path.join(template_path, 'template_yaml_postfix')
output_template_path = os.path.join(output_path, 'template.yaml')

line_list_list = []
line_list_list.append(open(prefix_path, 'r').readlines())

body_list = []
for name in os.listdir(models_path):
    if 'json' in name:
        name_body = name[:-5]
        func_id = name_body.split('_')[-1]
        if not int(func_id) == 0:
            split_name = name_body.split('_')
            identity = 'from{}To{}Worker{}'.format(split_name[0], split_name[1], split_name[2])
            
            body_list.append('  {}:\n'.format(identity))
            body_list.append('    Type: AWS::Serverless::Function\n')
            body_list.append('    Properties:\n')
            body_list.append('      FunctionName: {}\n'.format(identity))
            body_list.append('      CodeUri: lambda_functions/{}/\n'.format(identity))
            body_list.append('      Handler: worker.lambda_handler\n')
            body_list.append('\n')

body_list.append('\n')
line_list_list.append(body_list)
line_list_list.append(open(postfix_path, 'r').readlines())


output_obj = open(output_template_path, 'w+')
for line_list in line_list_list:
    dump_line_list(line_list, output_obj)
output_obj.close()