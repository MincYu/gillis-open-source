import policy_dp
import policy_rl
import policy_bayesian
import policy_bf
import partition
import click
from utils import *
from parallel_util import *

all_benchmarks = {
    'vgg11.onnx': [500, 600],
}

algos = {
    'lo': policy_dp.test_model,
    'sa': policy_rl.test_benchmark,
    'bf': policy_bf.test_benchmark,
    'bayesian': policy_bayesian.test_benchmark
}

def export_model_partitions(name, plan, workspace_name=None):
    work_path =  workspace_name + '_workspace/input' if workspace_name else name.split('.')[0] + '_workspace/input'
    work_dir = f'{partition_results_dir}/{work_path}'
    if create_dir(work_dir):
        partition.model_partition(name, plan, work_dir)
        plan_json_path = f'{work_dir}/structure.json'
        plan.to_json(plan_json_path)

@click.command()
@click.argument('algo', type=str)
@click.option('-p', '--require-partition', type=bool, default=False)
@click.option('-n', '--name', type=str)
@click.option('-t', '--threshold', type=int)
@click.option('-d', '--rl-model-dir', type=str)
def main(algo, require_partition, name, threshold, rl_model_dir):
    if not algo in ['lo', 'sa', 'bf', 'bayesian']:
        print('Unknown algorithm.') 
        return
    
    if algo == 'lo':
        # latency-optimal
        plan = algos[algo](name)
        if require_partition:
            export_model_partitions(name, plan)

    elif algo == 'sa':
        # slo-aware
        benchmarks = {name : [threshold]}
        
        if require_partition:
            plan = policy_rl.gen_plan_with_model(name, rl_model_dir)
            workspace_name = 'rl_' + name.split('.')[0] + f'_{threshold}'
            export_model_partitions(name, plan, workspace_name=workspace_name)
        else:
            algos[algo](benchmarks)
    else:
        benchmarks = {name : [threshold]}

        benchmark_res = algos[algo](benchmarks)
        action = benchmark_res[(name, threshold)][2]
        if require_partition:
            plan = gen_plan_with_action(name, action)
            workspace_name = f'{algo}_' + name.split('.')[0] + f'_{threshold}'
            export_model_partitions(name, plan, workspace_name=workspace_name)

if __name__ == "__main__":
    main()