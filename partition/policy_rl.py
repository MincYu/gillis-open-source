import os
import onnx
from onnx import helper
import math
import copy
from parallel_util import *
import time
from functools import reduce

import numpy as np
import glob
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn, Block
from mxnet import ndarray as F

class AttnDecoderRNN(Block):
    def __init__(self, hidden_size, output_size, max_length, n_layers=1, dropout_p=0):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        with self.name_scope():
            self.in_dense = nn.Dense(self.hidden_size)
            self.attn = nn.Dense(self.max_length, in_units=self.hidden_size * 2)
            self.attn_combine = nn.Dense(self.hidden_size, in_units=self.hidden_size * 2)
            if self.dropout_p > 0:
                self.dropout = nn.Dropout(self.dropout_p)
            self.rnn = rnn.GRU(self.hidden_size, input_size=self.hidden_size)
            self.out = nn.Dense(self.output_size, in_units=self.hidden_size)

    def forward(self, input, hidden, encoder_outputs):
        #input shape, (1,)
        embedded = self.in_dense(input)
        attn_weights = F.softmax(
            self.attn(F.concat(embedded, hidden[0].flatten(), dim=1)))
        attn_applied = F.batch_dot(attn_weights.expand_dims(0),
                                 encoder_outputs.expand_dims(0))

        output = F.concat(embedded.flatten(), attn_applied.flatten(), dim=1)
        output = self.attn_combine(output).expand_dims(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.rnn(output, hidden)

        output = self.out(output)

        return output, hidden, attn_weights

    def initHidden(self, ctx):
        return [F.zeros((1, 1, self.hidden_size), ctx=ctx)]

class EncoderRNN(Block):
    def __init__(self, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        with self.name_scope():
            self.rnn = rnn.GRU(hidden_size, input_size=self.hidden_size)

    def forward(self, input, hidden):
        output = input
        for i in range(self.n_layers):
            output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        return [F.zeros((1, 1, self.hidden_size))]

class PolicyNet(Block):
    def __init__(self, input_size, output_size):
        super(PolicyNet, self).__init__()

        with self.name_scope():
            self.dense1 = nn.Dense(2 * input_size, in_units=input_size)
            self.act = nn.Activation(activation='relu')
            
            self.out1 = nn.Dense(output_size, in_units=2 * input_size)

    def forward(self, input):
        x = self.dense1(input)
        x = self.act(x)
        out = self.out1(x)
        return out

class BasePolicyGradient:
    def __init__(
            self,
            layer_embeddings,
            param_dir,
            batch_size=20,
            learning_rate=0.001,
            weight_decay=0.0001,
            reward_decay=.99,
            require_mem_limit=False,
            disjoint_train=True,
            mem_batch_size=1,
    ):
        self.layer_embeddings = layer_embeddings
        self.embedding_size = layer_embeddings.shape[-1] # input size of partition net 
        self.param_dir = param_dir

        self.layer_size = layer_embeddings.shape[0]
        self.batch_size = batch_size
        self.lr = learning_rate
        self.wd = weight_decay
        self.gamma = reward_decay
        self.require_mem_limit = require_mem_limit
        self.disjoint_train = disjoint_train
        self.output_size = choice_num

        self.mem_input_size = self.layer_size + 3 # input size of mem net 
        self.mem_output_size = 2
        self.mem_batch_size = mem_batch_size if self.disjoint_train else 1

        self.actions, self.rewards, self.outputs = [], [], []
        self.build_net()

    def load_weights(self, net, file_path):
        if os.path.isfile(file_path):
            print(f'loading params from {file_path}')
            net.load_parameters(file_path, ctx=ctx)
        else:
            net.initialize(ctx=ctx)

    def build_net(self):
        pass

    def store_reward(self, rs):
        self.rewards = rs

    def discount_and_norm_rewards(self, batch_reward):
        # discount episode rewards
        discounted_ep_rs = []
        running_add = 0
        for t in reversed(range(0, len(batch_reward))):
            running_add = running_add * self.gamma + batch_reward[t]
            discounted_ep_rs.append(running_add)

        discounted_ep_rs.reverse()
        # normalize episode rewards
        # discounted_ep_rs -= np.mean(discounted_ep_rs)
        # discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

class NetPolicyGradient(BasePolicyGradient):
    def save_weights(self):
        self.partition_net.save_parameters(self.partition_net_file)
        if self.require_mem_limit:
            self.mem_net.save_parameters(self.mem_net_file)

    def build_net(self):
        self.partition_net_file = f'{self.param_dir}/partition_net.params'
        self.mem_net_file = f'{self.param_dir}/mem_net.params'

        self.partition_net = PolicyNet(self.embedding_size, self.output_size)
        self.load_weights(self.partition_net, self.partition_net_file)
        self.partition_net_optimizer = gluon.Trainer(self.partition_net.collect_params(), 'adam', {'learning_rate': self.lr, 'wd': self.wd})

        if self.require_mem_limit:
            self.mem_net = PolicyNet(self.mem_input_size, self.mem_output_size)
            self.load_weights(self.mem_net, self.mem_net_file)
            self.mem_net_optimizer = gluon.Trainer(self.mem_net.collect_params(), 'adam', {'learning_rate': self.lr, 'wd': self.wd})

        self.criterion = gluon.loss.SoftmaxCrossEntropyLoss()

    def choose_actions(self):
        all_stage_ranges = []
        for _ in range(self.batch_size):
            batch_partition_action = []
            
            for i in range(self.layer_size):
                output = self.partition_net(self.layer_embeddings[i])
                prob = F.softmax(output).asnumpy()
                action = np.random.choice(range(prob.shape[1]), p=prob.ravel())

                batch_partition_action.append(action)

            batch_stage_range = gen_stage_range(batch_partition_action, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes)

            if self.require_mem_limit:
                stage_embeddings = gen_stage_embedding(batch_stage_range, self.layer_size)
                stage_embeddings = stage_embeddings.expand_dims(0).swapaxes(0, 1)
                self.outputs.append(stage_embeddings)
                for _ in range(self.mem_batch_size):
                    batch_mem_action = []

                    for embedding in stage_embeddings:
                        men_output = self.mem_net(embedding)
                        mem_prob = F.softmax(men_output).asnumpy()
                        mem_action = np.random.choice(range(mem_prob.shape[1]), p=mem_prob.ravel())
                        batch_mem_action.append(mem_action)

                    new_batch_stage_range = copy.deepcopy(batch_stage_range)
                    all_stage_ranges.append(modify_stage_range(new_batch_stage_range, batch_mem_action))
                    self.actions.append((batch_partition_action, batch_mem_action))
            else:
                self.actions.append(batch_partition_action)
                all_stage_ranges.append(batch_stage_range)
        return self.actions, all_stage_ranges
    
    def learn(self):
        discounted_ep_rs_norm = [self.discount_and_norm_rewards(batch_reward) for batch_reward in self.rewards]

        # train partition net
        with autograd.record():
            loss = F.zeros((1,))

            for i in range(self.layer_size):
                output = self.partition_net(self.layer_embeddings[i])

                for b in range(self.batch_size):
                    if self.require_mem_limit:
                        neg_log_prob = self.criterion(output, F.array([self.actions[b*self.mem_batch_size][0][i]]))
                        partition_reward = np.average([ r[i] for r in discounted_ep_rs_norm[b*self.mem_batch_size: (b+1)*self.mem_batch_size]])
                        step_loss = neg_log_prob * partition_reward
                    else:
                        neg_log_prob = self.criterion(output, F.array([self.actions[b][i]]))
                        step_loss = neg_log_prob * discounted_ep_rs_norm[b][i]
                    loss = F.add(loss, step_loss)                   

            loss.backward()
        self.partition_net_optimizer.step(self.batch_size)

        # train mem net
        if self.require_mem_limit:

            with autograd.record():
                loss = F.zeros((1,))
                for batch_idx, stage_embeddings in enumerate(self.outputs):
                    base_idx = batch_idx * self.mem_batch_size
                
                    stage_len = stage_embeddings.shape[0]
                    rewards = [ [0] * (stage_len - 1) + [r[-1]] for r in self.rewards[base_idx:base_idx+self.mem_batch_size]]
                    discounted_ep_rs_norm = [self.discount_and_norm_rewards(batch_reward) for batch_reward in rewards]
                    for i in range(stage_len):
                        output = self.mem_net(stage_embeddings[i])

                        for b in range(self.mem_batch_size):
                            # print(base_idx+b, i)
                            neg_log_prob = self.criterion(output, F.array([self.actions[base_idx+b][1][i]]))
                            step_loss = neg_log_prob * discounted_ep_rs_norm[b][i]
                            loss = F.add(loss, step_loss)  
                loss.backward()
            self.mem_net_optimizer.step(self.mem_batch_size * self.batch_size)

        self.actions, self.rewards, self.outputs = [], [], []
        return loss.asscalar()

class Seq2SeqPolicyGradient(BasePolicyGradient):
    def save_weights(self):
        self.encoder.save_parameters(self.encoder_file)
        self.decoder.save_parameters(self.decoder_file)

    def build_net(self):
        self.encoder_file = f'{self.param_dir}/encoder.params'
        self.decoder_file = f'{self.param_dir}/decoder.params'

        self.GO_token = [0, 0, hyper_params['func_limit']] if self.require_mem_limit else [0]

        self.encoder = EncoderRNN(self.embedding_size)
        self.decoder = AttnDecoderRNN(self.embedding_size, self.output_size, self.layer_size)

        self.load_weights(self.encoder, self.encoder_file)
        self.load_weights(self.decoder, self.decoder_file)

        self.encoder_hidden = self.encoder.initHidden()

        self.encoder_optimizer = gluon.Trainer(self.encoder.collect_params(), 'adam', {'learning_rate': self.lr, 'wd': self.wd})
        self.decoder_optimizer = gluon.Trainer(self.decoder.collect_params(), 'adam', {'learning_rate': self.lr, 'wd': self.wd})

        self.criterion = gluon.loss.SoftmaxCrossEntropyLoss()

    def choose_actions(self):
        encoder_outputs, hidden = self.encoder(self.layer_embeddings, self.encoder_hidden)
        encoder_outputs = encoder_outputs.flatten()

        remain_mem = hyper_params['func_limit']

        for _ in range(self.batch_size):
            batch_action = []
            decoder_input = F.array(self.GO_token).expand_dims(0)
            decoder_hidden = hidden
            for _ in range(self.layer_size):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                
                prob = F.softmax(decoder_output).asnumpy()

                action = np.random.choice(range(prob.shape[1]), p=prob.ravel())

                if self.require_mem_limit:
                    pass # TODO memory limit
                    # mem_index = decoder_output[:, choice_num : choice_num + mem_num].argmax(axis=1).asscalar()

                    # if choice_index == 0: # only determine allocated memory at parallel phases
                    #     mem_alloc = 0
                    # else:
                    #     mem_alloc = mem_index * hyper_params['step']

                    # if mem_alloc > remain_mem:
                    #     mem_alloc = remain_mem
                    # remain_mem -= mem_alloc

                    # action = [choice_index, mem_alloc, remain_mem]

                batch_action.append(action)
                decoder_input = F.array([action]).expand_dims(0)
            self.actions.append(batch_action)
        return self.actions

    def learn(self):
        discounted_ep_rs_norm = [self.discount_and_norm_rewards(batch_reward) for batch_reward in self.rewards]
        with autograd.record():
            loss = F.zeros((1,))
            encoder_outputs, hidden = self.encoder(self.layer_embeddings, self.encoder_hidden)
            encoder_outputs = encoder_outputs.flatten()

            decoder_input = F.array(self.GO_token).expand_dims(0)
            decoder_hidden = hidden
            for i in range(self.layer_size):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                for b in range(self.batch_size):
                    neg_log_prob = self.criterion(decoder_output, F.array([self.actions[b][i]]))
                    step_loss = neg_log_prob * discounted_ep_rs_norm[b][i]

                    loss = F.add(loss, step_loss)
            
            loss.backward()
        self.encoder_optimizer.step(self.batch_size)
        self.decoder_optimizer.step(self.batch_size)

        self.actions, self.rewards, self.outputs = [], [], []
        return discounted_ep_rs_norm, loss.asscalar()

def get_reward(info_graph, hybrid_layers, all_range_actions, lat_threshold, require_mem_limit, debug=False):
    res = np.zeros(len(hybrid_layers))
    plan = get_plan(info_graph, hybrid_layers, all_range_actions)
    if not plan:
        res[-1] = penalty
        return res, None, None

    latency = plan.cal_latency()
    cost = plan.cal_cost()
    size = plan.cal_size()
    res[-1] = get_last_step_reward(latency, cost, lat_threshold)

    if debug:
        print(plan)
        print(f'overall. latency: {latency}, cost: {cost}, size: {size}')
        print('======Reward======')
        print(res)
        return res, plan
    return res, latency, cost

def get_last_step_reward(latency, cost, lat_threshold):
    if latency <= lat_threshold:
        return budget - cost
    else:
        return lat_threshold - latency

ctx = mx.cpu()

choice_num = len(shape_choice) + 1 # the first one indicates whether to parallel
mem_num = int(hyper_params['func_limit'] / hyper_params['step']) + 1

print_every = 10

def train_policy_iters(policy, info_graph, hybrid_layers, lat_threshold, param_dir, iter=1500, require_mem_limit=False):
    reward_file = f'{param_dir}/re.npy'

    print_loss_total, print_reward_total = 0, 0
    collect_rewards = np.array([])
    if os.path.isfile(reward_file):
        collect_rewards = np.load(reward_file)
    
    start = time.time()
    for i in range(iter):
        iter_clock = time.time()
        actions, all_stage_ranges = policy.choose_actions()

        rewards = []
        latencies = []
        costs = []
        for stage_ranges in all_stage_ranges:
            reward, latency, cost = get_reward(info_graph, hybrid_layers, stage_ranges, lat_threshold, require_mem_limit)
            rewards.append(reward)
            if latency:
                latencies.append(latency)
            if cost:
                costs.append(cost)

        policy.store_reward(rewards)
        loss = policy.learn()
        # loss = 0
        iter_duration = int((time.time() - iter_clock) * 1000)

        last_res = [r[-1] for r in rewards]
        avg_reward = np.average(last_res)
        avg_lat = np.average(latencies)
        avg_cost = np.average(costs)
        collect_rewards = np.append(collect_rewards, [avg_reward, avg_lat, avg_cost])

        print_loss_total += loss
        print_reward_total += avg_reward
        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_reward_avg = print_reward_total / print_every
            print_loss_total, print_reward_total = 0, 0

            best_idx = np.argmax(last_res)
            logging.info(f'iter: {i} dur: {iter_duration} reward: {print_reward_avg} best_r: {last_res[best_idx]} actions: {actions[best_idx]}')
    np.save(reward_file, collect_rewards)
    policy.save_weights()
    duration = int((time.time() - start) * 1000)
    logging.info(f'Solution generated. duration: {duration}')

def test_benchmark(benchmarks):
    for name, lat_thresholds in benchmarks.items():
        global budget, penalty, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes

        embeddings, info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes = get_graph_metrics(name)
        require_mem_limit = info_graph.get_model_size() > hyper_params['func_limit']

        param_name = name.split('.')[0]
        budget = get_cost(model_latency(info_graph)) * 10 # reward upper bound
        penalty = -budget # reward lower bound

        for lat_threshold in lat_thresholds:
            param_dir = os.path.dirname(os.path.abspath(__file__)) + f'/params_{param_name}_{lat_threshold}'
            if not os.path.exists(param_dir):
                os.mkdir(param_dir)
                policy = NetPolicyGradient(embeddings, param_dir, require_mem_limit=require_mem_limit)
                train_policy_iters(policy, info_graph, hybrid_layers, lat_threshold, param_dir, require_mem_limit=require_mem_limit)

            else:
                logging.warn(f'param dir {param_dir} exists.')

def gen_plan_with_model(name, single_model_dir):
    global budget, penalty, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes
    embeddings, info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes = get_graph_metrics(name)

    require_mem_limit = info_graph.get_model_size() > hyper_params['func_limit']
    
    budget = get_cost(model_latency(info_graph)) * 10 # reward upper bound
    penalty = -budget # reward lower bound
    lat_threshold = int(single_model_dir.split('/')[-1].split('_')[-1])

    policy = NetPolicyGradient(embeddings, single_model_dir, require_mem_limit=require_mem_limit)
    actions, all_stage_ranges = policy.choose_actions()
    opt_latency, opt_cost, opt_action = float('inf'), float('inf'), None

    for _ in range(20):
        for action, stage_ranges in zip(actions, all_stage_ranges):
            reward, latency, cost = get_reward(info_graph, hybrid_layers, stage_ranges, lat_threshold, require_mem_limit)
            if latency < lat_threshold and cost < opt_cost:
                opt_cost = cost
                opt_latency = latency
                opt_action = action
    logging.info(f'opt_latency: {opt_latency}, opt_cost: {opt_cost}, opt_action: {opt_action}')
    
    if require_mem_limit:
        stage_range = gen_stage_range(opt_action[0], phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes)
        stage_range = modify_stage_range(stage_range, opt_action[1])
    else:
        stage_range = gen_stage_range(opt_action, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes)
    plan = get_plan(info_graph, hybrid_layers, stage_range)
    return plan

def eval_rl_models(name, model_dir):
    res = {}
    global budget, penalty, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes
    embeddings, info_graph, hybrid_layers, phy_stages, aggre_stage_layer_sizes, aggre_layer_sizes = get_graph_metrics(name)

    require_mem_limit = info_graph.get_model_size() > hyper_params['func_limit']
    
    budget = get_cost(model_latency(info_graph)) * 10 # reward upper bound
    penalty = -budget # reward lower bound
    
    all_dirs = glob.glob(f'{model_dir}/*')
    all_dirs.sort(key=lambda e: int(e.split('/')[-1].split('_')[-1]))
    for d in all_dirs:
        d_name = d.split('/')[-1]
        lat_threshold = int(d_name.split('_')[-1])

        policy = NetPolicyGradient(embeddings, d, require_mem_limit=require_mem_limit)
        actions, all_stage_ranges = policy.choose_actions()
        opt_latency, opt_cost, opt_action = float('inf'), float('inf'), None

        for _ in range(30):
            for action, stage_ranges in zip(actions, all_stage_ranges):
                reward, latency, cost = get_reward(info_graph, hybrid_layers, stage_ranges, lat_threshold, require_mem_limit)
                if latency < lat_threshold and cost < opt_cost:
                    opt_cost = cost
                    opt_latency = latency
                    opt_action = action
        res[d_name] = (opt_latency, opt_cost, opt_action)
    
    for key, re in res.items():
        print(f'{key:15} {re[0]:20} {re[1]:20}')
    for key, re in res.items():
        print(re[2])
    return res