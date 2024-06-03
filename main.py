import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from time import time
from models import get_model
import os
import random
from dataset import get_data
from environment import Env
from time import time
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
from agent import Agent
from inference import Inference
from environment import Env
from math import ceil
from tqdm import tqdm

class arg:
    def __init__(self) -> None:
        pass

args = arg()
args.model = 'simple'
args.dataset = 'IHDP'
args.train_lr = float(0.001)
args.missing_ratio = float(0)
args.task_id = int(2)
args.val_test_split = [0.25,0.25]
args.n_feature = int(58) if args.dataset == 'ACIC2016' else int(25)
args.disable_cuda = False
args.complete = False
args.pretrain = int(1000)
args.pretrain_sample = str('both')
args.mode = str('double')
args.decay = float(0.999)
args.gamma = float(1)
args.dropout = False
args.batchnorm = False
args.done_action_train = False
args.data_type = str('dependent_independent_complex')
args.p = float(0)
args.group_norm = float(0)
args.save_dir = str('result')
args.embedder_hidden_sizes = [32,32]
args.inf_hidden_sizes = [32,32]
args.policy_hidden_sizes = [32]
args.shared_dim = int(16)
args.target_update_freq = int(100)
args.eps_start = float(1.)
args.eps_end = float(0.1)
args.decay_rate = float(2)
args.n_env = int(32)
args.nsteps = int(4)
args.normalize = True
args.embedded_dim = int(16)
args.lstm_size = int(16)
args.n_shuffle = int(5)
args.r_cost = float(-1)
args.cost_from_file = False
args.random_seed = int(1)
args.batch_size = int(128)
args.message = str('')
args.buffer_size = int(10000)
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(args.random_seed)
else:
    args.device = torch.device('cpu')
args.save_dir = os.path.join(os.path.dirname(os.path.abspath('/data/tianeq/project/main.ipynb')),
        args.save_dir)   
args.save_path = args.data_type +  \
    "_nenv{}_nsteps{}_cost{}_norm{}".format(args.n_env, args.nsteps,
        args.r_cost, args.normalize) + \
    'eps_start{}end{}decay{}_'.format(args.eps_start, args.eps_end,
            args.decay_rate) + \
    'complete{}_doneactiontrain{}'.format(args.complete,
            args.done_action_train) + \
    'emb' + '_'.join('%03d' % num for num in args.embedder_hidden_sizes + \
            [args.embedded_dim]) + \
    'inf' + '_'.join('%03d' % num for num in args.inf_hidden_sizes) + \
    'policy' + '_'.join('%03d' % num for num in args.policy_hidden_sizes + \
            [args.shared_dim])
args.save_path = args.save_path + '_batch_size{}'.format(args.batch_size)
if args.dropout:
    args.save_path = args.save_path + \
    '_dropout{}'.format(args.p)
if len(args.message) > 0:
    args.save_path += args.message
args.save_path +='lstm{}_'.format(args.lstm_size) + 'shuffle{}_'.format(args.n_shuffle)
if args.pretrain:
    args.save_path += '_pretrain{}_{}'.format(args.pretrain,
            args.pretrain_sample)
if args.batchnorm:
    args.save_path += '_batchnorm'
args.save_path = os.path.join(args.save_dir, args.save_path)
args.csv_path = args.save_path
args.save_path = args.save_path + 'seed{}'.format(args.random_seed)
args.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'dataset')
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

def exp_gen(agent,inference,env,val_env,n_step,record = None):
    env.reset()
    agent.replay_buffer.reset()
    inference.replay_buffer.reset()
    mb_acquired = [[] for _ in range(env.n_env)]
    mb_next_acquired = [[] for _ in range(env.n_env)]
    mb_observe = [[] for _ in range(env.n_env)]
    mb_next_observe = [[] for _ in range(env.n_env)]
    mb_actions = [[] for _ in range(env.n_env)]
    mb_rewards = [[] for _ in range(env.n_env)]
    mb_done = [[] for _ in range(env.n_env)]
    for step in tqdm(range(n_step)):
        acquired = env.acquired.clone()
        next_acquired = env.acquired.clone()
        observe = env.observe.clone()
        next_observe = env.observe.clone()
        treatments = env.treatments.clone()
        y_fact = env.y_fact.clone()
        _,actions = agent.select_action(observe,acquired,agent.eps)
        env.step(actions)
        rewards = env.rewards.clone()
        done = env.terminal.clone()
        next_acquired[~done] = env.acquired.clone()[~done]
        next_observe[~done] = env.observe.clone()[~done]
        for i in range(env.n_env):
            mb_acquired[i].append(acquired[i])
            mb_next_acquired[i].append(next_acquired[i])
            mb_observe[i].append(observe[i])
            mb_next_observe[i].append(next_observe[i])
            mb_actions[i].append(actions[i])
            mb_rewards[i].append(rewards[i])
            mb_done[i].append(done[i])
        
        if record != None:
            record.push(torch.cat([observe,acquired,actions.unsqueeze(1),rewards.unsqueeze(1)],dim=-1))
            
        for store_id in torch.where(done)[0]:
            agent.replay_buffer.push([list(zip(mb_observe[store_id],mb_acquired[store_id],mb_actions[store_id],mb_rewards[store_id],
                                               mb_next_observe[store_id],mb_next_acquired[store_id],mb_done[store_id]))])
            mb_observe[store_id],mb_acquired[store_id],mb_actions[store_id],mb_rewards[store_id]=[],[],[],[]
            mb_next_observe[store_id],mb_next_acquired[store_id],mb_done[store_id]=[],[],[]
            inference.replay_buffer.push(list(zip(observe[done],acquired[done],treatments[done],y_fact[done])))
    val_env.reset()
    agent.val_buffer.reset()
    inference.val_buffer.reset()
    mb_acquired = [[] for _ in range(val_env.n_env)]
    mb_next_acquired = [[] for _ in range(val_env.n_env)]
    mb_observe = [[] for _ in range(val_env.n_env)]
    mb_next_observe = [[] for _ in range(val_env.n_env)]
    mb_actions = [[] for _ in range(val_env.n_env)]
    mb_rewards = [[] for _ in range(val_env.n_env)]
    mb_done = [[] for _ in range(val_env.n_env)]
    for step in tqdm(range(ceil((val_env.dataset.n_data)*(val_env.n_action)/val_env.n_env + val_env.n_action))):
        acquired = val_env.acquired.clone()
        next_acquired = val_env.acquired.clone()
        observe = val_env.observe.clone()
        next_observe = val_env.observe.clone()
        treatments = val_env.treatments.clone()
        y_fact = val_env.y_fact.clone()
        _,actions = agent.select_action(observe,acquired,agent.eps)
        val_env.step(actions)
        rewards = val_env.rewards.clone()
        done = val_env.terminal.clone()
        next_acquired[~done] = val_env.acquired.clone()[~done]
        next_observe[~done] = val_env.observe.clone()[~done]
        for i in range(val_env.n_env):
            mb_acquired[i].append(acquired[i])
            mb_next_acquired[i].append(next_acquired[i])
            mb_observe[i].append(observe[i])
            mb_next_observe[i].append(next_observe[i])
            mb_actions[i].append(actions[i])
            mb_rewards[i].append(rewards[i])
            mb_done[i].append(done[i])
            
        if val_env.states.isnan().all():
            break
        if record != None:
            record.push(torch.cat([observe,acquired,actions.unsqueeze(1),rewards.unsqueeze(1)],dim=-1))
            
        for store_id in torch.where(done)[0]:
            agent.val_buffer.push([list(zip(mb_observe[store_id],mb_acquired[store_id],mb_actions[store_id],mb_rewards[store_id],
                                               mb_next_observe[store_id],mb_next_acquired[store_id],mb_done[store_id]))])
            mb_observe[store_id],mb_acquired[store_id],mb_actions[store_id],mb_rewards[store_id]=[],[],[],[]
            mb_next_observe[store_id],mb_next_acquired[store_id],mb_done[store_id]=[],[],[]
            inference.val_buffer.push(list(zip(observe[done],acquired[done],treatments[done],y_fact[done])))

def test(agent,inference,testenv):
    print('start_test')
    test_start = time()
    agent.model.eval()
    testenv.reset()
    mse_tau = torch.empty(0).to(agent.device)
    mse_y_fact = torch.empty(0).to(agent.device)
    n_feature = torch.empty(0).to(agent.device)
    n = ceil((testenv.dataset.n_data)*(testenv.n_action)/testenv.n_env + testenv.n_action)
    for epoch in range(n):
        observe = testenv.observe.clone()
        acquired = testenv.acquired.clone()
        treatments = testenv.treatments.clone()
        y_fact = testenv.y_fact.clone()
        y_cf = testenv.y_cf.clone() if hasattr(testenv,'y_cf') else None
        _,actions = agent.select_action(observe,acquired,0)
        testenv.step(actions)
        rewards = testenv.rewards.clone()
        done = testenv.terminal.clone()
        if done.any():
            y_hat = inference.model.get_y(observe[done],acquired[done])
            if hasattr(testenv,'y_cf'):
                tau_hat = y_hat[:,1] - y_hat[:,0]
                tau = torch.where(treatments[done].bool(),y_fact[done]-y_cf[done],y_cf[done]-y_fact[done])
                mse_tau = torch.cat([mse_tau,(nn.MSELoss(reduction= 'none')(tau_hat,tau.detach()))])
            
            y_fact_hat = torch.where(treatments[done].bool(),y_hat[:,1],y_hat[:,0])
            mse_y_fact = torch.cat([mse_y_fact,(nn.MSELoss(reduction= 'none')(y_fact_hat,y_fact[done].detach()))])
            n_feature = torch.cat([n_feature,acquired[done].sum(-1)])
        if testenv.states.isnan().all():
            break
    print('finish_test')
    print('time_use:',time()-test_start)
    if hasattr(testenv.dataset,'y_cf'):
        print('mse of tau:',mse_tau.nanmean())
    print('mse of y_fact:',mse_y_fact.nanmean())
    print('mean_n_feature:',n_feature.nanmean())
    if hasattr(testenv.dataset,'y_cf'):
        return mse_tau.nanmean(),n_feature.nanmean()
    else:
        return mse_y_fact.nanmean(),n_feature.nanmean()   

class samples_buffer:
    def __init__(self,capacity) -> None:
        self.capacity = capacity
        self.buffer = torch.empty(0)
        self.counter = 0
    
    def push(self,new_data):
        new_data = new_data.to(self.buffer.device)
        if self.counter != 0:
            assert self.buffer.shape[1:] == new_data.shape[1:], f"input dim {new_data.shape[1:]} must be same as recorded dim {self.buffer.shape[1:]}"
        n_new_data = new_data.shape[0]
        if n_new_data == 0:
            return
        if self.counter + n_new_data <= self.capacity: 
            self.buffer = torch.cat([self.buffer,new_data],dim=0)
            self.counter = self.counter + n_new_data
        else:
            self.buffer = torch.cat([self.buffer,new_data],dim=0)
            if self.counter < self.capacity:
                n_drop = self.counter + n_new_data - self.capacity
                prob = torch.cat([n_drop*torch.ones(self.buffer.shape[0])],dim=0)
                drop_ind = WeightedRandomSampler(prob,n_drop,replacement=False)
            else :
                prob = torch.cat([n_new_data*torch.ones(self.capacity),(self.counter - self.capacity)*torch.ones(n_new_data)],dim=0)
                drop_ind = WeightedRandomSampler(prob,n_new_data,replacement=False)
            
            remain_ind = torch.tensor(list(set(range(self.buffer.shape[0])) - set(drop_ind)))
            self.buffer = self.buffer[remain_ind]
            self.counter = self.counter + n_new_data
            
    def sample(self,num):
        if self.counter > self.capacity:
            index = random.sample(list(torch.arange(self.capacity)),num)
        else:
            index = random.sample(list(torch.arange(self.counter)),num)
        return self.buffer[index]
    
args.X_mode,args.T_mode,args.Y_mode = args.data_type.split('_')
traindata,testdata,valdata = get_data(args)


model = get_model(args)
inf = Inference(model,'T_mode',args,5000)
agent = Agent(model,args,5000)
try: 
    model.load(os.path.join(args.save_path, "pretrained_best.model"))
except:
    inf.pretrain(traindata,valdata,args,args.pretrain,args.batch_size)


train_env = Env(args.n_env,traindata,model,args.r_cost)
val_env = Env(args.n_env,valdata,model,args.r_cost)
test_env = Env(args.n_env,testdata,model,args.r_cost)
with open(args.save_path+'/result.txt',"w") as file:
    pass
for eps in np.arange(1,0,-0.05):
    agent.eps = eps
    print(eps)
    record_buffer = samples_buffer(capacity=10000)
    with open(args.save_path+'/result.txt',"a") as file:
        file.write("current eps:{}\n".format(eps))
        file.write("exp_gen start:\n")
    start_time = time()
    exp_gen(agent,inf,train_env,val_env,1000,record_buffer)
    time_use = time()-start_time
    with open(args.save_path+'/result.txt',"a") as file:
        file.write("exp_gen finish!time use:{}\n".format(time_use))

    observe = record_buffer.buffer[:,:25].clone()
    acquired = record_buffer.buffer[:,25:50].clone()
    action = record_buffer.buffer[:,50].clone()
    reward = record_buffer.buffer[:,51].clone()
    n_feature = acquired.sum(dim=-1)
    uplift = torch.zeros(25)
    for i in range(25):
        mask = (action == i)
        uplift[i] = reward[mask].mean()
    plt.bar(range(25),uplift.detach().cpu().numpy(),width=0.4)
    fig = plt.gcf()  #获取当前图像
    fig.savefig(args.save_path+'/reward_dist_'+'eps{}.png'.format(agent.eps))
    fig.clear()

    record_buffer = samples_buffer(capacity=10000)
    inf.train(args,epochs=500,record=record_buffer)

    record_buffer = samples_buffer(capacity=10000)
    agent.train(args.batch_size,500,args,record=record_buffer)

    mean_mse,mean_n_feature = test(agent,inf,test_env)
    with open(args.save_path+'/result.txt',"a") as file:
        file.write("test finish!\nmean_mse:{}\nmean_n_feature:{}\n".format(mean_mse,mean_n_feature))  
