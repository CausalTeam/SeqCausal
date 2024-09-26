# %%
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
import seaborn as sns
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# %%
class arg:
    def __init__(self) -> None:
        pass

# %%
args = arg()
args.random_seed = int(22)
args.model = 'simple'
args.dataset = 'simu_data'
args.sigma = float(0.3)
args.dist = float(2)
args.inf_lr = 0.003
args.agent_lr = 0.001
args.val_test_split = [0.25,0.25]
args.n_feature = int(58) if args.dataset == 'ACIC2016' else int(25)
args.disable_cuda = False
args.pretrain = int(10000)
args.decay = float(0.9)
args.alpha = float(1)
args.gamma = float(1)
args.inf_nepoch = int(100)
args.agent_nepoch = int(500)
args.data_type = str('dpeak_dependent_complex2')
args.inf_hidden_sizes = [512,512]
args.policy_hidden_sizes = [256,256]
args.target_update_freq = int(50)
args.eps_start = float(1)
args.eps_end = float(0.1)
args.n_env = int(32)
args.r_cost = float(1.0)
args.batch_size = int(512)
args.buffer_size = int(10000)
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(args.random_seed)
else:
    args.device = torch.device('cpu')
args.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'result')
if args.dataset == 'simu_data':
    args.save_path = args.dataset + '_' + args.data_type
elif args.dataset == 'ACIC2016':
    args.save_path = args.dataset + '_' + str(args.task_id)
else:
    args.save_path = args.dataset
args.save_path = os.path.join(args.save_dir, args.save_path)
args.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'dataset')
args.save_path = args.save_path + '_epsdecay{}'.format(args.decay)
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# %%
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
        y_cf = env.y_cf.clone()
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
            inference.replay_buffer.push(list(zip(observe[done],acquired[done],treatments[done],y_fact[done],y_cf[done])))
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
        y_cf = val_env.y_cf.clone()
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
            
        for store_id in torch.where(done)[0]:
            agent.val_buffer.push([list(zip(mb_observe[store_id],mb_acquired[store_id],mb_actions[store_id],mb_rewards[store_id],
                                               mb_next_observe[store_id],mb_next_acquired[store_id],mb_done[store_id]))])
            mb_observe[store_id],mb_acquired[store_id],mb_actions[store_id],mb_rewards[store_id]=[],[],[],[]
            mb_next_observe[store_id],mb_next_acquired[store_id],mb_done[store_id]=[],[],[]
            inference.val_buffer.push(list(zip(observe[done],acquired[done],treatments[done],y_fact[done],y_cf[done])))


# %%
def test(agent,inference,testenv):
    print('start_test')
    test_start = time()
    agent.model.eval()
    testenv.reset()
    mse_tau = torch.empty(0).to(agent.device)
    loss = torch.empty(0).to(agent.device)
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
            with torch.no_grad():
                y_hat = inference.model.get_y(observe[done],acquired[done])
            tau_hat = y_hat[:,1] - y_hat[:,0]
            tau = torch.where(treatments[done].bool(),y_fact[done]-y_cf[done],y_cf[done]-y_fact[done])
            mse_tau = torch.cat([mse_tau,(nn.MSELoss(reduction= 'none')(tau_hat,tau.detach()))])
            y_0 = torch.where(treatments[done].bool(),y_cf[done],y_fact[done])
            y_1 = torch.where(treatments[done].bool(),y_fact[done],y_cf[done])
            loss = torch.cat([loss,nn.MSELoss(reduction= 'none')(y_hat[:,0], y_0.detach()) + nn.MSELoss(reduction= 'none')(y_hat[:,1],y_1.detach())])
            n_feature = torch.cat([n_feature,acquired[done].sum(-1)])
        if testenv.states.isnan().all():
            break
    print('finish_test')
    print('time_use:',time()-test_start)
    if hasattr(testenv.dataset,'y_cf'):
        print('mse of tau:',mse_tau.nanmean())
    print('loss:',loss.nanmean())
    print('mean_n_feature:',n_feature.nanmean())
    if hasattr(testenv.dataset,'y_cf'):
        return mse_tau.nanmean(),n_feature.nanmean()
    else:
        return loss.nanmean(),n_feature.nanmean()   

# %%
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

# %%
args.X_mode,args.T_mode,args.Y_mode = args.data_type.split('_')
traindata,testdata,valdata = get_data(args)

# %%
model = get_model(args)

# %%
inf = Inference(model,'T_mode',args,args.buffer_size)
agent = Agent(model,args,args.buffer_size)

# %%
try: 
    model.load(os.path.join(args.save_path, "pretrained_best.model"))
except:
    inf.pretrain(traindata,valdata,args,args.pretrain,args.batch_size)

# %%
traindata.pred_ycf(model)
valdata.pred_ycf(model)
testdata.pred_ycf(model)
args.alpha = ((traindata.mu[:,1]-traindata.mu[:,0]).var()/args.n_feature)
print(args.alpha)
# %%
train_env = Env(args.n_env,traindata,model,args.alpha)
val_env = Env(args.n_env,valdata,model,args.alpha)
test_env = Env(args.n_env,testdata,model,args.alpha)

# %%
with open(args.save_path+'/result.txt',"w") as file:
    file.write("\n".join(f"{key}: {value}" for key, value in vars(args).items())+"\n")
all_start_time = time()
for _ in np.arange(30):
    agent.eps = max(args.decay*agent.eps,args.eps_end)
    print(agent.eps)
    record_buffer = samples_buffer(capacity=10000)
    with open(args.save_path+'/result.txt',"a") as file:
        file.write("current eps:{}\n".format(agent.eps))
        file.write("exp_gen start:\n")
    start_time = time()
    exp_gen(agent,inf,train_env,val_env,1000,record_buffer)
    time_use = time()-start_time
    with open(args.save_path+'/result.txt',"a") as file:
        file.write("exp_gen finish!time use:{}\n".format(time_use))

    record_buffer = samples_buffer(capacity=10000)
    inf.train(args,epochs=args.inf_nepoch,record=record_buffer)
    
    mean_mse,mean_n_feature = test(agent,inf,test_env)
    with open(args.save_path+'/result.txt',"a") as file:
        file.write("test finish!\nmean_mse:{}\nmean_n_feature:{}\n".format(mean_mse,mean_n_feature)) 

    record_buffer = samples_buffer(capacity=10000)
    with open(args.save_path+'/result.txt',"a") as file:
        file.write("current eps:{}\n".format(agent.eps))
        file.write("exp_gen start:\n")
    start_time = time()
    exp_gen(agent,inf,train_env,val_env,1000,record_buffer)
    time_use = time()-start_time
    with open(args.save_path+'/result.txt',"a") as file:
        file.write("exp_gen finish!time use:{}\n".format(time_use))

    record_buffer = samples_buffer(capacity=10000)
    agent.train(args.batch_size,args.agent_nepoch,args,val_env,record=record_buffer)

    mean_mse,mean_n_feature = test(agent,inf,test_env)
    with open(args.save_path+'/result.txt',"a") as file:
        file.write("test finish!\nmean_mse:{}\nmean_n_feature:{}\n".format(mean_mse,mean_n_feature))      
all_time_use = time()-all_start_time
with open(args.save_path+'/result.txt',"a") as file:
    file.write("all finish!all time use:{}\n".format(all_time_use))                                   


