import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD,Adam
import torch.nn.functional as F
from time import time
import os
import random
import copy
from time import time
from torch.utils.data import WeightedRandomSampler
from math import ceil

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = []
        self.counter = 0
    
    def push(self, new_data):
        n_new_data = len(new_data)
        if n_new_data == 0:
            return
        if self.counter + n_new_data <= self.capacity: 
            self.buffer = self.buffer + new_data
            self.counter = self.counter + n_new_data
        else:
            self.buffer = self.buffer + new_data
            if self.counter < self.capacity:
                n_drop = self.counter + n_new_data - self.capacity
                prob = torch.cat([n_drop*torch.ones(len(self.buffer))],dim=0)
                drop_ind = WeightedRandomSampler(prob,n_drop,replacement=False)
            else :
                prob = torch.cat([n_new_data*torch.ones(self.capacity),(self.counter - self.capacity)*torch.ones(n_new_data)],dim=0)
                drop_ind = WeightedRandomSampler(prob,n_new_data,replacement=False)
            
            for index in sorted(drop_ind, reverse=True):
                self.buffer.pop(index)
            self.counter = self.counter + n_new_data
            
    def sample(self, batch_size, mode = 'random'):
        assert mode == 'random' or 'near', 'mode should be "random" or "near"!'
        if mode == 'random':
            return random.sample(self.buffer, batch_size)
        else:
            prob = 1/(self.counter - torch.arange(self.counter))
            indices = list(WeightedRandomSampler(prob,batch_size,replacement=False))
            picked = [self.buffer[index] for index in indices]
            return picked
        
    def reset(self):
        self.buffer = []
        self.counter = 0

      
class Agent:
    def __init__(self,model,args,buffer_size=10000):
        self.model = model
        self.device = model.device
        self.n_feature =  model.n_feature
        self.n_action = model.n_feature + 1
        self.old_model = copy.deepcopy(self.model)
        self.eps = args.eps_start
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.val_buffer = ReplayBuffer(capacity=buffer_size//10)
        self.gamma = args.gamma
        params = model.policy.parameters()
        if hasattr(self,'encoder'):
            params.append(list(model.encoder.parameters()))
        self.optimizer = SGD(params, lr=args.agent_lr)

    
    def select_action(self,states,acquired,eps):
        
        q_val = self.model.get_q(states,acquired)
        q_val[:,:-1][acquired==1] = -torch.inf
        q_val_max,actions = q_val.max(dim = 1)
        noise = torch.rand_like(q_val_max)
        random_action = torch.multinomial(F.pad((1.-acquired),[0,1,0,0],value=1),1).squeeze()
        actions[noise<eps] = random_action[noise<eps]
        
        return q_val,actions
    
    '''
    def select_action(self,states,acquired,eps):
        
        q_val = self.model.get_q(states,acquired)
        q_val[:,:-1][acquired==1] = -torch.inf
        q_val_max,actions = q_val.max(dim = 1)
        noise = torch.rand_like(q_val_max)
        random_action = torch.multinomial(F.pad((1.-acquired),[0,1,0,0],value=1),1).squeeze()
        actions[noise<eps] = random_action[noise<eps]
        mask = acquired.int().sum(-1)>=1
        actions[mask] = self.n_feature
        
        return q_val,actions
    '''
               
    def train(self, batch_size, epochs, args,val_env,record = None):
        print('start_policy_train')
        train_start = time()
        self.model.train()

        with open(args.save_path+'/result.txt',"a") as file:
            file.write("start agent train:\n")
            with torch.no_grad():
                score = self.test(val_env)
                print(0, score.item(), "SAVE")
                file.write("epoch:0,agent_score:{}\n".format(score))
                self.model.train()
            for epoch in range(epochs):          
                if len(self.replay_buffer.buffer) < batch_size:
                    return
                transitions = self.replay_buffer.sample(batch_size)
                transitions = [(element,index) for index,sublist in enumerate(transitions) for element in sublist]
                index_batch = torch.Tensor(list(zip(*transitions))[1]).to(self.device).to(torch.int)
                transitions = list(zip(*transitions))[0]
                batch = list(zip(*transitions))
                observe_batch = torch.stack(batch[0]).to(self.device)
                acquired_batch = torch.stack(batch[1]).to(self.device)
                action_batch = torch.Tensor(batch[2]).to(self.device)
                action_batch = action_batch.to(torch.int64).to(self.device)
                reward_batch = torch.Tensor(batch[3]).to(self.device).reshape(-1,1)
                next_observe_batch = torch.stack(batch[4]).to(self.device)
                next_acquired_batch = torch.stack(batch[5]).to(self.device)
                done_batch = torch.Tensor(batch[6]).to(self.device).to(torch.bool)
                
                
                q_val = self.model.get_q(observe_batch,acquired_batch)
                q_values = q_val.gather(1,action_batch.unsqueeze(1))
                with torch.no_grad():
                    next_q_values = self.old_model.get_q(next_observe_batch,next_acquired_batch)
                able_choice = torch.cat([1-acquired_batch,torch.ones(len(action_batch),1).to(self.device)],dim=1).bool()
                next_q_values[~able_choice] = float('-inf')
                with torch.no_grad():
                    next_q_val = (self.model.get_q(next_observe_batch,next_acquired_batch))
                next_q_val[~able_choice] = float('-inf')
                best_action = next_q_val.argmax(-1)
                next_state_values=next_q_values.gather(1,best_action.unsqueeze(1))
                next_state_values[done_batch] = 0
                with torch.no_grad():
                    state_values = self.old_model.get_q(observe_batch,acquired_batch).gather(1,best_action.unsqueeze(1))
                state_values[done_batch] = 0
                target_q_values = (self.gamma * next_state_values).reshape(-1,1) + reward_batch - state_values.reshape(-1,1)
                # for offset in range(index_batch.bincount().max()-1):
                    # target_q_values = torch.where((index_batch==index_batch.roll(-1)).reshape(-1,1),torch.maximum(target_q_values,target_q_values.roll(-1)*self.gamma + reward_batch),target_q_values)
                    # target_q_values = torch.where((index_batch==index_batch.roll(-1)).reshape(-1,1),target_q_values.roll(-1)*self.gamma + reward_batch,target_q_values)
                    
                if record != None:
                    record.push(torch.cat([observe_batch,acquired_batch,action_batch.unsqueeze(1),target_q_values,q_val],dim=-1))
                loss = nn.MSELoss()(q_values, target_q_values.detach())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                
                if (epoch + 1) % args.target_update_freq == 0:
                    with torch.no_grad():
                        score = self.test(val_env)
                    print(epoch+1, score.item(), "SAVE")
                    self.model.save(os.path.join(args.save_path,"trained_best.model"))  
                    file.write("epoch:{},agent_loss:{},agent_score:{}\n".format(epoch+1,loss,score))
                    self.model.train()
                    self.old_model.load_state_dict(self.model.state_dict())
                
            train_time = time()-train_start
            print('policy_train_time:',train_time)
            file.write('Finish agent train!time use;{}\n'.format(train_time))
        
    def test(self, testenv, record=None):
        test_start = time()
        self.model.eval()
        testenv.reset()
        score = 0
        n = ceil((testenv.dataset.n_data)*(testenv.n_action)/testenv.n_env + testenv.n_action)
        for epoch in range(n):
            observe = testenv.observe.clone()
            acquired = testenv.acquired.clone()
            if record != None:
                treatments = testenv.treatments.clone()
                y_fact = testenv.y_fact.clone()
            _,actions = self.select_action(observe,acquired,0)
            testenv.step(actions)
            rewards = testenv.rewards.clone()
            done = testenv.terminal.clone()
            score += rewards.nansum()
            if record != None:
                record.push(torch.cat([observe[done],acquired[done],treatments[done].unsqueeze(-1),y_fact[done].unsqueeze(-1)],dim=-1))
            if testenv.states.isnan().all():
                break
        return score/(testenv.dataset.n_data)
            
            