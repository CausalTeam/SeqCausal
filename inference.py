import os
import torch
from torch import nn
import pandas as pd
import numpy as np
import random
from torch.optim import Adam,SGD
from torch.utils.data import WeightedRandomSampler, BatchSampler, DataLoader, TensorDataset
from time import time
from tqdm import tqdm

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
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
        
class Inference(nn.Module):
    
    def __init__(self,model,mode,args,buffer_size = 10000):
        super(Inference, self).__init__()
        self.model = model
        self.mode = mode
        self.device = model.device
        self.n_feature = model.n_feature
        if mode == 'S_mode':
            inf_params = dict(model.inf.named_parameters())
            weight_params = [inf_params[key] for key in inf_params.keys() if 'weight' in key]
            bias_params = [inf_params[key] for key in inf_params.keys() if 'weight' not in key]         
        elif mode == 'T_mode':
            inf0_params = dict(model.inference_0.named_parameters())
            inf1_params = dict(model.inference_1.named_parameters())
            weight_params = [inf0_params[key] for key in inf0_params.keys() if 'weight' in key] +\
                [inf1_params[key] for key in inf1_params.keys() if 'weight' in key]
            bias_params = [inf0_params[key] for key in inf0_params.keys() if 'weight' not in key] +\
                [inf1_params[key] for key in inf1_params.keys() if 'weight' not in key]   
        else:
            raise Exception()

        mult = 1 if args.dropout else 0
        params = [{'params' : weight_params,
            'weight_decay' : mult * (1 - args.p) / args.batch_size},
            {'params' : bias_params,
            'weight_decay' : mult * 1 / args.batch_size}]
        if hasattr(self.model,'encoder'):
            params = params + [{'params':(model.encoder.parameters())}]
        self.optimizer = SGD(params,lr=args.train_lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.val_buffer = ReplayBuffer(capacity=1000)

        
    def train(self,args,epochs=10,batch_size=256,criterion = nn.MSELoss(),record = None):
        self = self.to(self.device)
        start_time = time()
        self.model.eval()
        transitions = self.val_buffer.buffer
        transitions = list(zip(*transitions))
        observes = torch.stack(transitions[0])
        acquired = torch.stack(transitions[1])
        treatments = torch.stack(transitions[2])
        y_fact = torch.stack(transitions[3])
        y_hat = self.model.get_y(observes,acquired)
        y_fact_hat = torch.where(treatments.bool(),y_hat[:,1],y_hat[:,0])
        val_loss = ((y_fact_hat-y_fact.detach())**2).nanmean()
        min_mse = val_loss
        self.model.train()

        with open(args.save_path+'/result.txt',"a") as file:
            file.write("start inf train:\n")
            for epoch in range(epochs):
                transitions = self.replay_buffer.sample(batch_size)
                transitions = list(zip(*transitions))
                observes = torch.stack(transitions[0])
                acquired = torch.stack(transitions[1])
                treatments = torch.stack(transitions[2])
                y_fact = torch.stack(transitions[3])
                y_hat = self.model.get_y(observes,acquired)
                y_fact_hat = torch.where(treatments.bool(),y_hat[:,1],y_hat[:,0])
                loss = criterion(y_fact_hat, y_fact.detach())
                if record != None:
                    record.push(torch.concat([observes,acquired,treatments.reshape(-1,1),y_fact.reshape(-1,1),y_hat],dim=-1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (epoch + 1) % 10 == 0:
                    self.model.eval()
                    transitions = self.replay_buffer.sample(batch_size)
                    transitions = list(zip(*transitions))
                    observes = torch.stack(transitions[0])
                    acquired = torch.stack(transitions[1])
                    treatments = torch.stack(transitions[2])
                    y_fact = torch.stack(transitions[3])
                    y_hat = self.model.get_y(observes,acquired)
                    y_fact_hat = torch.where(treatments.bool(),y_hat[:,1],y_hat[:,0])
                    val_loss = ((y_fact_hat-y_fact.detach())**2).nanmean()
                    if val_loss < min_mse:
                        print('epoch:',epoch+1,' train_loss:',loss.item(),' val_loss:',val_loss.item(),' SAVE')
                        self.model.save(os.path.join(args.save_path,"trained_best.model"))
                        file.write("epoch:{},train_loss:{},val_loss:{}\n".format(epoch+1,loss,val_loss))
                        min_mse = val_loss
                    self.model.train()
            train_time = time() - start_time
            file.write("Finish inf train!time use:{}".format(train_time))
        self.model.load(os.path.join(args.save_path,"trained_best.model"))
        return loss
    
    def pretrain(self,trainset,valset,args,epochs=500,batch_size=64,criterion = nn.MSELoss()):
        print('start_pretrain')
        pretrain_start = time()
        self.model.train()
        min_val_mse = torch.inf
        mode = 'B'
        for epoch in range(epochs):
            index = random.sample(range(trainset.n_data),batch_size)
            features,treatments,y_fact = trainset[index]
            features,treatments,y_fact = features.to(self.device),treatments.to(self.device),y_fact.to(self.device)
            if mode == 'A':
                acquired_dimension = random.sample(range(self.n_feature),random.randint(0,self.n_feature))
                acquired = torch.zeros_like(features)
                for dimension in acquired_dimension:
                    acquired[:,dimension].fill_(1)
            elif mode == 'B':
                acquired = torch.rand_like(features)
                acquired = (acquired>args.missing_ratio).float()
            else:
                raise Exception
            observe = features*acquired
            y_hat = self.model.get_y(observe,acquired)
            y_fact_hat = torch.where(treatments.bool(),y_hat[:,1],y_hat[:,0])
            loss = criterion(y_fact_hat, y_fact.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                self.model.eval()
                features,treatments,y_fact = valset[range(valset.n_data)]
                features,treatments,y_fact = features.to(self.device),treatments.to(self.device),y_fact.to(self.device)
                if mode == 'A':
                    acquired_dimension = random.sample(range(self.n_feature),random.randint(0,self.n_feature))
                    acquired = torch.zeros_like(features)
                    for dimension in acquired_dimension:
                        acquired[:,dimension].fill_(1)
                elif mode == 'B':
                    acquired = torch.rand_like(features)
                    acquired = (acquired>args.missing_ratio).float()
                else:
                    raise Exception
                observe = features*acquired
                y_hat = self.model.get_y(observe,acquired)
                y_fact_hat = torch.where(treatments.bool(),y_hat[:,1],y_hat[:,0])
                val_loss = criterion(y_fact_hat, y_fact.detach())
                if val_loss < min_val_mse:
                    print('epoch:',epoch+1,' train_loss:',loss.item(),' val_loss:' ,val_loss.item())
                    self.model.save(os.path.join(args.save_path,"pretrained_best.model"))
                    min_val_mse = val_loss
                self.model.train()
        # self.model.load(os.path.join(args.save_path,"pretrained_best.model"))
        pretrain_time = time() - pretrain_start
        print('pretrain_time:',pretrain_time)
        
    def test(self,dataset,args):
        self.model.eval()
        print('start_inference_test')
        start_time = time()
        dataset.index = 0
        mse_tau = torch.empty(0).to(self.device)
        mse_y_fact = torch.empty(0).to(self.device)
        sampler = BatchSampler(np.arange(dataset.n_data),128,drop_last=False)
        for indices in sampler:
            features = dataset.features[indices].to(self.device)
            treatments = dataset.treatments[indices].to(self.device)
            acquired = torch.rand_like(features)
            acquired = (acquired>args.missing_ratio).float()
            y_fact = dataset.y_fact[indices].to(self.device)
            y_hat = self.model.get_y(features,acquired)
            if hasattr(dataset,'y_cf'):
                y_cf = dataset.y_cf[indices].to(self.device)
                tau_hat = y_hat[:,1] - y_hat[:,0]
                tau = torch.where(treatments.bool(),y_fact-y_cf,y_cf-y_fact)
                mse_tau = torch.cat([mse_tau,(nn.MSELoss(reduction= 'none')(tau_hat,tau.detach()))])
            y_fact_hat = torch.where(treatments.bool(),y_hat[:,1],y_hat[:,0])
            mse_y_fact = torch.cat([mse_y_fact,(nn.MSELoss(reduction= 'none')(y_fact_hat,y_fact.detach()))])
            

        print('finish_inference_test')
        print('time_use:',time()-start_time)
        if hasattr(dataset,'y_cf'):
            print('mse of tau:',mse_tau.nanmean())
        print('mse of y_fact:',mse_y_fact.nanmean())
        return mse_tau.nanmean(),mse_y_fact.nanmean() if hasattr(dataset,'y_cf') else mse_y_fact.nanmean()
        
    
    def train_probpred(self,train_data,epochs):
        train_dataset = TensorDataset(train_data)
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        optimizer = Adam(self.model.probpred.parameters(), lr=1e-4)
        test_data = torch.randn([1000,train_data.shape[-1]],device = self.device)
        
        for epoch in range(epochs):
            for batch in train_dataloader:
                x, = batch  
                x = x.to(self.device)
                optimizer.zero_grad()
                log_prob = self.model.probpred.log_prob(x)
                loss = -log_prob.mean()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                log_prob_q = torch.log(((1/(torch.sqrt(torch.tensor(2*torch.pi))))**(self.n_feature)*torch.exp(-0.5*((test_data)**2).sum(dim=-1))))
                log_prob_p = self.model.probpred.log_prob(test_data)
                kl_divergence = (log_prob_q - log_prob_p).mean()  # 计算 KL 散度的近似值

            print(f'Test KL Divergence Approximation: {kl_divergence.item():.4f}')
            print(f'Epoch {epoch+1}/{epochs}, train_Loss: {loss.item():.4f}')
    
                
    