import torch
import numpy as np
import torch.nn.functional as F

class Env:
    def __init__(self,n_env,dataset,model,alpha = 1,r_cost=None):
        self.device = model.device    
        self.n_env = n_env
        self.dataset = dataset
        self.alpha = alpha
        if hasattr(dataset,'y_cf'):
            states,treatments,y_fact,y_cf,neighbors,states_neighbor,y_fact_neighbor = self.dataset.next_batch(self.n_env,need_neighbor = True)
            self.y_cf = y_cf.to(self.device)
        else:
            states,treatments,y_fact,neighbors,states_neighbor,y_fact_neighbor = self.dataset.next_batch(self.n_env,need_neighbor = True)
        self.n_feature = states.shape[-1]
        self.n_action = self.n_feature + 1
        self.states = states.to(self.device)
        self.treatments = treatments.to(self.device)
        self.y_fact = y_fact.to(self.device)
        self.neighbors = neighbors.to(self.device)
        self.states_neighbor = states_neighbor.to(self.device)
        self.y_fact_neighbor = y_fact_neighbor.to(self.device)
        self.acquired = torch.zeros_like(self.states).int()
        self.observe = torch.masked_fill(self.states,~self.acquired.to(torch.bool),0)
        self.model = model
        self.r_cost = r_cost if r_cost!=None else torch.ones(self.n_feature).to(self.device)
        self.rewards = torch.zeros(self.n_env).to(self.device)
        with torch.no_grad():
            y_hat = self.model.get_y(self.observe,self.acquired)
            y_f_hat = torch.where(self.treatments.bool(),y_hat[:,1],y_hat[:,0])
            y_cf_hat = torch.where(self.treatments.bool(),y_hat[:,0],y_hat[:,1])
            mse_tau = F.mse_loss(y_f_hat,self.y_fact.detach(),reduction='none') + F.mse_loss(y_cf_hat,self.y_cf.detach(),reduction='none')
        self.mse_last = mse_tau
        self.terminal = torch.zeros(self.n_env).to(self.device)     
    def reset(self):
        self.dataset.index = 0
        if hasattr(self.dataset,'y_cf'):
            states,treatments,y_fact,y_cf,neighbors,states_neighbor,y_fact_neighbor = self.dataset.next_batch(self.n_env,need_neighbor = True)
            self.y_cf = y_cf.to(self.device)
        else:
            states,treatments,y_fact,neighbors,states_neighbor,y_fact_neighbor = self.dataset.next_batch(self.n_env,need_neighbor = True)
        self.states = states.to(self.device)
        self.treatments = treatments.to(self.device)
        self.y_fact = y_fact.to(self.device)
        self.neighbors = neighbors.to(self.device)
        self.states_neighbor = states_neighbor.to(self.device)
        self.y_fact_neighbor = y_fact_neighbor.to(self.device)
        self.acquired = torch.zeros_like(self.states).int()
        self.observe = torch.masked_fill(self.states,~self.acquired.to(torch.bool),0)
        self.rewards = torch.zeros(self.n_env).to(self.device)
        with torch.no_grad():
            y_hat = self.model.get_y(self.observe,self.acquired)
            y_f_hat = torch.where(self.treatments.bool(),y_hat[:,1],y_hat[:,0])
            y_cf_hat = torch.where(self.treatments.bool(),y_hat[:,0],y_hat[:,1])
            mse_tau = F.mse_loss(y_f_hat,self.y_fact.detach(),reduction='none') + F.mse_loss(y_cf_hat,self.y_cf.detach(),reduction='none')
        self.mse_last = mse_tau
        self.terminal = torch.zeros(self.n_env).to(self.device)
        
    
    def step(self,actions):
        self.terminal = (actions == self.n_feature)
        terminal  = self.terminal
        n_terminal = self.terminal.int().sum().item()
        if n_terminal!=0:
            if hasattr(self.dataset,'y_cf'):
                states,treatments,y_fact,y_cf,neighbors,states_neighbor,y_fact_neighbor = self.dataset.next_batch(n_terminal,need_neighbor = True)
                self.y_cf[terminal] = y_cf.to(self.device)
            else:
                states,treatments,y_fact,neighbors,states_neighbor,y_fact_neighbor = self.dataset.next_batch(n_terminal,need_neighbor = True)
            self.states[terminal] = states.to(self.device)
            self.treatments[terminal] = treatments.to(self.device)
            self.y_fact[terminal] = y_fact.to(self.device)
            self.neighbors[terminal] = neighbors.to(self.device)
            self.states_neighbor[terminal] = states_neighbor.to(self.device)
            self.y_fact_neighbor[terminal] = y_fact_neighbor.to(self.device)
            self.acquired[terminal] = torch.zeros_like(self.states[terminal]).int()
            self.observe[terminal] = torch.masked_fill(self.states[terminal],~self.acquired[terminal].to(torch.bool),0)
        self.acquired[~terminal,actions[~terminal]] = 1
        self.observe[~terminal,actions[~terminal]] = self.states[~terminal,actions[~terminal]]
        with torch.no_grad():
            y_hat = self.model.get_y(self.observe,self.acquired)
            y_f_hat = torch.where(self.treatments.bool(),y_hat[:,1],y_hat[:,0])
            y_cf_hat = torch.where(self.treatments.bool(),y_hat[:,0],y_hat[:,1])
            mse_tau = F.mse_loss(y_f_hat,self.y_fact.detach(),reduction='none') + F.mse_loss(y_cf_hat,self.y_cf.detach(),reduction='none')
        self.rewards[~terminal] = self.mse_last[~terminal] - mse_tau[~terminal] - self.alpha*(self.r_cost[actions[~terminal]])
        self.rewards[terminal] = 0
        self.mse_last = mse_tau
            