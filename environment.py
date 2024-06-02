import torch
import numpy as np
import torch.nn.functional as F

def prefect_pred(observe,acquired):
    
    y_1 = observe[:,0] + 2*observe[:,10] + 4*observe[:,20]
    y_0 = observe[:,0] + observe[:,5] + 3*observe[:,15] 
    return torch.concat([y_0.reshape(-1,1),y_1.reshape(-1,1)],dim=-1)
class Env:
    def __init__(self,n_env,dataset,model,r_cost=None):
        self.device = model.device    
        self.n_env = n_env
        self.dataset = dataset
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
        self.r_cost = r_cost if r_cost!=None else -torch.ones(self.n_feature).to(self.device)
        self.r_cost = torch.zeros(self.n_feature).to(self.device).fill_(r_cost)
        self.rewards = torch.zeros(self.n_env).to(self.device)
        self.mse_last = torch.zeros(self.n_env).to(self.device)
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
        self.mse_last = torch.zeros(self.n_env).to(self.device)
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
        y_hat = self.model.get_y(self.observe,self.acquired)
        # y_hat = prefect_pred(self.observe,self.acquired)
        y_f_hat = torch.where(self.treatments.bool(),y_hat[:,1],y_hat[:,0])
        mask = self.neighbors != -1
        observe_neighbor = self.states_neighbor.view(-1,self.n_feature)*self.acquired.repeat(1,self.states_neighbor.shape[1]).view(-1,self.n_feature)
        y_cf = self.y_fact_neighbor
        y_cf_hat = torch.zeros_like(y_cf)
        treatments_neighbor = (1 - self.treatments.unsqueeze(-1).repeat(1,self.states_neighbor.shape[1])[mask].view(-1,1)).to(torch.int64)
        y_cf_hat[mask] = self.model.get_y(observe_neighbor[mask.view(-1)],self.acquired.repeat(1,self.states_neighbor.shape[1]).view(-1,self.n_feature)[mask.view(-1)]).gather(1,treatments_neighbor).view(-1)
        #  y_cf_hat[mask] = prefect_pred(observe_neighbor[mask.view(-1)],self.acquired.repeat(1,self.states_neighbor.shape[1]).view(-1,self.n_feature)[mask.view(-1)]).gather(1,treatments_neighbor).view(-1)
        y_cf_hat[~mask] = torch.nan
        mse_y_cf = ((y_cf_hat-y_cf)**2).nanmean(dim=-1)
        mse_y_cf[mse_y_cf.isnan()] = 100
        mse_y_cf[mse_y_cf>50] = 100
        mse_tau = F.mse_loss(y_f_hat,self.y_fact.detach(),reduction='none') + mse_y_cf
        self.rewards[~terminal] = self.mse_last[~terminal] - mse_tau[~terminal] - self.r_cost[actions[~terminal]]
        self.rewards[terminal] = 0
        self.mse_last = mse_tau
            