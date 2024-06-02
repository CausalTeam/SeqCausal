import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, bias=True,
            dropout=False, p=0, group_norm=0, batch_norm=False):
        super(MLP, self).__init__()
        self.layers = []
        self.n_feature = int(input_size / 2)
        in_size = input_size
        cnt = 0
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(in_size, hidden_size, bias=bias))
            if group_norm > 0 and cnt == 0:
                cnt += 1
                self.w0 = self.layers[-1].weight
                print(self.w0.size())
                assert self.w0.size()[1] == input_size
            if batch_norm:
                print("Batchnorm")
                self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.ReLU())
            if dropout: # for classifier
                print("Dropout!")
                assert p > 0 and p < 1
                self.layers.append(nn.Dropout(p=p))
            in_size = hidden_size
        self.layers.append(nn.Linear(in_size, output_size, bias=bias))
        if batch_norm: # FIXME is it good?
            print("Batchnorm")
            self.layers.append(nn.BatchNorm1d(output_size))
        self.layers = nn.ModuleList(self.layers)

        self.output_size = output_size


    def forward(self, x, length=None):
        for layer in self.layers:
            x = layer(x)
        return x
    
class Model(nn.Module):
    def __init__(self, args) -> None:
        
        super(Model, self).__init__()
        self.device = args.device
        self.inference_0 = MLP(2*args.n_feature,args.inf_hidden_sizes,1)
        self.inference_1 = MLP(2*args.n_feature,args.inf_hidden_sizes,1)
        self.policy = MLP(2*args.n_feature,args.policy_hidden_sizes,args.n_feature + 1)
        self.n_feature = args.n_feature
        self.n_action = args.n_feature + 1   
        self.to(self.device)
        
    def get_q(self, observes, acquired):
        q_val = self.policy(torch.cat([observes, acquired],dim = -1))
        return q_val
        
    def get_y(self, observes, acquired):
        y_0 = self.inference_0(torch.cat([observes, acquired],dim = -1))
        y_1 = self.inference_1(torch.cat([observes, acquired],dim = -1))
        return torch.cat([y_0.reshape(-1,1),y_1.reshape(-1,1)],dim = -1)
    
    def save(self, save_path):
        torch.save(self.state_dict(), save_path)
        
    def load(self, load_path):
        self.load_state_dict(torch.load(load_path))
    