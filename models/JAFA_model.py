import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence as pad
import torch.nn.functional as F

def state2input(states,acquired):
    
    one_hoted_acquired = torch.multiply(acquired,(torch.arange(acquired.shape[-1])+1).to(acquired.device)).to(torch.int64)
    one_hoted_acquired,indices = one_hoted_acquired.sort(dim=-1,descending=True)
    one_hoted_acquired = F.one_hot(one_hoted_acquired,acquired.shape[-1]+1)[:,:,1:]    
    inputs = torch.concat([states[torch.arange(states.shape[0]).unsqueeze(-1),indices].unsqueeze(-1),one_hoted_acquired],dim=-1)
    
    return inputs

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
    
    
class SetEncoder(nn.Module):
    def __init__(self, input_dim, n_feature, embedder_hidden_sizes, embedded_dim, lstm_size, n_shuffle, simple=True, proj_dim=None, normalize=False, dropout=False, p=0):
        # embedder + lstm
        super(SetEncoder, self).__init__()

        self.n_shuffle = n_shuffle
        self.embedder = MLP(input_dim, embedder_hidden_sizes, embedded_dim,
                dropout=dropout, p=p)
        self.lstm = nn.LSTMCell(embedded_dim, lstm_size)
        #self.module_list = nn.ModuleList([self.embedder, self.lstm])
        self.n_feature = n_feature
        self.normalize = normalize

        self.lstm_size = lstm_size
        self.embedded_dim = embedded_dim

        if not simple:
            assert proj_dim is not None
            self.attention = nn.ModuleList(
                [nn.Linear(lstm_size, proj_dim, bias=False),
                 nn.Linear(embedded_dim, proj_dim, bias=True),
                 nn.Linear(proj_dim, 1, bias=True)]
            )
        elif embedded_dim != lstm_size:
            self.attention = torch.nn.Linear(lstm_size, embedded_dim,
                    bias=False)
            # torch.nn.init.xavier_normal(self.attention.weight)
            # module.apply(weight_xavier_init)

        def _compute_attention_sum(q, m, length):
            # q : batch_size x lstm_size
            # m : batch_size x max(length) x embedded_dim
            assert torch.max(length) == m.size()[1]
            max_len = m.size()[1]
            if simple:
                if q.size()[-1] != m.size()[-1]:
                    q = self.attention(q) # batch_size x embedded_dim
                weight_logit = torch.bmm(m, q.unsqueeze(-1)).squeeze(2) # batch_size x n_feature
            else:
                linear_m = self.attention[1]
                linear_q = self.attention[0]
                linear_out = self.attention[2]

                packed = pack(m, list(length), batch_first=True)
                proj_m = PackedSequence(linear_m(packed.data), packed.batch_sizes)
                proj_m, _ = pad(proj_m, batch_first=True)  # batch_size x n_feature x proj_dim
                proj_q = linear_q(q).unsqueeze(1) # batch_size x 1 x proj_dim
                packed = pack(F.relu(proj_m + proj_q), list(length), batch_first=True)
                weight_logit = PackedSequence(linear_out(packed.data), packed.batch_sizes)
                weight_logit, _ = pad(weight_logit, batch_first=True) # batch_size x n_feature x 1
                weight_logit = weight_logit.squeeze(2)

            # max_len = weight_logit.size()[1]
            indices = torch.arange(0, max_len,out=torch.Tensor(max_len).unsqueeze(0)).to(length.device)
            # TODO here.. cuda..
            mask = indices < length.unsqueeze(1)#.long()
            weight_logit[~mask] = -np.inf
            weight = F.softmax(weight_logit, dim=1) # nonzero x max_len
            weighted = torch.bmm(weight.unsqueeze(1), m)
            # batch_size x 1 x max_len
            # batch_size x     max_len x embedded_dim
            # = batch_size x 1 x embedded_dim
            return weighted.squeeze(1), weight  #nonzero x embedded_dim

        self.attending = _compute_attention_sum


    def forward(self, state, length):
        # length should be sorted
        assert (state.size()).__len__() == 3 # batch x n_feature x input_dim
                                      # input_dim == n_feature + 1
        batch_size = state.size()[0]
        self.weight = np.zeros((int(batch_size), self.n_feature))#state.data.new(int(batch_size), self.n_feature).fill_(0.)
        nonzero = torch.sum(length > 0).cpu().numpy() # encode only nonzero points
        if nonzero == 0:
            return state.new(int(batch_size), self.lstm_size + self.embedded_dim).fill_(0.)

        length_ = list(length[:nonzero].cpu().numpy())
        packed = pack(state[:nonzero], length_, batch_first=True)

        embedded = self.embedder(packed.data)


        if self.normalize:
            embedded = F.normalize(embedded, dim=1)
        embedded = PackedSequence(embedded, packed.batch_sizes)
        embedded, _ = pad(embedded, batch_first=True) # nonzero x max(length) x embedded_dim

        # define initial state
        qt = embedded.new(embedded.size()[0], self.lstm_size).fill_(0.)
        ct = embedded.new(embedded.size()[0], self.lstm_size).fill_(0.)

        ###########################
        # shuffling (set encoding)
        ###########################

        for i in range(self.n_shuffle):
            attended, weight = self.attending(qt, embedded, length[:nonzero])
            # attended : nonzero x embedded_dim
            qt, ct = self.lstm(attended, (qt, ct))

        # TODO edit here!
        weight = weight.detach().cpu().numpy()
        tmp = state[:, :, 1:]
        val, acq = torch.max(tmp, 2) # batch x n_feature
        tmp = (val.long() * acq).cpu().numpy()
        #tmp = tmp.cpu().numpy()
        tmp = tmp[:weight.shape[0], :weight.shape[1]]
        self.weight[np.arange(nonzero).reshape(-1, 1), tmp] = weight

        encoded = torch.cat((attended, qt), dim=1)
        if batch_size > nonzero:
            encoded = torch.cat(
                (encoded,
                 encoded.new(int(batch_size - nonzero),
                     encoded.size()[1]).fill_(0.)),
                dim=0
            )
        return encoded
    
class DuelingNet(nn.Module):
    def __init__(self, encoded_dim, hidden_sizes, shared_dim, n_action,
            group_norm=0, batch_norm=False):
        super(DuelingNet, self).__init__()
        self.shared = MLP(encoded_dim, hidden_sizes, shared_dim,
                group_norm=group_norm, batch_norm=batch_norm)
        self.pi_net = MLP(shared_dim, [shared_dim], n_action)
        self.v_net = MLP(shared_dim, [shared_dim], 1)
        self.n_action = n_action

    def forward(self, encoded):
        tmp = self.shared(encoded)
        tmp = F.relu(tmp)
        self.adv = self.pi_net(tmp) # batch_size x n_actions
        self.v = self.v_net(tmp) # batch_size x 1

        output = self.v + (self.adv - torch.mean(self.adv, dim=1, keepdim=True))

        return output #torch.cat((self.pi, self.v), 1)
    
class SeqCausalNet(nn.Module):
    def __init__(self, args):
        # TODO data uncertainty handling
        super(SeqCausalNet, self).__init__()
        self.device = args.device
        self.encoder = SetEncoder(
                args.n_feature + 1, args.n_feature,args.embedder_hidden_sizes, args.embedded_dim,
                args.lstm_size, args.n_shuffle,normalize=args.normalize,dropout=args.dropout, p=args.p)
        self.inference_0 = MLP(args.lstm_size + args.embedded_dim, args.inf_hidden_sizes, 1,
                dropout=args.dropout, p=args.p,batch_norm=args.batchnorm)
        self.inference_1 = MLP(args.lstm_size + args.embedded_dim, args.inf_hidden_sizes, 1,
                dropout=args.dropout, p=args.p,batch_norm=args.batchnorm)
        self.policy = DuelingNet(args.lstm_size + args.embedded_dim, args.policy_hidden_sizes, args.shared_dim,
                args.n_feature + 1)
        self.n_feature = args.n_feature
        self.n_action = args.n_feature + 1
        self.to(self.device)
    
    def get_q(self, observes, acquired):
        inputs = state2input(observes,acquired)
        lengths = acquired.int().sum(dim=-1)
        sorted_, indices = torch.sort(lengths, -1, descending=True)
        _, invert = torch.sort(indices)
        assert (lengths==sorted_[invert]).all()
        inputs = inputs[indices]#.long()] # sort
        inputs = self.encoder(inputs, sorted_)

        q_val = self.policy(inputs)
        q_val = q_val[invert]  # if setencoding else q_val
        
        return q_val
    
    def get_y(self, observes, acquired):
        inputs = state2input(observes,acquired)
        lengths = acquired.int().sum(dim=-1)
        sorted_, indices = torch.sort(lengths, -1, descending=True)
        _, invert = torch.sort(indices)
        assert (lengths==sorted_[invert]).all()
        inputs = inputs[indices]#.long()] # sort
        inputs = self.encoder(inputs, sorted_)
        
        y_0 = self.inference_0(inputs)
        y_1 = self.inference_1(inputs)
        
        return torch.cat([y_0.reshape(-1,1),y_1.reshape(-1,1)],dim = -1)
             
    
    
    def save(self, save_path):
        torch.save(self.state_dict(), save_path)
        
    def load(self, load_path):
        self.load_state_dict(torch.load(load_path))
        
        
    
        