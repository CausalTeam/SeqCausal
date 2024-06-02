import numpy as np
import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def data_split_n_ready(features, treatments, y_fact, y_cf=None, mu=None,random_seed=123,val_test_split=np.array([0.25, 0.25]),shuffle=True):
    val_test_split = np.array(val_test_split)
    dataset_size = len(features)
    indices = list(range(dataset_size))
    assert np.sum(val_test_split) < 1
    split = np.floor(val_test_split * dataset_size).astype(int)
    split = np.cumsum(split)
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices = indices[split[1]:]
    val_indices = indices[:split[0]]
    test_indices = indices[split[0]:split[1]]
    
    if y_cf != None and mu != None:
        train_set = Simu_Dataset(features[train_indices],treatments[train_indices],y_fact[train_indices],y_cf[train_indices],mu[train_indices],iter=True,shuffle=True)
        test_set = Simu_Dataset(features[test_indices],treatments[test_indices],y_fact[test_indices],y_cf[test_indices],mu[test_indices])
        valid_set = Simu_Dataset(features[val_indices],treatments[val_indices],y_fact[val_indices],y_cf[val_indices],mu[val_indices])
    else:
        train_set = BaseDataset(features[train_indices],treatments[train_indices],y_fact[train_indices],iter=True,shuffle=True)
        test_set = BaseDataset(features[test_indices],treatments[test_indices],y_fact[test_indices])
        valid_set = BaseDataset(features[val_indices],treatments[val_indices],y_fact[val_indices])
    return train_set,test_set,valid_set


class BaseDataset:
    def __init__(self, features, treatments, y_fact, iter=False, shuffle=True, min_dist=5, max_neighbor=3):
        self.features = features
        self.treatments = treatments
        self.y_fact = y_fact
        self.n_data, self.n_feature = features.shape
        self.index = 0
        self.iter = iter
        self.shuffle = shuffle
        distances = euclidean_distances(self.features, self.features)
        np.fill_diagonal(distances, np.inf)
        mask = (self.treatments.unsqueeze(0) + self.treatments.unsqueeze(-1))%2
        distances[mask==0] = np.inf
        neighbors = []
        for i in range(self.n_data):
            indices = np.argsort(distances[i,:])
            indices = indices[0:max_neighbor]
            indices[distances[i,indices]>min_dist] = -1
            neighbors.append(indices.tolist())
        self.neighbor = np.array(neighbors)
        
    def __len__(self):
        return self.n_data
        
    def __getitem__(self,index):
        return self.features[index], self.treatments[index], self.y_fact[index]
    
    def next_batch(self, batch_size, need_neighbor = False):
        new_index = (self.index + batch_size) % self.n_data

        if self.index + batch_size <= self.n_data:
            features = self.features[self.index: self.index + batch_size]
            treatments = self.treatments[self.index: self.index + batch_size]
            y_fact = self.y_fact[self.index: self.index + batch_size]
            neighbor = self.neighbor[self.index: self.index + batch_size]
        else:
            features = self.features[self.index:]
            treatments = self.treatments[self.index:]
            y_fact = self.y_fact[self.index:]
            neighbor = self.neighbor[self.index:]
            if self.iter:
                if self.shuffle:
                    self.permut()
                features = np.concatenate((features,self.features[:new_index]), axis=0)
                treatments = np.concatenate((treatments,self.treatments[:new_index]), axis=0)
                y_fact = np.concatenate((y_fact,self.y_fact[:new_index]), axis=0)
                neighbor = np.concatenate((neighbor,self.neighbor[:new_index]), axis=0)
            else:
                features = np.concatenate((features,np.full_like(self.features[:new_index],np.nan)), axis=0)
                treatments = np.concatenate((treatments,np.full_like(self.treatments[:new_index],np.nan)), axis=0)
                y_fact = np.concatenate((y_fact,np.full_like(self.y_fact[:new_index],np.nan)), axis=0)
                neighbor = np.concatenate((neighbor,np.full_like(self.neighbor[:new_index],-1)), axis=0)
        self.index = new_index if self.iter else min([self.n_data,self.index+batch_size])
        features = torch.Tensor(features)
        treatments = torch.Tensor(treatments)
        y_fact = torch.Tensor(y_fact)
        neighbor = torch.Tensor(neighbor).int()
        if need_neighbor:
            return features, treatments, y_fact, neighbor, self.features[neighbor], self.y_fact[neighbor]
        else:
            return features, treatments, y_fact
            
    def permut(self,p=None):
        p = np.random.permutation(self.n_data) if p==None else p
        self.features = self.features[p]
        self.treatments = self.treatments[p]
        self.y_fact = self.y_fact[p]
        self.neighbor = np.where(self.neighbor[p] != -1,p[self.neighbor[p]],-1)
        return p
    
class Simu_Dataset(BaseDataset):
    def __init__(self, features, treatments, y_fact, y_cf, mu, iter=False, shuffle=True, min_dist=5, max_neighbor=3):
        super().__init__(features, treatments, y_fact, iter, shuffle, min_dist, max_neighbor)
        self.y_cf = y_cf
        self.mu = mu
        
    def permut(self, p=None):
        p = super().permut(p)
        self.y_cf = self.y_cf[p]
        self.mu = self.mu[p]
        return p
    
    def next_batch(self, batch_size, need_neighbor=False):
        new_index = (self.index + batch_size) % self.n_data

        if self.index + batch_size <= self.n_data:
            features = self.features[self.index: self.index + batch_size]
            treatments = self.treatments[self.index: self.index + batch_size]
            y_fact = self.y_fact[self.index: self.index + batch_size]
            neighbor = self.neighbor[self.index: self.index + batch_size]
            y_cf = self.y_cf[self.index: self.index + batch_size]
        else:
            features = self.features[self.index:]
            treatments = self.treatments[self.index:]
            y_fact = self.y_fact[self.index:]
            neighbor = self.neighbor[self.index:]
            y_cf = self.y_cf[self.index:]
            if self.iter:
                if self.shuffle:
                    self.permut()
                features = np.concatenate((features,self.features[:new_index]), axis=0)
                treatments = np.concatenate((treatments,self.treatments[:new_index]), axis=0)
                y_fact = np.concatenate((y_fact,self.y_fact[:new_index]), axis=0)
                neighbor = np.concatenate((neighbor,self.neighbor[:new_index]), axis=0)
                y_cf = np.concatenate((y_cf,self.y_cf[:new_index]), axis=0)
            else:
                features = np.concatenate((features,np.full_like(self.features[:new_index],np.nan)), axis=0)
                treatments = np.concatenate((treatments,np.full_like(self.treatments[:new_index],np.nan)), axis=0)
                y_fact = np.concatenate((y_fact,np.full_like(self.y_fact[:new_index],np.nan)), axis=0)
                neighbor = np.concatenate((neighbor,np.full_like(self.neighbor[:new_index],-1)), axis=0)
                y_cf = np.concatenate((y_cf,np.full_like(self.y_cf[:new_index],np.nan)), axis=0)
        self.index = new_index if self.iter else min([self.n_data,self.index+batch_size])
        features = torch.Tensor(features)
        treatments = torch.Tensor(treatments)
        y_fact = torch.Tensor(y_fact)
        y_cf = torch.Tensor(y_cf)
        neighbor = torch.Tensor(neighbor).int()
        if need_neighbor:
            return features, treatments, y_fact, y_cf, neighbor, self.features[neighbor], self.y_fact[neighbor]
        else:
            return features, treatments, y_fact, y_cf
        
        
        
    
        