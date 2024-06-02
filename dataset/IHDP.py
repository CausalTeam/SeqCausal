import pandas as pd
import numpy as np
import torch
import random
def read_ihdp(filepath):
    data = pd.read_csv(filepath + '/IHDP/ihdp_npci_1.csv', header = None)
    index = np.arange(747)
    random.shuffle(index)
    data = data.astype('float32')
    col =  ["treatment", "y_factual", "y_cfactual", "mu0", "mu1" ,]
    for i in range(1,26):
        col.append("x"+str(i))
    data.columns = col
    data = torch.Tensor(np.array(data))
    treatments,y,mu,features = data[:,0],data[:,1:3],data[:,3:5],data[:,5:]
    features = (features-features.mean(dim=0))/features.var(dim=0)
    y_fact = y[:,0]
    y_unfact = y[:,1]
    return features,treatments,y_fact,y_unfact,mu