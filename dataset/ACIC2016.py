import pandas as pd
import numpy as np
import torch
def read_ACIC2016(filepath,task_id):
    features = pd.read_csv(filepath+"/ACIC2016/x.csv")
    mapping = {chr(ord('A') + i): i + 1 for i in range(26)}
    columns_to_convert = ['x_2', 'x_21','x_24']
    for col in columns_to_convert:
        features[col] = features[col].map(mapping)
    features = torch.Tensor(np.array(features).astype(float))
    zymu = pd.read_csv(filepath+'/ACIC2016/zymu_{}.csv'.format(task_id))
    zymu = torch.Tensor(np.array(zymu).astype(float))
    treatments,y,mu = zymu[:,0],zymu[:,1:3],zymu[:,3:]
    features = (features-features.mean(dim=0))/features.var(dim=0)
    y_fact = torch.where(treatments.bool(),y[:,1],y[:,0])
    y_cf = torch.where(~treatments.bool(),y[:,1],y[:,0])
    return features,treatments,y_fact,y_cf,mu


    
