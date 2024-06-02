import numpy as np
import torch
import numpy as np
import torch.distributions as dist

def gen_data(X_mode, T_mode, Y_mode, n_feature=25, n_data=20000, sigma=0.1, pro_score=0.2, seed=123):
    assert X_mode in ['independent','dependent'], 'X_mode should be "independent" or "dependent"!'
    assert T_mode in ['independent','dependent'], 'T_mode should be "independent" or "dependent"!'
    assert Y_mode in ['simple','linear','complex'], 'Y_mode shoule be "simple" or "linear" or "complex"!'
    if X_mode == 'independent':
        features = torch.randn([n_data,n_feature])
        # features = 2*torch.rand([n_data,n_feature])-1
        covariance_matrix = torch.diag(torch.ones(n_feature))
    else:
        mean = torch.zeros(n_feature)
        A = torch.randn([n_feature,n_feature])
        B = A.T@A
        C = torch.diagflat(B.diag()**(-1/2))
        covariance_matrix = C@A.T@A@C.T
        features_distribution = dist.MultivariateNormal(mean, covariance_matrix)
        features = features_distribution.sample((n_data,))
    if T_mode == 'independent':
        treatments = torch.Tensor(np.random.binomial(1,pro_score,size = n_data))
    else:
        k = torch.randn([n_feature])
        pro_score = torch.sigmoid((k*features).sum(-1))*pro_score*2
        treatments = torch.binomial(torch.ones(n_data),pro_score)
    if Y_mode == 'simple':
        mu_1 = 5*features[:,0] + 2*features[:,10] + 4*features[:,20]
        mu_0 = 5*features[:,0] + features[:,5] + 3*features[:,15]
    elif Y_mode == 'linear':
        k_1 = torch.randn([n_feature])
        k_0 = torch.randn([n_feature])
        k_1 = k_1/((k_1**2).sum().sqrt())*(21**(0.5))
        k_0 = k_0/((k_0**2).sum().sqrt())*(11**(0.5))
        mu_1 = (k_1*features).sum(-1)
        mu_0 = (k_0*features).sum(-1)
    else:
        k_1 = torch.randn([n_feature])
        k_0 = torch.randn([n_feature])
        k_1 = k_1/((k_1**2)*(1+(covariance_matrix.roll(-1,0).diag())**2)).sum().sqrt()*(21**(0.5))
        k_0 = k_0/((k_0**2)*(1+(covariance_matrix.roll(-1,0).diag())**2)).sum().sqrt()*(11**(0.5))
        mu_1 = (features*k_1*features.roll(1,1)).sum(-1)
        mu_0 = (features*k_0*features.roll(1,1)).sum(-1)
    mu = torch.cat([mu_0.reshape(-1,1),mu_1.reshape(-1,1)],dim=-1)
    y_0 = mu_0 + sigma*torch.randn(n_data)
    y_1 = mu_1 + sigma*torch.randn(n_data)
    y_fact = torch.where(treatments.bool(),y_1,y_0)
    y_cf = torch.where((1-treatments).bool(),y_1,y_0)
    return features, treatments, y_fact, y_cf, mu 