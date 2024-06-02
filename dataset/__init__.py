
def get_data(args):
    if args.dataset == 'simu_data':
        from .simu_data import gen_data
        features,treatments,y_facts,y_cf,mu = gen_data(args.X_mode,args.T_mode,args.Y_mode)
    elif args.dataset == 'IHDP':
        from .IHDP import read_ihdp
        features,treatments,y_facts,y_cf,mu = read_ihdp(args.data_path)
    elif args.dataset == 'ACIC2016':
        from .ACIC2016 import read_ACIC2016
        features,treatments,y_facts,y_cf,mu = read_ACIC2016(args.data_path,args.task_id)
    else:
        raise Exception
    from .data_deal import data_split_n_ready
    if 'y_cf' in locals():
        trainset,valset,testset = data_split_n_ready(features,treatments,y_facts,y_cf,mu,random_seed=args.random_seed,val_test_split=args.val_test_split)
    else:
        trainset,valset,testset = data_split_n_ready(features,treatments,y_facts,random_seed=args.random_seed,val_test_split=args.val_test_split)
    return trainset,valset,testset