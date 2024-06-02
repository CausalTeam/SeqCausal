
def get_model(args):
    if args.model == 'JAFA':
        from .JAFA_model import SeqCausalNet
        model = SeqCausalNet(args)
    elif args.model == 'simple':
        from .simple_model import Model
        model = Model(args)
    else:
        raise Exception
    return model