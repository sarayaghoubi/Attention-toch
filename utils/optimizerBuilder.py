import torch.optim as opt

list_of_optimizers = {
    'adam': opt.Adam,
    'sgd': opt.SGD,

}


def build_optimizer(model, cfg):
    optimizer = opt.list_of_optimizers(model.parameters(),cfg.args)
    return optimizer
