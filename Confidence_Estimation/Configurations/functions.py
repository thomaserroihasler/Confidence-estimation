from definitions import OPTIMIZERS

def create_optimizer(optimizer_name, model_params, lr, momentum=0, weight_decay=0):
    optimizer_class = OPTIMIZER_CONFIG[optimizer_name]['optimizer']
    optimizer_args = OPTIMIZER_CONFIG[optimizer_name]['args'](model_params, lr, momentum, weight_decay)
    return optimizer_class(model_params, **optimizer_args)

