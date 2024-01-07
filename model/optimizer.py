import torch
from model.params import getCustomParams

# Optimizer
def getOptim(args, model, vary_lyre=False, factor=1):  # Done
    """_summary_

    Args:
        model: A model object
        
        vary_lyre (bool, optional): 
        Whether to vary the learning rate of the layer.
        Defaults to False.
        
        factor (int, optional):
        The factor by which to multiply the learning rate of the lyre module.
        Defaults to 1.

    Returns:
        optimizer: An optimizer object
    """
    if vary_lyre:
        new_params, pre_params = getCustomParams(model)
        return torch.optim.AdamW(
            [
                {"params": new_params, "lr": args.learning_rate * factor},
                {"params": pre_params, "lr": args.learning_rate},
            ],
        )
    else:
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
        )