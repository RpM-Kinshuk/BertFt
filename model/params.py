import copy

# Copy Model Parameters
def get_model_params(model):  # Done
    """
    Args:
        model: A model object

    Returns:
        params: A dictionary of model parameters
    """
    params = {}
    for name in model.state_dict():
        params[name] = copy.deepcopy(model.state_dict()[name])
    return params

# Custom training for Parameters
def getCustomParams(model):  # Done
    """
    Args:
        model: A model object

    Returns:
        new_params: A list of new parameters
        pre_trained: A list of pre-trained parameters
    """
    new_params = []
    pre_trained = []
    for name, val in model.named_parameters():
        if "new_layer" in name or "classifier" in name:
            new_params.append(val)
        else:
            pre_trained.append(val)
    return new_params, pre_trained