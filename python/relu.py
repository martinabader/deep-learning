def relu(Z):

    """
    Implement a ReLU activation function.

    Arguments:
    Z -- input to the activation function (linear prediction, pre-activation parameter): (size of current layer, 1)

    Returns:
    a -- the output of the activation function
    cache -- a python tuple containing "Z"
    """
    a = np.maximum(0, Z)

    cache = (Z)

    return a, Z