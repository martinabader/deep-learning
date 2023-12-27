def sigmoid(Z):

    """
    Implement a sigmoid activation function.

    Arguments:
    Z -- input to the activation function (linear prediction, pre-activation parameter): (size of current layer, 1)

    Returns:
    a -- the output of the activation function
    cache -- a python tuple containing "Z"
    """

    a = 1 / (1 + np.exp(-Z))

    cache = (Z)

    return a, Z