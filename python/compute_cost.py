def compute_cost(AL, Y):
    """
    Implement the cost function defined by the above equation.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (0 or 1), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.

    cost = - (1/m) * np.sum(Y * np.log(AL) + (1-Y) * np.log(1 - AL))
        
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost