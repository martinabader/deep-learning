def L_model_forward(X, parameters):
    """
    Implements forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implements [LINEAR -> RELU]*(L-1). 
    # The for loop starts at 1 because layer 0 is the input
    for l in range(1, L):
        
        A_prev = A 

        A, cache = linear_activation_forward(A_prev, 
                                             parameters['W' + str(l)], 
                                             parameters['b' + str(l)], 
                                             activation = "relu")
        # adds "cache" to the "caches" list
        caches.append(cache)      
        
    
    # Implements LINEAR -> SIGMOID. 
    AL, cache = linear_activation_forward(A, 
                                          W = parameters['W' + str(L)], 
                                          b = parameters['b' + str(L)], 
                                          activation = "sigmoid")
    # adds "cache" to the "caches" list
    caches.append(cache)  
    
          
    return AL, caches