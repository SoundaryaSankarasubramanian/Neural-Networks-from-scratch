
'''
python 3.6
'''
import numpy as np
from load_mnist import mnist
import matplotlib.pyplot as plt
import pdb

def tanh(Z):
    '''
    computes tanh activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.tanh(Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def tanh_der(dA, cache):
    '''
    computes derivative of tanh activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    ### CODE HERE
    Z = cache["Z"]
    A, Cache = tanh(Z)
    dZ = dA *(1-A*A)
    return dZ

def sigmoid(Z):
    '''
    computes sigmoid activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = 1/(1+np.exp(-Z))
    cache = {}
    cache["Z"] = Z
    return A, cache

def sigmoid_der(dA, cache):
    '''
    computes derivative of sigmoid activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    ### CODE HERE
    Z = cache['Z']
    A, cache = sigmoid(Z)


    dZ = dA * A * (1-A)
    return dZ

def initialize_2layer_weights(n_in, n_h, n_fin):
    '''
    Initializes the weights of the 2 layer network

    Inputs: 
        n_in input dimensions (first layer)
        n_h hidden layer dimensions
        n_fin final layer dimensions

    Returns:
        dictionary of parameters
    '''
    # initialize network parameters
    ### CODE HERE

    parameters = {}
    W1=np.random.randn(n_h,n_in) * np.sqrt(1/n_h)
    W2=np.random.randn(n_fin,n_h) * np.sqrt(1/n_fin)
    b1=np.random.randn(n_h,1) * np.sqrt(1/n_h)
    b2=np.random.randn(n_fin,1) * np.sqrt(1/n_fin)
    parameters["W1"] = W1
    parameters["b1"] = b1
    parameters["W2"] = W2
    parameters["b2"] = b2

    return parameters

def linear_forward(A, W, b):
    '''
    Input A propagates through the layer 
    Z = WA + b is the output of this layer. 

    Inputs: 
        A - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer

    Returns:
        Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache - a dictionary containing the inputs A, W and b
        to be used for derivative
    '''
    ### CODE HERE
    Z = np.dot(W,A)+b
    cache = {}
    cache["A"] = A
    cache["W"] = W
    cache["b"] = b
    return Z, cache

def layer_forward(A_prev, W, b, activation):
    '''
    Input A_prev propagates through the layer and the activation

    Inputs: 
        A_prev - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
        activation - is the string that specifies the activation function

    Returns:
        A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache - a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
    '''
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, act_cache = sigmoid(Z)
    elif activation == "tanh":
        A, act_cache = tanh(Z)
    
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache

    return A, cache

def cost_estimate(A2, Y):
    '''
    Estimates the cost with prediction A2

    Inputs:
        A2 - numpy.ndarray (1,m) of activations from the last layer
        Y - numpy.ndarray (1,m) of labels
    
    Returns:
        cost of the objective function
    '''
    ### CODE HERE
    m = Y.shape[0]
    c1 = np.dot(Y,np.log(A2+1e-8).T)
    c2 = np.dot((1 - Y), np.log((1 - A2)+1e-8).T)

    cost = (np.mean(-(c1 + c2)))/m
    

    return cost

def linear_backward(dZ, cache, W, b):
    '''
    Backward propagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz 
        cache - a dictionary containing the inputs A
            where Z = WA + b,    
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)  
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    # CODE HERE
    A = cache["A"]
    dW = np.dot(dZ,A.T)
    db = np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        W - numpy.ndarray (n,p)  
        b - numpy.ndarray (n, 1)
    
    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def classify(X, parameters):
    '''
    Network prediction for inputs X

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    ### CODE HERE

    A1, cache1 = layer_forward(X, parameters["W1"], parameters["b1"], "sigmoid")
    YPred, cache2 = layer_forward(A1, parameters["W2"], parameters["b2"], "sigmoid")
    return np.round(YPred)

def two_layer_network(X, Y, net_dims, num_iterations=2000, learning_rate=0.1):
    '''
    Creates the 2 layer network and trains the network

    Inputs:
        X - numpy.ndarray (n,m) of training data
        Y - numpy.ndarray (1,m) of training data labels
        net_dims - tuple of layer dimensions
        num_iterations - num of epochs to train
        learning_rate - step size for gradient descent
    
    Returns:
        costs - list of costs over training
        parameters - dictionary of trained network parameters
    '''
    n_in, n_h, n_fin = net_dims
    parameters = initialize_2layer_weights(n_in, n_h, n_fin)
    
    A0 = X
    costs = []
    for ii in range(num_iterations):
        # Forward propagation
        ### CODE HERE
        A1, C1 = layer_forward(A0,parameters["W1"],parameters["b1"],"sigmoid")
        A2, C2 = layer_forward(A1,parameters["W2"],parameters["b2"],"sigmoid")
        # cost estimation
        ### CODE HERE
        cost = cost_estimate(A2,Y)
        # Backward Propagation
        ### CODE HERE
        num1 = Y/(A2+1e-8)
        num2 = (Y-1)/((A2+1e-8)-1)
        m = Y.shape[0]
        dA2 = (-num1+num2)/m

        dA1, dW2, dB2 = layer_backward(dA2,C2,parameters["W2"],parameters["b2"],"sigmoid")
        dA0, dW1, dB1 = layer_backward(dA1,C1,parameters["W1"],parameters["b1"],"sigmoid")


        #update parameters
        ### CODE HERE
        parameters["W1"] = parameters["W1"] - (learning_rate * dW1)
        parameters["b1"] = parameters["b1"] - (learning_rate * dB1)
        parameters["W2"] = parameters["W2"] - (learning_rate * dW2)
        parameters["b2"] = parameters["b2"] - (learning_rate * dB2)

        if ii % 10 == 0:
            costs.append(cost)
        if ii % 100 == 0:
            print("Cost at iteration %i is: %f" %(ii, cost))
    
    return costs, parameters

def main():
    # getting the subset dataset from MNIST
    # binary classification for digits 1 and 7
    digit_range = [1,7]
    data, data_label, test_data, test_label = \
            mnist(noTrSamples=2400,noTsSamples=1000,\
            digit_range=digit_range,\
            noTrPerClass=1200, noTsPerClass=500)
    print(data_label)

    validation_costs=[]
    test_accuracies=[]

    train_data = np.concatenate((data[:,:1000],data[:,1200:2200]),axis = 1)
    val_data = np.concatenate((data[:,1000:1200],data[:,2200:2400]),axis = 1)
    train_label = np.concatenate((data_label[0][:1000],data_label[0][1200:2200]))
    val_label = np.concatenate((data_label[0][1000:1200],data_label[0][2200:2400]))
    #convert to binary labels
    train_label[train_label==digit_range[0]] = 0
    train_label[train_label==digit_range[1]] = 1
    test_label[test_label==digit_range[0]] = 0
    test_label[test_label==digit_range[1]] = 1
    val_label[val_label==digit_range[0]] = 0
    val_label[val_label==digit_range[1]] = 1


    n_in, m = train_data.shape
    n_fin = 1
    n_h = [100,200,500]
    count = 1
    num_iterations = 1000
    iteration = [k for k in range(0,num_iterations,10)]
    #fig = plt.figure(figsize=(15, 10))
    fig = plt.figure()

    for n,j in enumerate(n_h):
        net_dims = [n_in, j, n_fin]
        # initialize learning rate and num_iterations
        learning_rate = 0.01
        out=[]
        out_t=[]
        out_v=[]

        

        print("Number of neurons in the hidden layer:",j)
        print()

        print("Training set")
        costs, parameters = two_layer_network(train_data, train_label, net_dims, \
                num_iterations=num_iterations, learning_rate=learning_rate)
        print("------------------------------------------------------------------------------")
        print("Validation set")

        costs_val,parameters_val = two_layer_network(val_data, val_label, net_dims, \
                num_iterations=num_iterations, learning_rate=learning_rate)
        validation_costs.append(costs_val[-1])
        print("------------------------------------------------------------------------------")
        
        # compute the accuracy for training set and testing set
        train_Pred = classify(train_data, parameters)
        val_Pred = classify(val_data,parameters)
        test_Pred = classify(test_data, parameters)


        trAcc = None
        teAcc = None
        
        output_test = (test_label == test_Pred)
        teAcc = (np.sum(output_test == True)/test_label.shape[1])*100
        test_accuracies.append(teAcc)


        output = (train_label == train_Pred)
        trAcc=(np.sum(output == True)/train_label.shape[0])*100

        

        output_val = (val_label == val_Pred)
        valAcc = (np.sum(output_val == True)/val_label.shape[0])*100
        print("Accuracy:")


        print("Accuracy for training set is {0:0.3f} %".format(trAcc))
        print("Accuracy for validation set is {0:0.3f} %".format(valAcc))
        print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
        print("------------------------------------------------------------------------------")


    # CODE HERE TO PLOT costs vs iterations

        if j==100:

            sub1 = fig.add_subplot(1, 3, 1)
            sub1.plot(iteration,costs,label="Train")
            sub1.plot(iteration,costs_val,label="Validation")
            sub1.legend()
            sub1.set_title('100 Neurons in the hidden layer')
            sub1.set_xlabel('Iterations')
            sub1.set_ylabel('Cost')
        if j==200:

            sub2 = fig.add_subplot(1, 3, 2)
            sub2.plot(iteration,costs,label="Train")
            sub2.plot(iteration,costs_val,label="Validation")
            sub2.legend()
            sub2.set_title('200 Neurons in the hidden layer')
            sub2.set_xlabel('Iterations')
            sub2.set_ylabel('Cost')

        if j==500:
            sub3 = fig.add_subplot(1, 3, 3)
            sub3.plot(iteration,costs,label="Train")
            sub3.plot(iteration,costs_val,label="Validation")
            sub3.legend()
            sub3.set_title('500 Neurons in the hidden layer')
            sub3.set_xlabel('Iterations')
            sub3.set_ylabel('Cost')
            plt.show()

    idx = validation_costs.index(min(validation_costs))
    print("Mimimum validation cost:",min(validation_costs),"observed with ",n_h[idx],"hidden layers")
    print("Test accuracy using the architecture with the best validation cost:",test_accuracies[idx])
    




if __name__ == "__main__":
    main()




