# pylint: disable=multiple-statements, invalid-name, line-too-long, E1101
# Imports required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

# Dataset is included in sklearn
from sklearn.datasets import load_boston

# time measurement
from timeit import default_timer as timer

miniStep = 1000

## gradient descent
def grad_descent(x1,y,theta,iter_num,alpha):
    m = len(y)
    costs=[]
    for i in range(iter_num):
        theta += x1.T*(y - x1*theta)*alpha/m
        cost = 1/2/m * (x1*theta-y).T*(x1*theta-y)
        if (i%miniStep)==0: 
            costs.append(np.asscalar(cost))
            # print('iteration = %i, cost = %.8f' %(i,cost)) 
    return theta, costs

if __name__ == '__main__':
    # Load dataset
    boston = load_boston()
    ## prepare training model data
    x = boston.data
    y = boston.target
    ## data preprocessing for normalization
    x = x / x.max(axis=0)

    ## convert y into (506. 1)
    m = len(y)   # number of examples
    y = y.reshape(m,1)

    ## in order to add theta0 for A0=1, just add one column ahead of x1
    x1 = np.insert(x,0,1,axis=1)

    ## hyper parameters
    alpha_list = [10**x for x in range(-7, 0)]
    # alpha = 0.01 # learning rate
    num_epoch = 40000
    i = 1
    
    for alpha in alpha_list:
        # measure training time
        start = timer()
        theta = np.matrix(np.zeros((x.shape[1]+1,1)))  # initialize theta
        new_theta,costs = grad_descent(x1, y, theta, num_epoch, alpha)
        steps = [x * miniStep for x in range(1, int(num_epoch/miniStep) + 1)]
        end = timer()

        print('{0} Execution total time: {1:.3g} seconds'.format(i, end - start))
        plt.figure(i)
        plt.plot(steps, costs, 'ro', steps, costs, 'k')
        i = i+1
    
    plt.show()
    