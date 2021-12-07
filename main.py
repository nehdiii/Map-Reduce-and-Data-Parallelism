# used libs
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from multiprocessing import Pool
import threading
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import os
import time

# we generate some random data
# data = np.random.randn(10000, 2) ## Col one=X values and Col is two=Y values
theta = np.zeros(2)  ## Model Parameters(Weights)


#################################
# calculation of the loss with mse (mean square error)
def loss_function(data, theta):
    # get m and b
    m = theta[0]
    b = theta[1]
    loss = 0
    # on each data point
    for i in range(0, len(data)):
        # get x and y
        x = data[i, 0]
        y = data[i, 1]
        # predict the value of y
        y_hat = (m * x + b)
        # compute loss
        loss = loss + ((y - (y_hat)) ** 2)
    # mean sqaured loss
    mean_squared_loss = loss / float(len(data))
    return mean_squared_loss


################################
# calculation of the gradient of each step in gradient descent algo
def compute_gradients(data, theta):
    gradients = np.zeros(2)
    # total number of data points
    N = float(len(data))
    m = theta[0]
    b = theta[1]
    # for each data point
    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        # gradient of loss function with respect to m
        gradients[0] += - (2 / N) * x * (y - ((m * x) + b))
        # gradient of loss funcction with respect to b
        gradients[1] += - (2 / N) * (y - ((theta[0] * x) + b))
    # add epsilon to avoid division by zero error
    epsilon = 1e-6
    gradients = np.divide(gradients, N + epsilon)
    return gradients


####################################################
# optimizing with normal sequential gradient descent
def optimization_Normal_Gradient_descent(data):
    theta = np.zeros(2)  # initializing theta
    tic = time.time()
    for t in range(500):  # will do 500 etr to converge
        # compute gradients
        gradients = compute_gradients(data, theta)
        # update parameter
        theta = theta - (1e-2 * gradients)
        # store the loss
    gr_loss = loss_function(data, theta)
    toc = time.time()
    print("cost calculated with normal excution: " + str(gr_loss))
    print("runnig time normal excution " + str(toc - tic))
    return (toc - tic)


########################################################
# we want to make a gain in running time so we use mapreduce and data parallelism technique to calculate the
# gradients in each step of gradient descent with multiprocessing useing process

def optimization_Multiprocessing_Gradient_descent(data):
    theta = np.zeros(2)
    jobs = np.array_split(data, 4, axis=0)  # we spleet the data set in 10 equal parts
    cd = partial(compute_gradients, theta=theta)
    """"
        partial : 
        Return a new partial object which when called will behave like func called with the positional arguments 
        args and keyword arguments keywords. If more arguments are supplied to the call, 
        they are appended to args. If additional keyword arguments are supplied, they extend and override keywords
        t3awen bech nejmo nhtot akther men parametre lel fn map 
    """
    pool = Pool(4)
    # Process Pools : One can create a pool of processes which will carry out tasks submitted to it with the Pool class.
    tic = time.time()
    for t in range(500):
        # compute gradients
        # map bech tlansi les process li san3thom pool wo jobs houma les data set eli 9smenhom
        gradients = sum(pool.map(cd,jobs))  # pool.map bech t5arjlna 10 valeur eli houma les some ta3 gradient li 7esbeto chaque process
        # update parameter
        theta = theta - (1e-2 * gradients)
        # store the loss
    gr_loss = loss_function(data, theta)
    toc = time.time()
    print("cost calculated with multiprocessing :"+str(gr_loss))
    print("runnig time with multiprocessor " + str(toc - tic))
    return (toc - tic)


######################################################################"

def optimization_Multithreading_Gradient_descent(data):
    theta = np.zeros(2)
    jobs = np.array_split(data,4,axis=0)
    tic = time.time()
    cd = partial(compute_gradients, theta=theta)  # meme logique que miltirocessing
    for t in range(500):
        # compute gradients
        with ThreadPoolExecutor(max_workers=4) as exe:  # meme logique pour cree pool voir doc of python
            gradients = sum(exe.map(cd, jobs))
        # update parameter
        theta = theta - (1e-2 * gradients)
        # store the loss
    gr_loss = loss_function(data, theta)
    toc = time.time()
    print("cost calculated with multithreading :" + str(gr_loss))
    print("runnig time with threading " + str(toc - tic))
    return (toc - tic)


###################################################################
def calculating_gaine():
    run_time_n = []
    run_time_multi = []
    run_time_multith = []
    for i in range(1,6):
        data = np.random.randn(pow(10, i), 2)
        run_time_n.append(optimization_Normal_Gradient_descent(data))
        run_time_multi.append(optimization_Multiprocessing_Gradient_descent(data))
        run_time_multith.append((optimization_Multithreading_Gradient_descent(data)))
        print('---------------------------------------------------------------------')
    return run_time_n, run_time_multi,run_time_multith


if __name__ == '__main__':
    tup = calculating_gaine()
    n, multi,multith = tup[0], tup[1],tup[2]
    x = np.arange(5)
    print(n, multi,multith)
    n, multi,multith = np.array(n), np.array(multi),np.array(multith)
    plt.plot(x, n, color="blue", label='normal')
    plt.plot(x, multi, color="green", label='multiprocessing')
    plt.plot(x,multith,color="red",label='multithreading')
    plt.legend()
    # plt.savefig("grape_of_run_time_gaine1.png",dpi=72)
    plt.show()
