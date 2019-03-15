import numpy as np
import math
from pyswarm import pso
import utils
from scipy.optimize import differential_evolution

def train(train_data, train_labels, mod, hyperparams): #mutate=>[0,2], recombine=>[0,1]
    """
    """

    # Set hyperparameters:
    lam, eta, batch_size = (hyperparams['lam'],hyperparams['eta'],hyperparams['batch_size'])

    if 'mutate' in list(hyperparams.keys()):
        mutate = hyperparams['mutate']
    else:
        mutate = 1.6960892059264037
    if 'recombine' in list(hyperparams.keys()):
        recombine = hyperparams['recombine']
    else:
        recombine = 0.7311832672620681
    max_iter = 50
    #n = len(train_data[0])
    # print("n: %d" % n)

    #Save parameters
    params = list(2*math.pi*np.random.rand(mod.count))
    dim = len(params)
    print("Number of parameters in circuit: %d " % dim)

    bounds = (np.zeros(dim), 2*math.pi*np.ones(dim))

    xopt, fopt = differential_evolution(mod.get_loss, tuple(zip(bounds[0],bounds[1])), args=(train_data, train_labels, lam, eta, batch_size), maxiter = max_iter, mutation=mutate, recombination=recombine, tol= 0.000002)
    print(xopt, fopt)
    params = xopt
    return params, mod
