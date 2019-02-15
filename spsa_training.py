import numpy as np
import math
from noisyopt import minimizeSPSA


import numpy as np
import utils

def train(train_data, train_labels, mod, hyperparams):
    """
    """

    # Set hyperparameters:
    lam, eta, batch_size, = (hyperparams['lam'],hyperparams['eta'],hyperparams['batch_size'])

    if 'a' in list(hyperparams.keys()):
        a = hyperparams['a']
    else:
        a = .03

    if 'c' in list(hyperparams.keys()):
        a = hyperparams['c']
    else:
        c = .1
    num_iterations = 40 #100



    n = len(train_data[0])
    print("n: %d" % n)

    #Save parameters
    params = list(2*math.pi*np.random.rand(mod.count))
    dim = len(params)
    print("Number of parameters in circuit: %d " % dim)

    bounds = (np.zeros(dim), 2*math.pi*np.ones(dim))

    args = (train_data, train_labels, lam, eta, batch_size)
    result = minimizeSPSA(mod.get_loss, args=args, x0=params, paired=True, niter=num_iterations, a=a, c=c)#, disp=True)#, bound=bounds)

#    xopt, fopt = pso(model.get_loss, bounds[0], bounds[1], args=(n, train_data, train_labels, lam, eta, batch_size), swarmsize=swarm_size, maxiter=num_epochs)'
    print(result)
    params = result.x
    print("number of training circuit runs = ", mod.num_runs)
    return params, mod
