import numpy as np
import math
from pyswarm import pso
import utils


def train(train_data, train_labels, mod, hyperparams):
    """
    """

    # Set hyperparameters:
    lam, eta, batch_size = (hyperparams['lam'],hyperparams['eta'],hyperparams['batch_size'])

    if 'swarm_size' in list(hyperparams.keys()):
        swarm_size =  hyperparams['swarm_size']
    else:
        swarm_size = 70

    num_epochs = 20
    n = len(train_data[0])
    # print("n: %d" % n)

    #Save parameters
    params = list(2*np.random.rand(mod.count)-1)
    dim = len(params)
    print("Number of parameters in circuit: %d " % dim)

    bounds = (np.zeros(dim), 2*math.pi*np.ones(dim))

    xopt, fopt = pso(mod.get_loss, bounds[0], bounds[1], args=(train_data, train_labels, lam, eta, batch_size), swarmsize=swarm_size, maxiter=num_epochs)
    print(xopt, fopt)
    params = xopt
    return params, mod
