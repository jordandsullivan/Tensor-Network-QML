import numpy as np
import math
import numpy as np
import utils
from tqdm import tqdm
import data
import time

def train(train_data, train_labels, mod, hyperparams, params=None):
    """
    """

    locals().update(hyperparams) #More convenient for calling variables

    # n = len(train_data[0])
    # print("n: %d" % n)

    #Save parameters
    if params is None:
        print("No params")
        params = np.random.normal(loc=0.0, scale=1.0, size=mod.count)
    else:
        print("Params provided")
        #2*np.random.random(mod.count)-1

        #2*math.pi*np.random.rand(mod.count)
    dim = len(params)
    print("Number of parameters in circuit: %d " % dim)

    v = np.zeros(params.shape)
    print("Number of training samples =", len(train_data))
    print("Batch size =", batch_size)
    for k in tqdm(range(num_iterations)):
        #Good choice for Delta is the Radechemar distribution acc 2*np.random.random(dim)-1to wiki
        delta = 2*np.random.random(dim)-1#np.random.binomial(n=1, p=0.5, size=dim)
        alpha = a/(k+1+A)**s
        beta = b/(k+1)**t
        # perturb = params + alpha*delta
        batch_iter = data.batch_generator(train_data, train_labels, batch_size)
        j = 0
        for (images, labels) in batch_iter:
            print("Epoch ", k, " batch ", j)
            j+=1
            perturb = params + alpha*delta
            # start = time.time()
            L1 = mod.get_loss(perturb, images, labels, lam, eta, len(images))
            perturb = params - alpha*delta
            L2 = mod.get_loss(perturb, images, labels, lam, eta, len(images))
            # end = time.time()
            # print("Time for update for a single batch =", end-start)
            g = (L1-L2)/(2*alpha)
            v = gamma*v - g*beta*delta
            params = params + v
            utils.save_params(params)
        # L1 = mod.get_loss(perturb, train_data, train_labels, lam, eta, batch_size)
        # perturb = params - alpha*delta
        # L2 = mod.get_loss(perturb, train_data, train_labels, lam, eta, batch_size)
        # g = (L1-L2)/(2*alpha)
        # v = gamma*v - g*beta*delta
        # params = params + v
        # utils.save_params(params)



    print(params)

    print("number of training circuit runs = ", mod.num_runs)
    return params, mod
