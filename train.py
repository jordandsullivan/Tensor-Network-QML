import pyswarm_training
import spsa_training
import evolve_training
import sigopt_training

import test
import argparse
import utils
import numpy as np
import model_simulate_wf as model
import time

start = time.time()

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument("--savefile", help="filename to save params", type=str)
# parser.add_argument("--data", help="bw, grey, or mnist", type = str, default = 'mnist')
parser.add_argument("--method", help="pyswarm or spsa", type = str, default = 'spsa')
parser.add_argument("--sigopt", help="optimze hyperparameters: True or False", type = bool, default = False)
args = parser.parse_args()

train_data, train_labels, test_data, test_labels = utils.load_data('mnist')
hyperparams = dict({'lam':0.023953588780914498, 'eta':4.465676125689359, 'batch_size':18})

params = None
n = len(train_data[0])
mod = model.Model(n=n, num_trials=1)

if args.sigopt:
    hyperparams, mod = sigopt_training.train(train_data, train_labels, mod, args.method, hyperparams)
elif args.method == 'spsa':
    params, mod = spsa_training.train(train_data, train_labels, mod, hyperparams)
elif args.method == 'pyswarm':
    params, mod = pyswarm_training.train(train_data, train_labels, mod, hyperparams)
elif args.method == 'evolve':
    params, mod = evolve_training.train(train_data, train_labels, mod, hyperparams)
else:
    print("invalid optimization method")
    exit(0)

if params is not None:
    np.savetxt("params/" + args.savefile, params, delimiter = ",")
test.get_accuracy(test_data, test_labels, mod, params)
print('Total training time:',(time.time()-start)/60,'minutes')
