import pyswarm_training
import spsa_training
import evolve_training
import sigopt_training

import test
import argparse
import utils
import numpy as np

# import model_full_simulate_wf as model
import model
import model2
import time

start = time.time()

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument("--savefile", help="filename to save params", type=str)
parser.add_argument("--resumefile", help="filename to resume params from", type=str)
# parser.add_argument("--data", help="bw, grey, or mnist", type = str, default = 'mnist')
parser.add_argument("--method", help="pyswarm or spsa", type = str, default = 'spsa')
parser.add_argument("--model", help="balanced or cascade", type=str, default='balanced')
args = parser.parse_args()
classes = (0,1)
train_data, train_labels, test_data, test_labels = utils.load_mnist_data('data/4', classes, val_split=0.1)
print(len(train_data), len(test_data))
#TODO: make hyperparameters also arguments to train()
params = None
n = 16#len(train_data[0])
mod = model.Model(n=n, num_trials=1, classes=classes)
if args.model == 'cascade':
    mod = model2.Model(n=n, num_trials=1, classes=classes)
# print(np.linalg.norm(train_data[0]), train_data[0], "hello")
init_params= None
init_params = np.random.normal(loc=0.0, scale=1.0, size=mod.count)

hyperparams = dict({'lam':.234, 'eta':5.59, 'batch_size':25, 'num_iterations':20,
                    'a':28.0, 'b':33.0, 'A':74.1, 'gamma':0.882, 't':0.658, 's':4.13})

if args.sigopt:
    hyperparams, mod = sigopt_training.train(train_data, train_labels, mod, args.method, hyperparams)

if args.resumefile:
    init_params = utils.load_params(args.resumefile)

test.get_accuracy(test_data[:200], test_labels[:200], mod, init_params)
if args.method == 'spsa':
    params, mod = spsa.train(train_data, train_labels, mod, hyperparams, params=init_params)
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
