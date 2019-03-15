import pyswarm_training
import spsa_training
import evolve_training
import sigopt_training

import numpy as np
import math
import utils
from sigopt import Connection


conn = Connection(client_token="EUXXAFUFQRXKISDCAACDHOGJSOXOBZQRUCDSQTYRBXINEDJF")

def train(train_data, train_labels, mod, method, hyperparams):
    """
    """
    #Add model hyperparameters to be optimized
    if method =='spsa':
        hyperparams.update({'a':0, 'c':0})
    elif method =='pyswarm':
        hyperparams.update({'swarm_size':0})
    elif method == 'evolve':
        hyperparams.update({'mutate':0, 'recombine':0})

    keys = hyperparams.keys()
    #Define model hyperparameter ranges
    lam_range = [0,1]
    eta_range = [0,5]
    batch_size_range = [10,100]

    ###Define method hyperparamater ranges
    #SPSA
    a_range = [0,1]
    c_range = [0,1]

    #Evolutionary:
    mutate_range = [0,2]
    recombine_range = [0,1]

    #Particle Swarm
    swarm_size_range = [50,100]

    #Define overall parameters for sigopt
    parameters = []
    for i in keys:
        if i == 'batch_size' or i == 'swarm_size':
            parameters.append(dict(name=i, type='int', bounds=dict(min=eval(i+'_range')[0], max=eval(i+'_range')[1])))
        else:
            parameters.append(dict(name=i, type='double', bounds=dict(min=eval(i+'_range')[0], max=eval(i+'_range')[1])))

    #Create sigopt experiment
    print(parameters)
    experiment = conn.experiments().create(
    name='QML Tensor Networks',
    parameters=parameters,)
    print("Created experiment: https://app.sigopt.com/experiment/" + experiment.id)

    # Evaluate your model with the suggested parameter assignments

    def evaluate_model(assignments):
        return eval(method+'_training').train(train_data, train_labels, mod, assignments)

    # Run the Optimization Loop between 10x - 20x the number of parameters
    for _ in range(30):
        suggestion = conn.experiments(experiment.id).suggestions().create()
        value = evaluate_model(suggestion.assignments)
        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=value,
        )

    hyperparams = suggestion.assignments
    np.savetxt("hyper_params/" + args.savefile, hyperparams, delimiter = ",")
    return hyperparams, mod
