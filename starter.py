from pyquil.quil import Program
from pyquil.api import QVMConnection
from pyquil.gates import CNOT, H, RZ, RY, RX, CZ
from pyquil.parameters import Parameter
from pyquil.parametric import ParametricProgram
from collections import deque
# from pyswarms.single.global_best import GlobalBestPSO
from pyswarm import pso
import numpy as np
import math
import scipy


def prep_state_program(x):
    #TODO: Prep a state given classical vector x
    prog = Program()
    for i in range(0, len(x)):
        angle = math.pi*x[i]/2
        prog.inst(RY(angle, i))
    return prog

def single_qubit_unitary(angles):
    return RX(angles[0]), RZ(angles[1]), RX(angles[2])


def prep_parametric_gates(n, params):
    #n is the number of qubits. Here we create a circuit using only using 1 and 2 qubit unitaries
    #Every 1 qubit unitary can be encoded as 3 parametric rotations - rx,rz,rx
    gates = []
    l = math.floor(math.log(n, 2))
    for i in range(0, l):
        layer_gates = []
        for j in range(0, math.floor(math.pow(2, l-i-1))):
            single_qubit_gates = []
            q1 = 2**i - 1 + j*(2**(i + 1))
            q2 = q1  + 2**i
            for k in range(0, 3):
                single_qubit_gates += [[single_qubit_unitary(params[i][j][k][0]), single_qubit_unitary(params[i][j][k][1])]]
            layer_gates += [single_qubit_gates]
        gates += [layer_gates]
    gates += [single_qubit_unitary(params[math.floor(math.log(n, 2))])]
    return gates

def init_params(n):
    params = []
    l = math.floor(math.log(n, 2))
    for i in range(0, l):
        i_level = []
        for j in range(0, math.floor(math.pow(2, l-i-1))):
            j_level = []
            for k in range(0, 3):
                qubit_one_params = 2*math.pi*np.random.rand(3)
                qubit_two_params = 2*math.pi*np.random.rand(3)
                k_level = [list(qubit_one_params), list(qubit_two_params)]
                j_level += [k_level]
            i_level += [j_level]
        params += [i_level]
    params += [list(2*math.pi*np.random.rand(3))]
    return params



def prep_circuit(n, params):
    prog = Program()
    #Prepare parametric gates
    single_qubit_unitaries = prep_parametric_gates(n, params)
    l = math.floor(math.log(n, 2))
    for i in range(0, l):
        #number of gates in the ith layer is 2^(log(n)-i-1)
        for j in range(0, math.floor(math.pow(2, l-i-1))):
            q1 = 2**i - 1 + j*(2**(i + 1))
            q2 = q1  + 2**i
            # print(q1, q2)
            for k in range(0, 3):
                prog.inst([g(q1) for g in single_qubit_unitaries[i][j][k][0]])
                prog.inst([g(q2) for g in single_qubit_unitaries[i][j][k][1]])
                prog.inst(CZ(q1, q2))
    prog.inst([g(n - 1) for g in single_qubit_unitaries[math.floor(math.log(n, 2))]])
    prog.measure(n - 1, 0)
            #create each block
    return prog

def vectorize(params, n):
    #TODO: Validate the input
    params_vec = []
    l = math.floor(math.log(n, 2))
    for i in range(0, l):
        i_level = []
        for j in range(0, math.floor(math.pow(2, l-i-1))):
            j_level = []
            for k in range(0, 3):
                params_vec += params[i][j][k][0]
                params_vec += params[i][j][k][1]
    params_vec += params[math.floor(math.log(n, 2))]
    return params_vec

def tensorize(params_vec, n):
    #TODO: Validate the input
    params_vec = deque(params_vec)
    params = []
    l = math.floor(math.log(n, 2))
    for i in range(0, l):
        i_level = []
        for j in range(0, math.floor(math.pow(2, l-i-1))):
            j_level = []
            for k in range(0, 3):
                qubit_one_params = []
                for l in range(0, 3):
                    qubit_one_params.append(params_vec.popleft())
                qubit_two_params = []
                for l in range(0, 3):
                    qubit_two_params.append(params_vec.popleft())
                k_level = [qubit_one_params, qubit_two_params]
                j_level += [k_level]
            i_level += [j_level]
        params += [i_level]
    last_unitary = []
    for l in range(0, 3):
        last_unitary += [params_vec.popleft()]
    params.append(last_unitary)
    return params

def get_distribution(params, n, sample):
    num_trials = 10
    p = Program().inst(prep_state_program(sample) + prep_circuit(n, params))
    qvm = QVMConnection()
    res = qvm.run(p, [0], trials = num_trials)
    count_0 = 0.0
    for x in res:
        if x[0] == 0:
            count_0 += 1.0
    return (count_0/num_trials, 1-count_0/num_trials)

def get_loss(params_vec, *args):
    loss = 0.0
    n, samples, labels, lam, eta = args
    print("Vec", params_vec)
    params = tensorize(params_vec, n)
    for i in range(len(samples)):
        sample = samples[i]
        label = math.floor(labels[i])
        dist = get_distribution(params, n, sample)
        loss += (max(dist[1 - label] - dist[label] + lam, 0.0)) ** eta
    print(loss)
    return loss

def get_multiloss(params_vecs, *args):
    return [get_loss(x, args) for x in params_vecs]


def train():
    data = np.loadtxt(open("data/data.csv", "rb"), delimiter = ",")
    labels = np .loadtxt(open("data/labels.csv", "rb"), delimiter = ",")
    #prep_state_program([7.476498897658023779e-01,2.523501102341976221e-01,0.000000000000000000e+00,0.000000000000000000e+00])
    #Number of qubits
    n = len(data[0])

    #Prepare circuit
    #TODO: Can this be optimized into 1 parametric program?
    # par_prep_state = ParametricProgram(prep_state_program)
    # par_pred_circuit = ParametricProgram(prep_circuit)

    #Shuffle data
    combined = np.hstack((data, labels[np.newaxis].T))
    np.random.shuffle(combined)
    data = combined[:,list(range(n))]
    labels = combined[:,n]
    num_training = math.floor(0.7*len(data))
    train_data = data[:num_training]
    train_labels = labels[:num_training]
    test_data = data[num_training:]
    test_labels = labels[num_training:]
    #Save parameters
    params = init_params(n)
    num_epochs = 15
    batch_size = 5
    lam = 0
    eta = 1
    # Can optimize in batches
    # for i in range(num_epochs):
    #     sample_indices = np.random.randint(train_data.shape[0], size=batch_size)
    #     samples = train_data[sample_indices, :]
    #     sample_labels = train_labels[sample_indices, :]
    vec_params = vectorize(params, n)
    # print(get_loss(vectorize(params, n), n, train_data, train_labels, lam, eta))

    dim = len(vec_params)
    print(dim)
    bounds = (np.zeros(dim), 2*math.pi*np.ones(dim))
    # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    # optimizer = GlobalBestPSO(n_particles=2, dimensions=dim, options=options, bounds=bounds)
    # cost, pos = optimizer.optimize(get_multiloss, 100, print_step=10, verbose=3,\
    # n=n, samples=train_data, labels=train_labels, lam=lam, eta=eta)
    # print(cost, pos)
    xopt, fopt = pso(get_loss, bounds[0], bounds[1], args=(n, train_data, train_labels, lam, eta), swarmsize=10, maxiter=10)
    print(xopt, fopt)
    params_vec = xopt

    params = tensorize(params_vec, n)
    num_correct = 0
    for i in range(len(data)):
        dist =  get_distribution(params, n, data[i])
        if dist[math.floor(labels[i])] > 0.5:
            num_correct += 1
    print("test accuracy = ", num_correct/len(data))
if __name__ == '__main__':
    train()
