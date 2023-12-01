'''
This code is a part of unachieved work concerning the energy of networks.
'''

import numpy as np
import functions as fun
import matplotlib.pyplot as plt

# ---------------#
#   FUNCTIONS    #
# ---------------#

def sync_hebian(perturbed_pattern, hebian_weights):
    energy_history = []
    history = fun.dynamics(perturbed_pattern, hebian_weights)
    for value in history:
        energy_history.append(fun.energy(value, hebian_weights))
    plt.plot(range(len(history)), energy_history)
    print (energy_history)
    
def async_hebian(perturbed_pattern, hebian_weights):
    energy_history = []
    history = fun.dynamics_async(perturbed_pattern, hebian_weights, 30000, 10000)
    for value in history:
        energy_history.append(fun.energy(value, hebian_weights))
    plt.plot(range(len(history)), energy_history)
    print(energy_history)

# ---------------#
#   ALGORITHM    #
# ---------------#

N = 2500  # pattern size
M = 50  # amount of memorized patterns
num_perturb = 1000  # number of perturbations
index_perturb = 2  # state to perturb

memorized_patterns = fun.generate_patterns(M, N)
hebian_weights = fun.hebbian_weights(memorized_patterns)
perturbed_pattern = fun.perturb_pattern(memorized_patterns[index_perturb], num_perturb)

sync_hebian(perturbed_pattern, hebian_weights)
async_hebian(perturbed_pattern, hebian_weights)
