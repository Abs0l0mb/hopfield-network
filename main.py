import numpy as np
import functions as fun
import matplotlib.pyplot as plt


N = 2500  # pattern size
M = 50  # amount of memorized patterns
num_perturb = 1000  # number of perturbations
index_perturb = 2  # state to perturb

# ---------------#
#   ALGORITHM    #
# ---------------#


# ---------------------#
#     1.1 : energy     #
# ---------------------#

#memorized_patterns = fun.generate_patterns(M, N)
#hebian_weights = fun.hebbian_weights(memorized_patterns)
#storkey_weights = fun.storkey_weights(memorized_patterns)
#perturbed_pattern = fun.perturb_pattern(memorized_patterns[index_perturb], num_perturb)

patterns = np.array([[1,1,-1,-1],[ 1,1,-1,1],[-1,1,-1,1]])

#print(fun.storkey_weights(pattern), "\n")
print(fun.storkey_weights_test(patterns), "\n")


def sync_hebian():
    energy_history = []
    history = fun.dynamics(perturbed_pattern, hebian_weights)
    for value in history:
        energy_history.append(fun.energy(value, hebian_weights))
    #plt.plot(range(len(history)), energy_history)
    print (energy_history)

def sync_storkey():
    energy_history = []
    history = fun.dynamics(perturbed_pattern, storkey_weights)
    for value in history:
        energy_history.append(fun.energy(value, storkey_weights))
    plt.plot(range(len(history)), energy_history)
    
def async_hebain():
    energy_history = []
    history = fun.dynamics_async(perturbed_pattern, hebian_weights, 30000, 10000)
    for value in history:
        energy_history.append(fun.energy(value, hebian_weights))
    #plt.plot(range(len(history)), energy_history)
    print(energy_history)
    
def async_storkey():
    energy_history = []
    history = fun.dynamics_async(perturbed_pattern, storkey_weights, 30000, 10000)
    for value in history:
        energy_history.append(fun.energy(value, storkey_weights))
    plt.plot(range(len(history)), energy_history)


#sync_hebian()
#sync_storkey()
#async_hebain()
#async_storkey()


# ---------------------#
#   1.2 : checkboard   #
# ---------------------#

'''

checkerBoard = np.array([[1 if ((i // 5) + (j // 5)) % 2 == 0 else -1 for j in range(50)] for i in range(50)])
flattenedCheckerBoard = checkerBoard.flatten().tolist()
patternList = fun.generate_patterns(M, 2500).tolist()
patternList.append(flattenedCheckerBoard)
flattenedCheckerBoardDisturbed = fun.perturb_pattern(flattenedCheckerBoard, 1000)


# sync

hebbianSyncHistory = fun.dynamics(np.array(flattenedCheckerBoardDisturbed), fun.hebbian_weights(patternList))
reshapedHistory = [np.reshape(state, (50, 50)).tolist() for state in hebbianSyncHistory]
fun.save_video(reshapedHistory, './hebbianSync.gif')


# async

hebbianAsyncHistory = fun.dynamics_async(np.array(flattenedCheckerBoardDisturbed), fun.hebbian_weights(patternList), 30000, 10000)
reshapedAsyncHistory = [np.reshape(state, (50, 50)).tolist() for state in hebbianAsyncHistory]
fun.save_video(reshapedAsyncHistory, './hebbianAsync.gif')

'''