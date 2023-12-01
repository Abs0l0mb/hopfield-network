'''
This code is the main file of the project. It runs what is described in the readme of the project.
'''

import numpy as np
import functions as fun
import matplotlib.pyplot as plt

# pattern size, must be 2500 to match the size of the 50*50 checkerboard
N = 2500  

# amount of memorized patterns
M = 50  

# amount of perturbations to the pattern
num_perturb = 1000  

# ---------------#
#   ALGORITHM    #
# ---------------#

#generate the checkerboard pattern in 2D, then flatten it in 1D to be used as input for the hopfield network
checkerBoard = np.array([[1 if ((i // 5) + (j // 5)) % 2 == 0 else -1 for j in range(50)] for i in range(50)])
flattenedCheckerBoard = checkerBoard.flatten().tolist()

#generate the pattern list and add the checkerboard pattern to it
patternList = fun.generate_patterns(M, N).tolist()
patternList.append(flattenedCheckerBoard)

#disturb the checkerboard pattern
flattenedCheckerBoardDisturbed = fun.perturb_pattern(flattenedCheckerBoard, 1000)

# sync dynamics
hebbianSyncHistory = fun.dynamics(np.array(flattenedCheckerBoardDisturbed), fun.hebbian_weights(patternList))
reshapedHistory = [np.reshape(state, (50, 50)).tolist() for state in hebbianSyncHistory]
fun.save_video(reshapedHistory, './hebbianSync.gif')

# async dynamics
hebbianAsyncHistory = fun.dynamics_async(np.array(flattenedCheckerBoardDisturbed), fun.hebbian_weights(patternList), 30000, 10000)
reshapedAsyncHistory = [np.reshape(state, (50, 50)).tolist() for state in hebbianAsyncHistory]
fun.save_video(reshapedAsyncHistory, './hebbianAsync.gif')

