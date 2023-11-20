import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

# ---------------#
#   FUNCTIONS    #
# ---------------#

# useful for debugging, checks the amount of differences between two patterns and displays that number
def test_differences(pattern1, pattern2):
    if len(pattern1) != len(pattern2):
        print("not the same size")
        return
    diff = 0
    for i in range(0, len(pattern1)):
        if pattern1[i] != pattern2[i]:
            diff += 1
    print("number of differences : ", diff, "\n")


def generate_patterns(num_patterns, pattern_size):
    """
    Generate a specified number of random patterns.

    Parameters:
    num_patterns (int): Number of patterns to generate.
    pattern_size (int): Size of each pattern.

    Returns:
    array: Randomly generated patterns as an array.
    """
    return np.random.choice([-1, 1], size=(num_patterns, pattern_size))


def perturb_pattern(pattern, nb_changes):
    """
    Perturb a pattern by changing a specified number of elements.

    Parameters:
    pattern (array): Pattern to be perturbed.
    nb_changes (int): Number of changes to apply.

    Returns:
    array: Perturbed pattern.
    """
    random_indexes = np.random.choice(len(pattern), nb_changes, replace=False)
    disturbed_pattern = np.copy(pattern)
    disturbed_pattern[random_indexes] *= (-1)
    return disturbed_pattern



def pattern_match(memorized_patterns, pattern):
    """
    Check if a pattern matches any of the memorized patterns.
    !! modify the index thing !!

    Parameters:
    memorized_patterns (list): List of memorized patterns for comparison.
    pattern (array): Pattern to match against the memorized patterns.

    Returns:
    int or None: Index of matching pattern if found, otherwise None.
    """
    for memorized_pattern in memorized_patterns: 
        if memorized_pattern == pattern:
            return memorized_pattern.index()
    return None


def hebbian_weights(patterns):
    """
    Compute the weight matrix using Hebbian learning rule.

    Parameters:
    patterns (array): List of patterns to calculate weights.

    Returns:
    array: Weight matrix computed using Hebbian learning rule.
    """
    weight_matrix = np.zeros((len(patterns[0]), len(patterns[0])))
    for pattern in patterns:
        weight_matrix += np.outer(pattern, pattern)
    weight_matrix /= len(patterns)
    np.fill_diagonal(weight_matrix, 0)
    return weight_matrix

def storkey_weights(patterns):
    """
    Compute the weight matrix using Storkey learning rule.

    Parameters:
    patterns (array): List of patterns to calculate weights.

    Returns:
    array: Weight matrix computed using Storkey learning rule.
    """
    M = len(patterns)
    N = len(patterns[0])
    weight_matrix = np.zeros((N, N))
    for pattern in patterns:
        #compute the effect of self connexions to later remove them
        self_connexions = weight_matrix * pattern.reshape((1, -1))
        np.fill_diagonal(self_connexions, 0)
        #calculate  the PH_product in advance since the pi*hij product is just the transposed pi*hji product.
        PH_product = pattern * np.dot(weight_matrix, pattern.reshape((-1, 1))) - (np.diag(weight_matrix)*pattern).reshape((-1, 1)) - self_connexions
        weight_matrix += ((np.outer(pattern, pattern)) - PH_product - PH_product.T)/len(pattern)
    np.fill_diagonal(weight_matrix, 0)
    return weight_matrix


def update(state, weights):
    """
    Apply the update rule to a given state using weights.

    Parameters:
    state (array): Current state to be updated.
    weights (array): Weight matrix used for the update rule.

    Returns:
    array: Updated state based on the update rule.
    """
    state = np.dot(weights, state)
    for i in range(0, len(state)):
        if state[i] < 0:
            state[i] = -1
        else:
            state[i] = 1
    return state.astype(int)


def update_async(state, weights):
    """
    Apply the update rule to a state asynchronously using weights.

    Parameters:
    state (array): Current state to be updated.
    weights (array): Weight matrix used for the update rule.

    Returns:
    array: Updated state based on the asynchronous update rule.
    """
    new_state = np.copy(state)
    i = np.random.randint(len(state))
    new_state[i] = np.dot(weights[i], new_state)
    if new_state[i] < 0:
        new_state[i] = -1
    else:
        new_state[i] = 1
    return new_state


def dynamics(state, weights, max_ite=20):
    """
    Run a dynamical system by updating the state using a specified weight matrix.
    Commented the energy part for faster testing
    
    Parameters:
    state (array): Initial state of the system.
    weights (array): Weight matrix used for state update.
    max_ite (int): Maximum number of iterations. Default is 20.

    Returns:
    tuple: A tuple containing the history of states and energy values during iterations.
    """
    state_history = [np.copy(state)]
    for t in range(max_ite):
        print(t)
        state = update(state, weights)
        if np.array_equal(state, state_history[-1]) and t != 0:
            break
        state_history.append(np.copy(state))
    return state_history


def dynamics_async(state, weights, max_ite, convergence_num_iter):  # check si la fin marche
    """
    Run a dynamical system asynchronously until convergence or a maximum number of iterations.
    Commented the energy part for faster testing

    Parameters:
    state (array): Initial state of the system.
    weights (array): Weight matrix used for state update.
    max_ite (int): Maximum number of iterations.
    convergence_num_iter (int): Consecutive iterations to check for convergence.

    Returns:
    tuple: A tuple containing the history of states and energy values during iterations.
    """
    state_history = [np.copy(state)]
    state_video_history = [np.copy(state)]
    nb_recurences = 0
    for t in range(max_ite):
        state = update_async(state, weights)
        state_history.append(state.copy())

        print(t)
        if(t%999 == 0):
            state_video_history.append(state.copy())

        if t >= convergence_num_iter:
            # check after every step of convergence_num_iter ? faster but less acurate ?
            #recent_history = state_history[-convergence_num_iter:]
            if np.array_equal(state_history[-1], state_history[-2]):
                nb_recurences += 1
            else:
                nb_recurences = 0
        if nb_recurences >= convergence_num_iter:
                break
    return state_video_history


def energy(state, weights): #A OPTIMISER !!!
    """
    Calculate the energy of a state in the system based on given weights.

    Parameters:
    state (array): State of the system.
    weights (array): Weight matrix used for energy calculation.

    Returns:
    float: Energy of the state within the system.
    """
    sum = 0
    for i in range(len(weights)):
        for j in range(len(weights)):
            sum += weights[i, j] * state[i] * state[j]
    return -sum/2
            
def save_video(state_list, out_path):
    """
    Generate and save an animation of the states as a video.

    Parameters:
    state_list (list): List of states representing frames in the video.
    out_path (str): Output file path for the generated video.

    Returns:
    None
    """    
    frames = []
    for state in state_list:
        frames.append([plt.imshow(state, cmap='gray')])

    animation = ArtistAnimation(plt.gcf(), frames, interval=500, blit=True)
    animation.save(out_path, writer='pillow', fps=2) 
    
def storkey_weights_test(patterns):
    M, N = len(patterns), len(patterns[0])

    h = np.zeros((N, N))
    new_matrix = np.zeros((N, N))
    old_matrix = np.zeros((N, N))

    for mu in range(M):
        old_matrix = np.copy(new_matrix)
        new_matrix = np.zeros((N, N))
        
        h += np.dot(old_matrix, patterns[:,mu]) - np.outer(patterns[:,mu], patterns[:,mu])
        
        for i in range(N):
            for j in range(N):
                new_matrix[i, j] = old_matrix[i, j] + (1 / N) * (
                        patterns[mu, i] * patterns[mu, j] -
                        patterns[mu, i] * h[i, j] -
                        h[i, j] * patterns[mu, j]
                )

    return new_matrix

def storkey_weights_test_VALID(patterns):

    M = len(patterns)
    N = len(patterns[0])

    # Create empty weight matrix
    h = np.zeros((N, N))
    new_matrix = np.zeros((N, N))
    old_matrix = np.zeros((N, N))

    for mu in range(M):

        old_matrix = np.copy(new_matrix)
        new_matrix = np.zeros((N, N))
        h = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if(k != i and k != j):
                        h[i][j] += old_matrix[i][k]*patterns[mu][k]

        for i in range(N):
            for j in range(N):                
                new_matrix[i][j] = old_matrix[i][j] + (1/N) * ((patterns[mu][i] * patterns[mu][j]) - (patterns[mu][i] * h[j][i]) - (patterns[mu][j] * h[i][j]))
    
    return new_matrix

def storkey_weights_test_2(patterns):
    """
    Compute the weight matrix using Storkey learning rule.

    Parameters:
    patterns (array): List of patterns to calculate weights.

    Returns:
    array: Weight matrix computed using Storkey learning rule.
    """
    M = len(patterns)
    N = len(patterns[0])
    weight_matrix = np.zeros((N, N))

    for pattern in patterns:
        # Compute the effect of self-connections to later remove themw
        self_connections = np.dot(weight_matrix, pattern) - np.diag(weight_matrix) * pattern

        # Calculate the h term
        h_term = np.outer(self_connections, pattern)

        # Update the weight matrix with a corrected sign
        weight_matrix += np.outer(pattern, pattern) - h_term - h_term.T

    # Add identity matrix to the final weights
    weight_matrix += np.eye(N)

    # Normalize by the number of patterns
    weight_matrix /= N

    # Ensure the matrix is symmetric
    weight_matrix = 0.5 * (weight_matrix + weight_matrix.T)

    return weight_matrix

