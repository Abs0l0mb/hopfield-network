import numpy as np
import time

# Generate two random arrays
a = np.random.rand(200, 200)
b = np.random.rand(200, 200)

# NumPy dot product
start_time = time.time()
result_numpy = np.dot(a, b)
numpy_time = time.time() - start_time

# Manual dot product with for loops
def manual_dot(a, b):
    result = np.zeros((a.shape[0], b.shape[1]))
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):
                result[i, j] += a[i, k] * b[k, j]
    return result

start_time = time.time()
result_manual = manual_dot(a, b)
manual_time = time.time() - start_time

# Check if the results are the same
print(np.allclose(result_numpy, result_manual))

# Compare execution times
print(f"NumPy dot product time: {numpy_time} seconds")
print(f"Manual dot product time: {manual_time} seconds")