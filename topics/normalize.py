import numpy as np

X = np.array([[8, -2, 3], [2, 25, 0], [4, 0, -2]])
print(X / np.max(X, axis=0))