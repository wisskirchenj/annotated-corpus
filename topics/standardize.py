import numpy as np
from math import sqrt
from sklearn.preprocessing import StandardScaler

data = np.array([-2, 5, 0, 8, -32, 1])
variance = np.var(data)
mean = np.mean(data)
data = (data - mean) / sqrt(variance)
print(data)
print(f'{data.min():.2f} {data.max():.2f}')

scaler = StandardScaler()
data = np.array([-2, 5, 0, 8, -32, 1]).reshape(-1, 1)
scaler.fit(data)
data = scaler.transform(data)
print(data)

data = np.array([-20, 10, 2, -6, -30, 15])
min = data.min()
print((data - min)/(data.max() - min))
