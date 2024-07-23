# 160,165,170,155,162,168,158,163,169,172
import numpy as np

data = np.array([160, 165, 170, 155, 162, 168, 158, 163, 169, 172])
variance = np.var(data)
print(f'{variance:.2f}')
# sample variance
variance = np.var(data, ddof=1)
print(f'{variance:.2f}')
# mean
print(f'{np.mean(data):.2f}')
sample_var = data - np.mean(data)
sample_var = sample_var@sample_var
sample_var /= len(data) - 1
print(f'{sample_var}')

predictions = [7.693, 8.01, 7.743, 7.87, 7.7658, 7.832, 8.022]
print(f'{np.mean(predictions):.5f}')
