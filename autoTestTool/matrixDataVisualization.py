import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# 生成一个随机稀疏矩阵
size = 100
density = 0.05  # 5% 的元素是非零的
A = sp.rand(size, size, density=density, format='csr')

# 可视化
plt.figure(figsize=(6,6))
plt.spy(A, markersize=2)  # markersize 控制点的大小
plt.title("Sparse Matrix Visualization")
plt.show()