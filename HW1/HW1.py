import numpy as np

A = np.array([[8, -5], [-5, 10]])
B = np.array([[3, 7], [7, -5]])
C = np.array([[8, 5], [3, 3]])
D = np.array([[3, 7], [7, 4]])
E = np.array([[5, 4], [4, 8]])
F = np.array([[6, 5], [5, 4]])
G = np.array([[25, -10], [-10, 40]])
H = np.array([[3, -6], [-4, 6]])
I = np.array([[9, -3], [-3, 5]])


total = [A, B, C, D, E, F, G, H, I]
for t in total:
    print(t)
    print(np.linalg.eig(t))

# w, v = np.linalg.eig(E)
# print(w, v)

# d = 0.99
# print((0.009 / (0.009 + 0.198 * d)))
# print((0.01 - 0.001 * d) / (0.01 - 0.001 * d + 0.198))