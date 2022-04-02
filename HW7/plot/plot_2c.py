import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

initial_data = "../plot/2_initial.txt"
optimized_data = "../plot/2_c_optimized.txt"

initial_x = []
initial_y = []
initial_z = []
with open(initial_data) as f:
    lines = f.readlines()
    for line in lines:
        x, y, z = line.split()
        initial_x.append(float(x))
        initial_y.append(float(y))
        initial_z.append(float(z))

# plt.plot(initial_x, initial_y, initial_z)
# plt.show()

optimized_x = []
optimized_y = []
optimized_z = []
with open(optimized_data) as f:
    lines = f.readlines()
    for line in lines:
        x, y, z = line.split()
        optimized_x.append(float(x))
        optimized_y.append(float(y))
        optimized_z.append(float(z))

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(initial_x, initial_y, initial_z, 'green')
ax.plot3D(optimized_x, optimized_y, optimized_z, 'blue')

# # Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');
# ax.show()

# plt.plot(optimized_x, optimized_y, optimized_z)
# plt.legend(['Initial trajectory', 'optimized trajectory'])
# plt.xlabel('x')
# plt.ylabel('y')
plt.show()
