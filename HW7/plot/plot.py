import matplotlib.pyplot as plt

initial_data = "/home/leekt/UMich/ROB530/HW7/plot/initial.txt"
optimized_data = "/home/leekt/UMich/ROB530/HW7/plot/1_c_optimized.txt"

initial_x = []
initial_y = []
initial_theta = []
with open(initial_data) as f:
    lines = f.readlines()
    for line in lines:
        x, y, theta = line.split()
        initial_x.append(float(x))
        initial_y.append(float(y))
        initial_theta.append(float(theta))

plt.plot(initial_x, initial_y)
# plt.show()

optimized_x = []
optimized_y = []
optimized_theta = []
with open(optimized_data) as f:
    lines = f.readlines()
    for line in lines:
        x, y, theta = line.split()
        optimized_x.append(float(x))
        optimized_y.append(float(y))
        optimized_theta.append(float(theta))

plt.plot(optimized_x, optimized_y)
plt.show()
