#!/usr/bin/python
# Script to test the spread of the variables are
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

parameters = []
num_lines = 0
with open("init_params","r") as f:
    next(f)
    for line in f:
        num_lines += 1
        line_numbs = [float(x) for x in line.split()]
        parameters.append(line_numbs)
  
        
maximums = [0,0,0]
minimums = [5,5,5]
xs = np.zeros(num_lines)
ys = np.zeros(num_lines)
zs = np.zeros(num_lines)
for i in range(len(parameters)):
    xs[i] = parameters[i][0]
    ys[i] = parameters[i][1]
    zs[i] = parameters[i][2]    
    for j in range(3):
        if parameters[i][j] > maximums[j]:
            maximums[j] = parameters[i][j]
        if parameters[i][j] < minimums[j]:
            minimums[j] = parameters[i][j]

x_bins = np.linspace(minimums[0],maximums[0],11)
y_bins = np.linspace(minimums[1],maximums[1],11)
z_bins = np.linspace(minimums[2],maximums[2],11)

#for i in range(len(xs)):
#    if xs[i] 

#3D Plot of points
fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.set_title("Scatter of the C-C, C-O, Si-O ReaxFF parameters")
ax.set_xlabel("C-C")
ax.set_ylabel("Si-O")
ax.set_zlabel("C-O") 
ax.set_xlim3d(1.55, 2.45)
ax.set_ylim3d(1.55, 2.45)
ax.set_zlim3d(1.55, 2.45)
ax.scatter(xs,ys,zs)
plt.show()