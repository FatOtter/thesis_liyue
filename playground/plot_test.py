from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import pandas as pd

RECORDING_PATH = "../trainig_records/"
# points = [(-4.785964818324741,-0.7167800192913755),
#           (-3.3783938148933323,0.3663016315213275),
#           (-2.03922728748871,0.7670726133681581),
#           (-0.9746591960609418,0.6450843395979435),
#           (0, 0)]
trajectory = pd.read_csv("./records/Trajectory2021_09_17_19.csv")
# points.append((0,0))


loss = pd.read_csv("./records/Landscape2021_09_17_19.csv")
coords = pd.read_csv("./records/trajectory_coords2021_09_04_14.csv")

x_line = loss['x'].to_numpy()
y_line = loss['y'].to_numpy()
X = x_line.reshape([30,30])
Y = y_line.reshape([30,30])
z = loss['loss'].to_numpy().reshape([30, 30])
# z = np.clip(z, 0, 100)
plot.contour(X, Y, z, [2**x/100 for x in range(15)], linewidths=0.5)
#
points = []
for i in range(len(trajectory)):
    if i % 3 == 0:
        points.append((trajectory.loc[i]["x"], trajectory.loc[i]["y"]))
for i in range(len(points)-2):
    x, y = points[i]
    dx = points[i+1][0] - points[i][0]
    dy = points[i+1][1] - points[i][1]
    plot.arrow(x, y, dx, dy,length_includes_head=True, color='red', head_width=0.05)
x, y = points[-3]
dx = points[-3][0] - points[-6][0]
dy = points[-3][1] - points[-6][1]
plot.arrow(x, y, dx, dy,length_includes_head=False, color='red', head_width=2)

points = []
for i in range(len(trajectory)):
    if i % 3 == 1:
        points.append((trajectory.loc[i]["x"], trajectory.loc[i]["y"]))
for i in range(len(points)-2):
    x, y = points[i]
    dx = points[i+1][0] - points[i][0]
    dy = points[i+1][1] - points[i][1]
    plot.arrow(x, y, dx, dy,length_includes_head=True, color='green', head_width=0.05)
x, y = points[-2]
dx = points[-2][0] - points[-5][0]
dy = points[-2][1] - points[-5][1]
plot.arrow(x, y, dx, dy,length_includes_head=False, color='green', head_width=2)

points = []
for i in range(len(trajectory)):
    if i % 3 == 2:
        points.append((trajectory.loc[i]["x"], trajectory.loc[i]["y"]))
for i in range(len(points)-2):
    x, y = points[i]
    dx = points[i+1][0] - points[i][0]
    dy = points[i+1][1] - points[i][1]
    plot.arrow(x, y, dx, dy,length_includes_head=True, color='blue', head_width=0.05)
x, y = points[-1]
dx = points[-1][0] - points[-4][0]
dy = points[-1][1] - points[-4][1]
plot.arrow(x, y, dx, dy,length_includes_head=False, color='blue', head_width=2)
# ax1 = plot.gca()
# x1 = x_line[0::4]
# y1 = y_line[0::4]
# x2 = x_line[1::4]
# y2 = y_line[1::4]
# x3 = x_line[3::4]
# y3 = y_line[3::4]

# scatter1 = ax1.scatter(x1, y1, c="r", s=0.5)
# scatter2 = ax1.scatter(x2, y2, c="b", s=0.5)
# scatter3 = ax1.scatter(x3, y3, c="y", s=0.5)

#
color_bar = plot.colorbar(format="%.3f")
plot.show()

# fig = plot.figure()
# ax = plot.axes(projection="3d")
# cmap = plot.get_cmap('hot')
# surf = ax.plot_trisurf(x_line, y_line, z, cmap=cmap)
# ax.zaxis.set_major_locator(LinearLocator(10))
# fig.colorbar(surf, ax=ax)
# ax.set_title('loss landscape')
# ax.set_zlabel('loss value')
# plot.show()
