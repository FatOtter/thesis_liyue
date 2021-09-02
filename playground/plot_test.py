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
trajectory = pd.read_csv("./records/trajectory_test2021_08_29_01.csv")
points = []
for i in range(len(trajectory)):
    points.append((trajectory.loc[i][1], trajectory.loc[i][2]))
points.append((0,0))


loss = pd.read_csv("./records/Landscape2021_08_29_01.csv")
coords = pd.read_csv("./records/pca_coords2021_08_29_01.csv")
x_line = coords['0'].to_numpy()
y_line = coords['1'].to_numpy()
X = x_line.reshape([10,10])
Y = y_line.reshape([10,10])
z = loss['loss'].to_numpy().reshape([10, 10])
z = np.clip(z, 0, 0.005)
plot.contour(X, Y, z, 10)
for i in range(len(points)-1):
    x, y = points[i]
    dx = points[i+1][0] - points[i][0]
    dy = points[i+1][1] - points[i][1]
    plot.arrow(x, y, dx, dy,length_includes_head=True, color='red', head_width=0.1)

plot.colorbar()
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
