from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import pandas as pd

RECORDING_PATH = "../trainig_records/"
points = [(49, 49), (40, 40), (34, 33), (29, 29), (25, 25)]

df = pd.read_csv(RECORDING_PATH+"landscape2021_08_03_18.csv")
x_line = df['alpha'].to_numpy()
y_line = df['beta'].to_numpy()
X, Y = np.meshgrid(x_line, y_line)
z = df['loss'].to_numpy()#.reshape([10,10])
z = z.clip(0, 0.003)
# plot.contour(z,levels=[x**2/100000 for x in range(30)])
# for i in range(len(points)-1):
#     x, y = points[i]
#     dx = points[i+1][0] - points[i][0]
#     dy = points[i+1][1] - points[i][1]
#     plot.arrow(x, y, dx, dy,length_includes_head=True, color='red', head_width=0.5)

# plot.colorbar()
# plot.show()

fig = plot.figure()
ax = plot.axes(projection="3d")
cmap = plot.get_cmap('hot')
surf = ax.plot_trisurf(x_line, y_line, z, cmap=cmap)
ax.zaxis.set_major_locator(LinearLocator(10))
fig.colorbar(surf, ax=ax)
ax.set_title('loss landscape')
ax.set_zlabel('loss value')
plot.show()
