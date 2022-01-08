import matplotlib.pyplot as plot
import pandas as pd
import numpy as np

MNIST_CONTOUR_ARRAY = [2**x/100 for x in range(15)]
COLORS = ["red", "green", "blue", "yellow", "orange", "black", "purple"]


class ThesisPlotter:
    def __init__(self, loss_landscape: str, trajectory: str = None):
        self.fig = plot.figure()
        self.loss_landscape = pd.read_csv(loss_landscape)
        if trajectory is not None:
            self.trajectory = pd.read_csv(trajectory)

    def loss_contour(self):
        x = self.loss_landscape["x"].to_numpy()
        y = self.loss_landscape["y"].to_numpy()
        z = self.loss_landscape["loss"].to_numpy()
        resolution = round(np.sqrt(len(self.loss_landscape)))
        x = x.reshape(resolution, resolution)
        y = y.reshape(resolution, resolution)
        z = z.reshape(resolution, resolution)
        plot.contour(x, y, z, MNIST_CONTOUR_ARRAY, linewidths=0.5)
        color_bar = plot.colorbar(format="%.3f")
        plot.margins(x=-0.3, y=-0.3)

    def contour_trajectory(self, participant_count, to_plot=None):
        points = []
        for i in range(len(self.trajectory)):
            points.append((self.trajectory.loc[i]["x"], self.trajectory.loc[i]["y"]))
        for j in range(participant_count):
            samples = points[j::participant_count]
            if to_plot is None or j in to_plot:
                for i in range(len(samples) - 2):
                    x, y = samples[i]
                    dx = samples[i + 1][0] - samples[i][0]
                    dy = samples[i + 1][1] - samples[i][1]
                    plot.arrow(x, y, dx, dy, length_includes_head=True, color=COLORS[j], head_width=0.05)
                x, y = samples[-1]
                dx = samples[-1][0] - samples[-2][0]
                dy = samples[-1][1] - samples[-2][1]
                plot.arrow(x, y, dx, dy, length_includes_head=False, color=COLORS[j], head_width=0.5)

    def loss_surface(self, participants_count=0, trajectory=False):
        x = self.loss_landscape["x"].to_numpy()
        y = self.loss_landscape["y"].to_numpy()
        z = self.loss_landscape["loss"].to_numpy()
        z = z.clip(0,100)
        ax = plot.axes(projection="3d")
        cmap = plot.get_cmap('gray_r')
        surf = ax.plot_trisurf(x, y, z, cmap=cmap, zorder=1)
        self.fig.colorbar(surf, ax=ax)
        ax.set_title('loss landscape')
        ax.set_zlabel('loss value')
        if trajectory:
            xx = self.trajectory["x"].to_numpy()
            yy = self.trajectory["y"].to_numpy()
            zz = self.trajectory["loss"].to_numpy()
            for i in range(len(xx)):
                zz[i] = self.find_loss((xx[i], yy[i])) + 10
            for i in range(participants_count):
                ax.plot(xx[i::participants_count], yy[i::participants_count], zz[i::participants_count], "-", color=COLORS[i], alpha=0.5, zorder=2)

    def find_loss(self, point):
        x = self.loss_landscape["x"].to_numpy()
        y = self.loss_landscape["y"].to_numpy()
        z = self.loss_landscape["loss"].to_numpy()
        x_to_find, y_to_find = point
        distance = (x-x_to_find) ** 2 + (y-y_to_find) ** 2
        closest = distance.argsort()[:4]
        loss = np.average(z[closest])
        # print(loss)
        return loss


    def show(self, save_path=None):
        if save_path:
            plot.savefig(save_path)
        plot.show()


if __name__ == '__main__':
    plotter = ThesisPlotter("./playground/records/Landscape2022_01_08_00.csv",
                            "./playground/records/Trajectory2022_01_08_00.csv")
    plotter.loss_contour()
    plotter.contour_trajectory(3)
    # plotter.loss_surface(3, True)
    plotter.show("Traditional_FL.pdf")