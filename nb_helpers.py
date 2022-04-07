import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import mediapipe as mp

pose_connections = mp.solutions.pose.POSE_CONNECTIONS
mp_pose = mp.solutions.pose

poselandmarks_list = []
for idx, elt in enumerate(mp_pose.PoseLandmark):
    lm_str = repr(elt).split('.')[1].split(':')[0]
    poselandmarks_list.append(lm_str)


def scale_axes(ax):
    # Scale axes properly
    # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.axes.zaxis.set_ticks([])


def plot_data(data, ax, rotate=True):
    if rotate:
        ax.scatter(data[0, :], data[2, :], -data[1, :])

        for i in pose_connections:
            ax.plot3D([data[0, i[0]], data[0, i[1]]],
                      [data[2, i[0]], data[2, i[1]]],
                      [-data[1, i[0]], -data[1, i[1]]],
                      color='k', lw=1)

        ax.view_init(elev=10, azim=-60)

    else:
        ax.scatter(data[0, :], data[1, :], data[2, :])

        for i in pose_connections:
            ax.plot3D([data[0, i[0]], data[0, i[1]]],
                      [data[1, i[0]], data[1, i[1]]],
                      [data[2, i[0]], data[2, i[1]]],
                      color='k', lw=1)

        ax.view_init(elev=-90, azim=-90)


def rotate_and_save(figure, axis, filename, save=False):
    def init():
        return figure,

    def animate(i):
        axis.view_init(elev=10., azim=i)
        return figure,

    # Animate
    anim = animation.FuncAnimation(figure, animate, init_func=init,
                                   frames=360, interval=20, blit=True)
    plt.close()

    # Save
    if save:
        anim.save(filename, fps=30, extra_args=['-vcodec', 'libx264'], dpi=300)


def time_animate(data, figure, ax, rotate_data=True, rotate_animation=False):
    frame_data = data[:, :, 0]
    if rotate_data:
        plot = [ax.scatter(frame_data[0, :], frame_data[2, :], -frame_data[1, :], color='tab:blue')]

        for i in pose_connections:
            plot.append(ax.plot3D([frame_data[0, i[0]], frame_data[0, i[1]]],
                                  [frame_data[2, i[0]], frame_data[2, i[1]]],
                                  [-frame_data[1, i[0]], -frame_data[1, i[1]]],
                                  color='k', lw=1)[0])

        ax.view_init(elev=10, azim=-60)

    else:
        ax.scatter(frame_data[0, :], frame_data[1, :], frame_data[2, :], color='tab:blue')

        for i in pose_connections:
            ax.plot3D([frame_data[0, i[0]], frame_data[0, i[1]]],
                      [frame_data[1, i[0]], frame_data[1, i[1]]],
                      [frame_data[2, i[0]], frame_data[2, i[1]]],
                      color='k', lw=1)

        ax.view_init(elev=-90, azim=-90)

    scale_axes(ax)

    def init():
        return figure,

    def animate(i):
        frame_data = data[:, :, i]

        for idxx in range(len(plot)):
            plot[idxx].remove()

        plot[0] = ax.scatter(frame_data[0, :], frame_data[2, :], -frame_data[1, :], color='tab:blue')

        idx = 1
        for pse in pose_connections:
            plot[idx] = ax.plot3D([frame_data[0, pse[0]], frame_data[0, pse[1]]],
                                  [frame_data[2, pse[0]], frame_data[2, pse[1]]],
                                  [-frame_data[1, pse[0]], -frame_data[1, pse[1]]],
                                  color='k', lw=1)[0]
            idx += 1

        if rotate_animation:
            ax.view_init(elev=10., azim=-60 + (360 / data.shape[-1]) * i)

        return figure,

    # Animate
    anim = animation.FuncAnimation(figure, animate, init_func=init,
                                   frames=144, interval=20, blit=True)

    plt.close()

    return anim
