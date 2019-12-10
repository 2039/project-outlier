import util
import transform
import numpy as np
from plot import Figure

def get_axes(n, m, fig):
    # quickfix
    axes_ = np.empty((n, m), dtype=object)
    double_range=lambda a, b: __import__("itertools").product(range(a),range(b))
    for i, j in double_range(n, m): axes_[i, j] = fig.ax[i, j]
    return axes_

def arrow(col):
    for ax in col: ax.set_axis_off()

    ax.arrow(0, 1.05, 0.9, 0, fc='k', ec='k', lw=1.7,
         head_width=0.2, head_length=0.2, overhang=1.2,
         length_includes_head=True, clip_on=False)

def clean(col):
    for ax in col: ax.set_xticks([]); ax.set_yticks([])

def square(col):
    for ax in col: ax.set_aspect(aspect=1)

def cols(steps): return 3*steps - 1

def block_loop(n):
    c = 3*(n-1)
    R = 2
    for i in range(R):
        for j in range(c, c+2):
            yield i, j

def block_imshow(n, axes, imgs):
    for k, (i, j) in enumerate(block_loop(n)):
        axes[i, j].imshow(imgs[k], cmap="gray")


def block_circle(n, axes, angles):
    from math import tau, pi
    circle_rad = np.linspace(0, tau, 100)
    circle_x = [np.cos(rad) for rad in circle_rad]
    circle_y = [np.sin(rad) for rad in circle_rad]

    line = lambda a: [(np.cos(a), np.cos(a+pi)), (np.sin(a), np.sin(a+pi))]

    for k, (i, j) in enumerate(block_loop(n)):
        axes[i, j].plot(circle_x, circle_y, color='k', lw=1.2)
        axes[i, j].plot(*line(angles[k]), color='k', lw=1.2)



with Figure(2, 5) as fig:
    axes = get_axes(2, 5, fig)

    arrow(axes[..., 2])

    for col_index in [0,1,3,4]:
        clean(axes[..., col_index])
        square(axes[..., col_index])

    angles = lambda: (
        util.images()
            .pad()
            .map(transform.PCA_angle)
    )

    # block 1
    img_mean = util.mean(util.images().pad(100, 100))
    img_3 = [util.pad(util.image(i), 100, 100) for i in["1377", "1590", "9728"]]

    samples = [img_mean] + img_3

    block_imshow(1, axes, samples)

    # block 2
    angle_mean = util.angle_mean(angles())
    angle_3 = list(map(transform.PCA_angle, img_3))

    transformed_samples = [angle_mean] + angle_3

    block_circle(2, axes, transformed_samples)

    fig.save("figure/example.pdf")


