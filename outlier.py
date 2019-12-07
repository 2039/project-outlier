import transform
import util
import numpy as np
import logging

from plot import Figure
from heapq import nlargest

# Set debug level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s -- %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

handler.setFormatter(formatter)

logger.addHandler(handler)

"""
This module finds odd objects
"""



def PCA_angle():
    angles = lambda: (
        util.images()
            .pad()
            .map(transform.PCA_angle)
    )

    logger.info("Calculating PCA angle mean")
    mean = util.angle_mean(angles())

    logger.info("Calculating distance from mean")
    distances = ((util.circle_dist(mean, angle), angle) for angle in angles())

    N = 10
    logger.info(f"Finding {N} angles farthest away from mean")
    oddities = nlargest(N, zip(distances, util.image_ids()))

    logger.info(f"Finished PCA angle outliers")
    return oddities

def mean():
    images = lambda: util.images().pad(100, 100)

    logger.info("Calculating mean image")
    mean = util.mean(images)

def rotated_mean():
    pass

def shifted_mean():
    pass

def rotated_and_shifted_mean():
    pass

def frequency_mean():
    pass

def amplitude_mean():
    pass

def histogram_mean():
    pass

# add smoothed versions of the above functions with varying degree of smoothness

def smoothed_mean():
    pass


if __name__ == "__main__":
    if True:
        result = PCA_angle()

        print(result)

        with Figure(1, 1) as fig:
            ax = fig.ax[0, 0]

            for error, key in result:
                img = util.image(key)
                ax.imshow(img, cmap="gray")

                fig.update(2)
