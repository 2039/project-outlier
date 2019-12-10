import transform
import util
import numpy as np
import logging

from plot import Figure
from heapq import nlargest

"""
This module finds anomalies

[ ] https://en.wikipedia.org/wiki/Distance_correlation
"""

# Set logging parameters
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



def PCA_angle(N=100):
    """
    Computes the mean angle, the finds N angles that deviates the most from
    the mean, modulo pi.

    There was an issue with the angle of the PCA to be inverted 180 degrees.
    This shouldn't affect the mean too much, but caused some images to be
    incorrectly classified as anomalies. Thus, we consider only angles
    between [-90, 90).
    """

    angles = lambda: util.images().pad(100, 100).map(transform.PCA_angle)

    logger.info("Calculating PCA angle mean")
    mean = util.angle_mean(angles())

    # distances from mean angle
    distances = (util.circle_dist(mean, angle) for angle in angles())

    logger.info(f"Finding {N} angles farthest away from mean")
    oddities = nlargest(N, zip(distances, util.image_ids()))

    logger.info(f"Finished PCA angle outliers")
    return oddities



def mean(N=100):
    images = lambda: util.images().pad(100, 100)

    logger.info("Calculating mean image")
    mean = util.mean(images())

    # max-correlation of mean
    d = util.max_correlation
    d = np.linalg.norm
    maxcorr = (d(img - mean) for img in util.progress(images()))

    logger.info(f"Finding {N} with least correlation with the mean")
    oddities = nlargest(N, zip(maxcorr, util.image_ids()))

    logger.info(f"Finished maxcorr outliers")
    return oddities



def rotated_mean(N=100):
    images = lambda: util.images().pad(100, 100)
    angles = lambda: util.images().pad(100, 100).map(transform.PCA_angle)
    rotated = lambda: util.seq(transform.rotate(img, ang) for img, ang in zip(images(), angles()))

    logger.info("Calculating mean rotated image")
    mean_r = util.mean(rotated(), np.zeros((100, 100)))

    # distance
    d = np.linalg.norm
    dist = (d(img - mean_r) for img in util.progress(rotated()))

    logger.info(f"Finding {N} farthest away from the mean")
    oddities = nlargest(N, zip(dist, util.image_ids()))

    logger.info(f"Finished rotated mean outliers")
    return oddities



def centered_mean(N=100):
    centered = lambda: util.images().pad(100, 100).map(transform.center)

    logger.info("Calculating mean centered image")
    mean_c = util.mean(centered(), np.zeros((100, 100)))

    # distance
    d = np.linalg.norm
    dist = (d(img - mean_c) for img in util.progress(centered()))

    logger.info(f"Finding {N} farthest away from the mean")
    oddities = nlargest(N, zip(dist, util.image_ids()))

    logger.info(f"Finished centered mean outliers")
    return oddities



def rotated_and_centered_mean(N=100):
    images = lambda: util.images().pad(100, 100)
    angles = lambda: util.images().pad(100, 100).map(transform.PCA_angle)
    rotated = lambda: util.seq(transform.rotate(img, ang) for img, ang in zip(images(), angles()))
    rotated_and_centered = lambda: util.seq(map(transform.center, rotated()))

    logger.info("Calculating mean rotated and centered image")
    mean_rc = util.mean(rotated_and_centered(), np.zeros((100, 100)))

    # distance
    d = np.linalg.norm
    dist = (d(img - mean_rc) for img in util.progress(rotated_and_centered()))

    logger.info(f"Finding {N} farthest away from the mean")
    oddities = nlargest(N, zip(dist, util.image_ids()))

    logger.info(f"Finished rotated and centered mean outliers")
    return oddities



def frequency_mean(N=100):
    phases = lambda: util.images().pad(100, 100).map(transform.phase)

    logger.info("Calculating mean phase")
    mean_p = util.mean(phases(), np.zeros((100, 100)))

    # mean_p = transform.shape(mean_p, shape="circle", mask_value=0, radius=25)

    # distance
    d = np.linalg.norm
    d = util.max_correlation
    dist = (-d(phase, mean_p) for phase in util.progress(phases()))

    logger.info(f"Finding {N} farthest away from the mean")
    oddities = nlargest(N, zip(dist, util.image_ids()))

    logger.info(f"Finished phase mean outliers")
    return oddities


def amplitude_mean(N=100):
    amps = lambda: util.images().pad(100, 100).map(transform.amplitude)

    logger.info("Calculating mean amplitude")
    mean_a = util.mean(amps(), np.zeros((100, 100)))

    # distance
    d = np.linalg.norm
    dist = (d(amp - mean_a) for amp in util.progress(amps()))

    logger.info(f"Finding {N} farthest away from the mean")
    oddities = nlargest(N, zip(dist, util.image_ids()))

    logger.info(f"Finished amplitude mean outliers")
    return oddities



def histogram_mean(N=100):
    hists = lambda: util.images().map(transform.smoothen).map(transform.grayscale)

    logger.info("Calculating mean histogram")
    mean_h = util.mean(hists(), np.zeros((256)))

    # distance
    d = np.linalg.norm
    d = util.max_correlation1d
    dist = (d(hist, mean_h) for hist in util.progress(hists()))

    logger.info(f"Finding {N} farthest away from the mean")
    oddities = nlargest(N, zip(dist, util.image_ids()))

    logger.info(f"Finished histogram mean outliers")
    return oddities



if __name__ == "__main__":
    import time, timeit
    if True:

        a1, a2, a3 = time.time(), time.process_time(), timeit.default_timer()
        result = frequency_mean(10)
        b1, b2, b3 = time.time(), time.process_time(), timeit.default_timer()

        #print(b1-a1, b2-a2, b3-a3)

        #print(result)

        with Figure(1, 1) as fig:
            ax = fig.ax[0, 0]

            for error, key in result:
                img = util.pad(util.image(key), 100, 100)
                ax.imshow(img, cmap="gray")

                fig.update(2)
                input("Press any key to continue")
