import numpy as np
from math import tau, pi # 2*pi = tau
from pathlib import Path
import cv2


"""
This module consists of various helper functions for images and math
"""



IMAGE_PATH = Path("boneage-training-dataset-simple")
IMAGE_COUNT = sum(1 for _ in IMAGE_PATH.iterdir()) # length of generator
# IMAGE_COUNT = 12611

def minmax(x):
    """
    The range of an array
    """
    return (np.min(x), np.max(x))



def circle_dist(a, b):
    """
    The minimum angle between two straight lines through the center of a circle,
    defined by two angles in [-pi, pi).
    """
    # https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    phi = abs((a - b + pi ) % tau - pi)
    return phi if phi < tau/4 else pi-phi


def _read_image(path):
    """
    Reads (gray) image from file and converts to array.

    Black pixels correspond to the value 0, whites to 255.
    """
    img = cv2.imread(path.as_posix())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img



def pad(array, height=100, width=100, pad_value=0):
    """
    This pads the array so that it's size is (at least*) of the
    shape (height, width)

    * if the given height/width is smaller than the image's, nothing
    is padded, and the array remains the same size.
    """
    # Get the initial array size
    array_height, array_width = array.shape

    # Compute the top and bottom padding
    # If odd, add extra padding to bottom
    missing_height = (height - array_height)
    padded_height_top    = max(0, missing_height//2)
    padded_height_bottom = max(0, missing_height//2 + missing_height%2)

    # Compute the left and right padding
    # If odd, add extra padding to right side
    missing_width = (height - array_width)
    padded_width_left    = max(0, missing_width//2)
    padded_width_right   = max(0, missing_width//2 + missing_width%2)

    # padding tuple-tuple
    padding = (
        (padded_height_top, padded_height_bottom),
        (padded_width_left, padded_width_right)
    )

    # Return the padded image array
    return np.pad(
        array,
        pad_width=padding,
        mode="constant",           # pad a constant value
        constant_values=pad_value, # set the constant pad value
    )



def scale(image, antialiasing=True, box_height=100, box_width=100):
    """
    This scales the image such that it is not taller nor wider than the given
    width/height, while keeping the aspect ratio.
    """
    # Get image size
    image_width, image_height = image.size

    # Get scaling constant
    scale = min(box_height/image_height, box_width/image_width)

    # Compute new width and height
    new_width, new_height = map(lambda e: round(scale*e),
                                (image_width, image_height))

    # Return scaled image
    if antialiasing:
        return image.resize((new_width, new_height), Image.ANTIALIAS)
    else:
        return image.resize((new_width, new_height))



def crop(array, height=100, width=100, mode="center"):
    NotImplemented



def image(name="1377", simple=True):
    """
    Loads image from file given the image id (name), and converts to array.

    If simple is false, the hi-res image is loaded, otherwise the low-res one.
    """
    path = Path(f"boneage-training-dataset" + ("-simple" * simple))
    path /= f"{name}.png"

    img = cv2.imread(path.as_posix())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img



class image_ids:
    """
    An ordered sequence of the image ids
    """
    def __init__(self):
        self.path = IMAGE_PATH

    def __iter__(self):
        image_ids_ = [int(file.stem) for file in self.path.iterdir()]
        image_ids_.sort()
        yield from image_ids_

    def __len__(self):
        return IMAGE_COUNT



class images:
    """
    An ordered sequence of the images
    """
    def __init__(self):
        self.path = IMAGE_PATH
        self.width = None
        self.height = None
        self.seq = (
            _read_image(self.path / f"{key}.png") for key in image_ids()
        )

    @property
    def init(self):
        """Empty image"""
        return np.zeros((self.height, self.width))


    def __iter__(self):
        yield from self.seq


    def __len__(self):
        return IMAGE_COUNT


    def pad(self, width=100, height=100):
        """
        Pads the images in the sequence to the specified size.
        """
        self.width = width
        self.height = height

        self.seq = (
            pad(img, width=self.width, height=self.height) for img in self.seq
        )

        return self


    def map(self, func):
        """
        Applies a function to each image in the sequence
        """
        self.seq = map(func, self.seq)

        return self



def DUMMY_IMAGE():
    """
    Creates an example image
    """

    from scipy.stats import multivariate_normal
    from math import sqrt

    R = np.array((
        (0.5, -sqrt(3)/2),
        (sqrt(3)/2, 0.5),
    ))
    L = np.array((
        (0.01, 0),
        (0, 1)
    ))

    mean = np.array((2.5, 0.0))
    cov = R @ L @ np.linalg.inv(R)
    #cov = np.array(((0.5, 0.3), (0.2, 0.5)))

    mvn = multivariate_normal(mean=mean, cov=cov)

    x, y = np.mgrid[-1:1:.1, -1:1:.1]
    pos = np.empty(x.shape + (2,))

    pos[:, :, 0] = x
    pos[:, :, 1] = y

    return mvn.pdf(pos)



def mean(seq, init=None):
    """
    Computes the mean of a sequence
    """

    # reduce := (left) fold
    from functools import reduce
    from operator import add

    # init ??= init.seq
    init = seq.init if init is None else init

    # average
    return reduce(add, seq, init) / len(seq)


def angle_mean(seq, init=0):
    """
    Computes the vector mean of a sequence of angles

    https://en.wikipedia.org/wiki/Mean_of_circular_quantities
    """

    # reduce := (left) fold
    from functools import reduce
    from operator import add

    # init ??= init.seq
    init = seq.init if init is None else init

    # get unit vector given complex angle
    unit = lambda alpha: np.exp(1j * alpha)

    return np.angle(reduce(add, map(unit, seq), init))



def center(array):
    h, w = array.shape

    W = np.sum(array)

    cx = np.sum(array, axis=0) @ np.r_[0:w] / W
    cy = np.sum(array, axis=1) @ np.r_[0:h] / W

    return np.array((cx, cy), dtype=int)



def progress(iterable):
    l = len(iterable)
    progress_ = 0
    for i, element in enumerate(iterable, 1):
        p = (i*100) // l
        if p > progress_:
            progress_ = p
            print(f"{progress_} % complete")
        yield element



if __name__ == "__main__":

    if True:
        a = np.deg2rad(77)
        b = np.deg2rad(79)

        print(np.rad2deg(circle_dist(a, b)))

    img = image()

    x, y = np.mgrid[0:10, 0:10]

    print(center(img))
    print(mean(images().pad(100, 100)))
