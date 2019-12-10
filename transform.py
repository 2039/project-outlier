from scipy import ndimage, signal
from math import tau
import numpy as np
import util


"""
This module has various complex mathematical transformations
"""



# fourier

def freqtransform(grid, shift=True):
    """
    Fourier/frequency transform
    """
    f = np.fft.fft2(grid) # 2D discrete FT
    fshift = np.fft.fftshift(f) if shift else f

    amplitude = np.abs(fshift)
    angle = np.angle(fshift)

    return amplitude, angle


def amplitude(grid, shift=True):
    """
    Fourier/frequency transform
    """
    f = np.fft.fft2(grid) # 2D discrete FT
    fshift = np.fft.fftshift(f) if shift else f

    return np.abs(fshift)


def phase(grid, shift=True):
    """
    Fourier/frequency transform
    """
    f = np.fft.fft2(grid) # 2D discrete FT
    fshift = np.fft.fftshift(f) if shift else f

    return np.angle(fshift)



# fourinv

def invfreqtransform(amplitude, angle):
    """
    Inverse fourier/frequency transform
    """

    fshift = amplitude * np.exp(1j * angle)

    grid = np.fft.ifft2(fshift)

    # this might be the same as grid.real
    real_grid = np.abs(grid)

    return real_grid



# square/circle/shape

def shape(array, mask_value, shape, radius=5):
    """
    Creates an array with zeros everywhere except in a defined shape
    in the center with ones. (Can be used as a kernel.)
    """

    # width, height
    w, h = array.shape

    # indexes
    xi, yi = np.mgrid[0:w, 0:h]

    # center indexes
    cx = w // 2
    cy = h // 2

    # alias radius
    r = radius

    if shape == "circle":
        # masked indexes
        mask = (xi - cx)**2 + (yi - cy)**2 >= r**2

    if shape == "square":
        # masked indexes
        mask = np.amax((np.abs(xi-cx), np.abs(yi-cy)), axis=0) <= r


    # set masked indexes to null value
    array[mask] = mask_value

    return array



# histogram

def grayscale(array):
    """
    Computes the histogram of the 256 grays in the image
    """

    histogram, _bins = np.histogram(array, 256, (0, 255))

    return histogram



# KDE

def smoothen(array, sigma=3, truncate=4):
    """
    Gaussian smoothing (2d)
    """

    # The following program is a more explicit version of the commented code
    # The filtered array is approximately equal convoluted array,
    #   with an error of ~1E-10 (non-significant)

    # Gaussian filter: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html

    # from scipy import ndimage
    # filtered = ndimage.gaussian_filter(array, sigma=3, output=np.float64, mode="constant", cval=0, truncate=4)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html

    # truncation radius
    radius = int(truncate * sigma + 0.5)

    # First a 1-D Bell distribution, truncated at radius
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)

    # make a 2-D kernel out of it
    kernel = phi_x[:, np.newaxis] * phi_x[np.newaxis, :]
    kernel /= kernel.sum() # normalize the integral to 1

    # compute the convolution
    convolution = signal.convolve2d(array, kernel, mode="same", boundary="fill", fillvalue=0)

    return convolution



# KDE 1d

def smoothen1d(array, sigma=3, truncate=4):
    """
    Gaussian smoothing (1d)
    """

    # not used: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html

    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html

    # truncation radius
    radius = int(truncate * sigma + 0.5)

    # First a 1-D Bell distribution, truncated at radius
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    kernel = np.exp(-0.5 / sigma2 * x ** 2)

    kernel /= kernel.sum() # normalize the integral to 1

    # compute the convolution
    convolution = np.convolve(array, kernel, mode="same")

    return convolution



# center

def center(array):
    """
    Shifts the array such that it is centered at the mean.

    This also changes this mean, because part of the part of the array shifted
    outside the shape is deleted.
    """

    # weighted mean of image, weighted by pixel value
    c_x, c_y = util.center(array)

    # grab height and width from the shape
    h, w = array.shape

    # calculate offset
    shift_x = w//2 - c_x
    shift_y = h//2 - c_y

    # return center-shifted image
    return ndimage.shift(array, (shift_x, shift_y))



# shift

def shift(array, shift_x, shift_y):
    """
    Shifts the image. The part of the array shifted outside the shape is
    deleted.
    """

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html

    return ndimage.shift(array, (shift_x, shift_y))



# rotate

def rotate(array, angle):
    """
    rotates the array, using spline interpolation
    """
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html

    return ndimage.rotate(array, angle - 90, reshape=False)



# scale

def scale(array):
    NotImplemented



#  Principal component analysis

def PCA_angle(array):
    """
    Computes the angle of the principal component of the array
    """
    # https://stats.stackexchange.com/questions/113485/weighted-principal-components-analysis

    from itertools import product
    from scipy import sparse as sp

    # Mass center of image (white pixels are heavier)
    c = util.center(array)

    h, w = array.shape

    # Get an array of the coordinates of the image
    X = np.array(list(product(range(h), range(w))))

    # zero-shift mass center
    X -= c

    # Use pixel values as weights
    W = sp.spdiags(array.reshape(h*w), 0, h*w, h*w)
    W_sum = np.sum(array)

    # Compute the covariance matrix
    cov = X.T @ W @ X / W_sum

    # Find the eigenvectors of the svd
    u, _s, _v = np.linalg.svd(cov)

    # print(u, _s, _v)

    # Get the angle of the principal component
    # Not sure why these components work, possible source of future bug
    angle = np.arctan2(u[1,1], u[0,1])

    return angle



if __name__ == "__main__":
    from plot import Figure

    #--- Smoothing

    if False:
        with Figure(1, 1) as fig:
            ax = fig.ax[0, 0]

            img = util.image()
            img_hi = util.image(simple=False)

            hist = smoothen1d(grayscale(img))
            hist_hi = smoothen1d(grayscale(img_hi))

            ax.plot(hist_hi/hist_hi.sum())

            hist_hi = smoothen1d(grayscale(smoothen(img_hi)))
            ax.plot(hist_hi/hist_hi.sum())

            img = smoothen(img)

            hist = grayscale(img)

            ax.plot(hist/hist.sum())

            hist = smoothen1d(hist)

            ax.plot(hist/hist.sum())

    #--- Centering shift

    if False:
        with Figure(1, 1) as fig:
            ax = fig.ax[0, 0]

            img = util.DUMMY_IMAGE()

            img = center(img)

            ax.imshow(img, cmap="gray")

            fig.update(1)



    #--- Fixed shift

    if False:
        with Figure(1, 1) as fig:
            ax = fig.ax[0, 0]

            key =  ["3337", "1590", "1377"][0]
            for img in util.images():
                img = util.pad(img, 100, 100)

                # angle = PCA_angle(img) / tau * 360
                # img = rotate(img, angle=angle)

                img = shift(img, 20, 50)

                ax.clear()
                ax.imshow(img, cmap="gray")

                fig.update()



    #--- PCA angles

    if False:
        for img in util.progress(util.images()):
            PCA_angle(img)
        input("Press any key to continue")



    #--- PCA angles w/ plot

    if False:
        with Figure(1, 1) as fig:
            ax = fig.ax[0, 0]

            example_image = util.DUMMY_IMAGE()
            #example_image = util.pad(example_image, 100, 100)
            angle = PCA_angle(example_image)

            from math import tau
            print(angle / tau * 360)
            ax.imshow(example_image, cmap="gray")



    #--- Grayscale histogram

    if False:
        with Figure(1, 1) as fig:
            ax = fig.ax[0, 0]

            example_image = util.image()
            ax.plot(grayscale(example_image), color="gray")

            fig.show()



    #--- PCA angles plot
    if False:
        with Figure(2, 1) as fig:
            ax1 = fig.ax[0, 0]
            ax2 = fig.ax[1, 0]

            ax1.set_xlim(-1, 1)
            ax1.set_ylim(-1, 1)
            ax1.set_aspect(aspect=1)

            fig.show()

            for key, img_ in util.images().pad(150, 150):
                angle = PCA_angle(img_)
                ax1.scatter(np.cos(angle), np.sin(angle), color='red', alpha=0.1)
                if angle < 0.1:
                    ax2.imshow(img_, cmap="gray")

                fig.update()


    #--- Ass. examples

    if True:
        array = np.ones((16, 16))
        shape(array, mask_value=0, shape="circle", radius=5)
        print(array)

