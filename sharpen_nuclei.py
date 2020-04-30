import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import rank, laplace
from skimage.util import img_as_float
from skimage.morphology import square, disk
from skimage import img_as_ubyte


def sharpen_nuclei(image, selem=square(8), ksize=10, alpha=0.2, sigma=40, imshow=True):
    """
    Highlight nucleis in the image.

    Make a sharp contrast between nucleis and background to highlight nucleis in the input image,
    achieved by mean blurring, laplace sharpening, and Gaussian high-pass filter.
    Selem, ksize, alpha, sigma parameters have default values while could be customize by user.


    Parameters
    ----------
    image : numpy.ndarray
        one-color channel image which needs to enhance the nucleis.
    selem : numpy.ndarray
        area used for scanning in blurring, default to be square(8).
    ksize : int
        ksize used for laplace transform, default to be 10.
    alpha : float
        coefficient used in laplace sharpening, default to be 0.2.
    sigma : int
        power coefficient in Gussian filter, default to be 40.
    imshow : bool, str
        users choose whether to show the processed images, default to be True.

    Examples
    ----------
    >>>sharpen_nuclei(image_nuclei, imshow=False)
    [array([[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]]),
    array([[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]])]

    """
    image = img_as_ubyte(image)

    def custom(image):
        imin = np.min(image)
        imax = np.max(image)
        full = imax - imin
        new = (image - imin)/full
        return new

    im = custom(image)
    print(im.shape)

    threshold2 = np.mean(im) + 3*np.std(im)
    print(threshold2)
    im1 = im > threshold2
    im2 = rank.mean(im1, selem)

    im21 = custom(im2)
    threshold3 = np.mean(im21) + np.std(im21)
    print(threshold3)
    im3 = im > threshold3

    im5 = laplace(im2, ksize=ksize)
    im4 = im2 + alpha*im5
    threshold4 = np.mean(im4) + np.std(im4)
    im4 = im4 > threshold4

    xi = np.linspace(0, (im.shape[1]-1), im.shape[1])
    yi = np.linspace(0, (im.shape[0]-1), im.shape[0])
    x, y = np.meshgrid(xi, yi)
    sigma = sigma
    mi = im.shape[1]/2
    ni = im.shape[0]/2
    gfilt = np.exp(-((x-mi)**2+(y-ni)**2)/(2*sigma**2))

    fim = np.fft.fft2(im1)
    fim2 = np.fft.fftshift(fim)
    fim3 = np.multiply(fim2, gfilt)
    fim4 = np.fft.ifftshift(fim3)
    im6 = np.real(np.fft.ifft2(fim4))

    im7 = custom(im6)
    threshold6 = np.mean(im7)+0.2*np.std(im7)
    print(threshold6)
    im7 = im6 > threshold6
    f1 = im4*1
    f2 = im7*1

    if imshow == True:
        fig, ax = plt.subplots(1, 3, figsize=(18, 10))

        ax[0].imshow(image)
        ax[1].imshow(f1, cmap='gray')
        ax[2].imshow(f2, cmap='gray')

        ax[0].set_title('original image', fontsize=25)
        ax[1].set_title('Blur and Laplace', fontsize=25)
        ax[2].set_title('Gaussian Filter', fontsize=25)

        for i in [0, 1, 2]:
            ax[i].axis('off')

    else:
        pass

    return [f1, f2]
