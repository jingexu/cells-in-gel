import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from skimage import img_as_ubyte
from skimage.exposure import adjust_sigmoid
from skimage.filters import threshold_otsu, rank, laplace
from skimage.measure import label
from skimage.morphology import square, remove_small_objects
from skimage.color import label2rgb
from skimage.transform import rescale


def frequency_filter(im, mu, sigma, passtype='low'):
    '''
    This function applies a lowpass or highpass filter to an image.

    Paramters
    ---------
    im : (N, M) ndarray
        Grayscale input image.
    mu : float, optional
        Average for input in low pass filter. Default value is 500.
    sigma : float, optional
        Standard deviation for input in low pass filter. Default value is 70.
    passtype: string
        Applies a 'high' or 'low' pass filter. Default value is 'low'.

    Returns
    -------
    out : ndarray
        Low or high pass filtered output image.

    Examples
    --------
    >>> image = plt.imread('C3-NTG-CFbs_NTG5ECM_1mMRGD_20x_003.tif')
    >>> lowpass = frequency_filter(im, 500, 70, passtype='low')
    '''
    # define x and y based on image shape
    y_length, x_length = np.shape(im)
    xi = np.linspace(0, x_length-1, x_length)
    yi = np.linspace(0, y_length-1, y_length)
    x, y = np.meshgrid(xi, yi)

    # define lowpass or highpass filter
    if passtype == 'low':
        gfilt = np.exp(-((x-mu)**2 + (y-mu)**2)/(2*sigma**2))
    if passtype == 'high':
        gfilt = 1 - np.exp(-((x-mu)**2 + (y-mu)**2)/(2*sigma**2))

    fim = np.fft.fft2(im)  # moving to spacial domain
    fim_c = np.fft.fftshift(fim)  # centering
    fim_filt = np.multiply(fim_c, gfilt)  # apply the filter
    fim_uc = np.fft.ifftshift(fim_filt)  # uncenter
    im_pass = np.real(np.fft.ifft2(fim_uc))  # perform inverse transform

    return im_pass


def phalloidin_labeled(im, mu=500, sigma=70, cutoff=0, gain=100,
                       min_size=250, connectivity=1):
    """
    Signature: phalloidin_labeled(*args)
    Docstring: Segment and label image

    Extended Summary
    ----------------
    The colorize function applies preprocessing filters (contrast and high
    pass) then defines the threshold value for the desired image. Thresholding
    is calculated by the otsu function creates a binarized image by setting
    pixel intensities above that thresh value to white, and the ones below to
    black (background). Next, it cleans up the image by filling in random noise
    within the cell outlines and removes small background objects. It then
    labels adjacent pixels with the same value and defines them as a region.
    It returns an RGB image with color-coded labels.

    Paramters
    ---------
    im : (N, M) ndarray
        Grayscale input image.
    cutoff : float, optional
        Cutoff of the sigmoid function that shifts the characteristic curve
        in horizontal direction. Default value is 0.
    gain : float, optional
        The constant multiplier in exponential's power of sigmoid function.
        Default value is 100.
    mu : float, optional
        Average for input in low pass filter. Default value is 500.
    sigma : float, optional
        Standard deviation for input in low pass filter. Default value is 70.
    min_size : int, optional
        The smallest allowable object size. Default value is 250.
    connectivity : int, optional
        The connectvitivy defining the neighborhood of a pixel. Default value
        is 1.

    Returns
    -------
    out : label_image (ndarray) segmented and object labeled for analysis

    Examples
    --------
    >>> image = plt.imread('C3-NTG-CFbs_NTG5ECM_1mMRGD_20x_003.tif')
    >>> label_image = phalloidin_488_binary(image, mu=500, sigma=70,
                                            cutoff=0, gain=100)
    """
    # contrast adjustment
    im_con = adjust_sigmoid(im, cutoff=cutoff, gain=gain, inv=False)

    # contrast + low pass filter
    im_lo = frequency_filter(im_con, mu, sigma, passtype='low')

    # contrast + low pass + binary
    thresh = threshold_otsu(im_lo, nbins=256)
    im_bin = im_lo > thresh

    # fill holes and remove small objects
    im_fill = ndimage.binary_fill_holes(im_bin)
    im_clean = remove_small_objects(im_fill, min_size=min_size,
                                    connectivity=connectivity, in_place=False)

    # labelling regions that are cells
    label_image = label(im_clean)

    # coloring labels over cells
    image_label_overlay = label2rgb(label_image, image=im, bg_label=0)
    print(image_label_overlay.shape)

    # plot overlay image
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    return (label_image)


def sharpen_nuclei(image, selem=square(8), ksize=10, alpha=0.2, sigma=40,
                   imshow=True):
    """
    Highlight nucleis in the image.
    Make a sharp contrast between nucleis and background to highlight nucleis
    in the input image, achieved by mean blurring, laplace sharpening, and
    Gaussian high-pass filter. Selem, ksize, alpha, sigma parameters have
    default values while could be customize by user.

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
