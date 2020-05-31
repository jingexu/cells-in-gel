import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from skimage import img_as_ubyte
from skimage.util import img_as_float
from skimage.exposure import adjust_sigmoid
from skimage.filters import threshold_otsu, threshold_triangle, rank, laplace
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square, disk, remove_small_objects
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
    >>> image = plt.imread('..\C3-NTG-CFbs_NTG5ECM_1mMRGD_20x_003.tif')
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


def phalloidin_488_segment(im, mu=500, sigma=70, cutoff=0, gain=100,
                           min_size=250, connectivity=1):
    """
    This function binarizes a phalloidin 488 fluorescence microscopy channel
    using contrast adjustment, high pass filter, otsu thresholding, and removal
    of small objects.

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
    out : label_image (ndarray) segmented and object labeled for analysis,
        image_label_overlay (ndarray)

    Examples
    --------
    >>> image = plt.imread('..\C3-NTG-CFbs_NTG5ECM_1mMRGD_20x_003.tif')
    >>> label, overaly = phalloidin_488_binary(image, mu=500, sigma=70,
                                               cutoff=0, gain=100)

    """
    # contrast adjustment
    im_con = adjust_sigmoid(im, cutoff=cutoff, gain=gain, inv=False)

    # contrast + low pass filter
    im_lo = frequency_filter(im_con, mu, sigma, passtype='low')

    # contrast + low pass + binary
    thresh = threshold_otsu(im_lo, nbins=256)
    im_bin = im_lo > thresh

    # remove small objects
    im_bin_clean = remove_small_objects(im_bin, min_size=min_size,
                                        connectivity=connectivity,
                                        in_place=False)
    # labelling regions that are cells
    label_image = label(im_bin_clean)

    # coloring labels over cells
    image_label_overlay = label2rgb(label_image, image=im, bg_label=0)

    return label_image, image_label_overlay


def colorize(image, i, x):
    """
    Signature: colorize(*args)
    Docstring: segment and label image
    Extended Summary:
    ----------------
    The colorize function defines the threshold value for the desired image by
    the triangle function and then creates a binarized image by setting pixel
    intensities above that thresh value to white, and the ones below to black
    (background). Next, it closes up the image by filling in random noise
    within the cell outlines and smooths/clears out the border. It then labels
    adjacent pixels with the same value and defines them as a region. It
    returns an RGB image with color-coded labels.

    Parameters:
    ----------
    image : 2D array
            greyscale image
    i : int
        dimension of square to be used for binarization
    x : float
        dimension of image in microns according to imageJ
    Returns:
    --------
    RGB image overlay
    int : 2D ndarray
    """
    # resizing image
    image = rescale(image, x/1024, anti_aliasing=False)
    # applying threshold to image
    thresh = threshold_triangle(image)
    binary = closing(image > thresh, square(i))
    binary = ndimage.binary_fill_holes(binary)

    # cleaning up boundaries of cells
    cleared = clear_border(binary)

    # labelling regions that are cells
    label_image = label(cleared)

    # coloring labels over cells
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
    print(image_label_overlay.shape)

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
        grayscale image which needs to enhance the nucleis.
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

    Returns
    ----------
    Return to 2 processed grayscale images with sharpened nucleis(2 dimension arrays)
    in the image using two different sharpening styles.
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


def enhance_nucleis(image, open_selem=disk(5), image_display=True):
    """
    Highlight nucleis in the image.
    Make a sharp contrast between nucleis and background to highlight nucleis
    in the input image, achieved by opening, dilation, sobel, watershed, and threshod. 
    Selem have default values while could be customize by user.

    Parameters
    ----------
    image : numpy.ndarray
        grayscale image which needs to enhance the nucleis.
    selem : numpy.ndarray
        area used for opening process, default to be disk(5).
    image_display : bool, str
        users choose whether to show the enhanced images, default to be True.
    ----------

    Return
    ----------
    Return a processed grayscale image(2 dimension array) with enhanced nucleis.

    """

    im1 = img_as_ubyte(image)
    im_open = m.opening(im1, open_selem)
    elevation_map = img_as_ubyte(m.dilation(sobel(im_open)), disk(4))
    im2 = m.watershed(elevation_map, im_open)
    im22 = (im2 > otsu(im2))*1
    im3 = m.erosion((fillholes(elevation_map))*1, disk(2))

    if image_display == True:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(im22, cmap='gray')
        ax[1].imshow(im3, cmap='gray')
        ax[0].set_title('method1', fontsize=20)
        ax[1].set_title('method2', fontsize=20)
        ax[0].axis('off')
        ax[1].axis('off')
    else:
        pass

    return im22


def list_of_images(image_channel):
    """
    Automatically extract all the images belonging to a channel into a list

    Parameters
    ----------
    image_channel : str
        the channel of which user wants to extract images
    ----------

    Returns
    ----------
    Return to a list composed of all images belonging to a channel

    """
    mypath = '/content/drive/My Drive/Fibroblasts in PEG Gels with NTG I61Q exm/20x TIFFs/Channel Separated'
    namelist = []
    tifflist = []
    for root, dirs, files in os.walk(mypath):
        for name in files:
            if name[0:2] == image_channel:
                namelist.append(name)
                j = os.path.join(root, name)
                tifflist.append(j)

    return tifflist
