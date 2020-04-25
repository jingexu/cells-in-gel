def phalloidin_488_binary(im, mu=500, sigma=70, cutoff=0, gain=100):
    """
    This function binarizes a phalloidin 488 fluorescence microscopy channel.

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

    Returns
    -------
    out : ndarray
        Contrast adjusted, low pass filtered, binarized output image.

    Examples
    --------
    >>> image = plt.imread('..\C3-NTG-CFbs_NTG5ECM_1mMRGD_20x_003.tif')
    >>> binary = phalloidin_488_binary(image, mu=500, sigma=70, cutoff=0, gain=100)

    """
    from skimage.exposure import adjust_sigmoid
    from skimage.filters import threshold_otsu

    import frequency_filter as ff

    # contrast adjustment
    im_con = adjust_sigmoid(im, cutoff=0, gain=100, inv=False)

    # contrast + low pass filter
    im_con_lo = ff.frequency_filter(im_con, mu, sigma, passtype='low')

    # contrast + low pass + binary
    thresh_lo = threshold_otsu(im_con_lo, nbins=256)
    im_con_lo_bin = im_con_lo > thresh_lo

    return im_con_lo_bin
