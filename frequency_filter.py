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
    import numpy as np

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

    fim = np.fft.fft2(im) # moving to spacial domain
    fim_c = np.fft.fftshift(fim) # centering
    fim_filt = np.multiply(fim_c, gfilt) # apply the filter
    fim_uc = np.fft.ifftshift(fim_filt) # uncenter
    im_pass = np.real(np.fft.ifft2(fim_uc)) # perform inverse transform

    return im_pass
