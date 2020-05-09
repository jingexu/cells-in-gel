import pandas as pd
from skimage.measure import regionprops_table
from skimage.color import label2rgb

def im_properties(label, im):
    '''
    This function returns a dataframe which includes properties of an object
    identified image

    Paramters
    ---------
    im : (N, M) ndarray
        Grayscale input image.
    label : (N, M) ndarray
        Object labeled image.

    Returns
    -------
    out : pandas dataframe including properties of regions

    Examples
    --------
    >>> image = plt.imread('..\C3-NTG-CFbs_NTG5ECM_1mMRGD_20x_003.tif')
    >>> label, overaly = phalloidin_488_binary(image, mu=500, sigma=70,
                                               cutoff=0, gain=100)
    >>> regions = im_properties(label, image)
    """
    '''
    # define properties of interest
    props = ('area', 'major_axis_length', 'minor_axis_length', 'mean_intensity',
             'eccentricity', 'extent', 'coords')

    # place in pandas dataframe
    regions = pd.DataFrame(regionprops_table(label, im, props))

    return regions
