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


def counting_nucleis(imagelist):
    """
    Enhance and count the nucleis in a list of image.
    For every image in the input list, convert it to grayscale and enhance nucleis in the image
    by calling `enhance_nucleis` function. Then label the preprocessed image in segmented area, 
    regard the numbers of regions minus 1(as the background) in the colored image as the number of 
    nucleis in this image. Operate a for loop in the input image list and assign their counts to a dataframe.

    Parameters
    ----------
    imagelist : list
        a list of rgb images where the nucleis are expected to be sharpened and counted
    ----------

    Return
    ----------
    Return a dataframe composed of the cell type, extracellular matrix type, RGD concentration, 
    and the count of nucleis

    """
    nuclei_counts = []
    cell_type = []
    ecm_cond = []
    rgd_cond = []
    for i in np.arange(len(imagelist)):
        im = plt.imread(imagelist[i])
        im = rgb2gray(im)
        im_thre = enhance_nucleis(im, image_display=False)

        labeled_nucleis, _ = label(im_thre)
        image_label_overlay = label2rgb(labeled_nucleis, image=im)

        props = ('area', 'major_axis_length',
                 'minor_axis_length', 'mean_intensity')
        regions = pd.DataFrame(regionprops_table(
            labeled_nucleis, im_thre, props))
        count = regions.shape[0]-1
        nuclei_counts.append(count)

        name = imagelist[i][94:]
        a = name.split('_')
        cell_type.append(a[0])
        ecm_cond.append(a[1])
        rgd_cond.append(a[2])

        dt = pd.DataFrame(
            data=[cell_type, ecm_cond, rgd_cond, nuclei_counts]).T
        dt.columns = ['cell_type', 'ecm_cond', 'rgd_cond', 'counts']

        print(i)

    return dt
