import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

from skimage import io
from skimage.morphology import square, disk

from cells_in_gel import preprocess as pp
from cells_in_gel import properties as props


def max_projection(files):
    '''
    This function takes multiple z-stack image files and creates a dictionary
    of max intesntiy projection images for each original z-stack group.

    Parameters
    ----------
    files : list, str
        A list of filenames each corresponding to z-stack images.

    Returns
    -------
    dictionary with filename as the key and max intensity projection as
    the entry

    Example
    -------
    files = glob.glob('*.tif')
    max_proj = max_projection(files)
    '''
    # empty dictionary for image stacks
    stacks = {}

    # read images into a dictionary - one entry per file
    for file in files:
        stacks[file] = io.imread(file)

    # empty dictionary for max projections
    max_proj = {}

    # find the max projection of each stack
    for file in files:
        max_proj[file] = np.max(stacks[file], axis=0)

    return max_proj


def labels_regions(files, selem=disk(3), min_size=250):
    '''
    This function labels objects in max projections and returns region
    properties for each image in a dictionary.

    Parameters
    ----------
    files : list, str
        A list of filenames each corresponding to z-stack images.
    selem : numpy.ndarray, optional
        Area used for separating cells. Default value is
        skimage.morphology.disk(3).
    min_size : int, optional
        The smallest allowable object size. Default value is 250.

    Returns
    -------
    dictionary with filename as the key and region properties dataframe as
    the entry

    Example
    -------
    files = glob.glob('*.tif')
    regions = labels_regions(files)
    '''
    # create empty dictionaries
    labels = {}
    regions = {}

    # find max projection
    max_proj = max_projection(files)

    # preprocessing and segmentation
    for file in files:
        labels[file] = pp.phalloidin_labeled(max_proj[file], selem=selem,
                                             min_size=min_size)

    # extract region properties
    for file in files:
        regions[file] = props.im_properties(labels[file], max_proj[file])

    return regions

def labels_SMA(files, selem=disk(3), min_size=250):
    '''
    This function labels objects in max projections and returns region
    properties for each image in a dictionary.

    Parameters
    ----------
    files : list, str
        A list of filenames each corresponding to z-stack images.
    selem : numpy.ndarray, optional
        Area used for separating cells. Default value is
        skimage.morphology.disk(3).
    min_size : int, optional
        The smallest allowable object size. Default value is 250.

    Returns
    -------
    dictionary with filename as the key and region properties dataframe as
    the entry

    Example
    -------
    files = glob.glob('*.tif')
    regions = labels_regions(files)
    '''
    # create empty dictionaries
    labels = {}
    regions = {}

    # find max projection
    max_proj = max_projection(files)

    # preprocessing and segmentation
    for file in files:
        labels[file] = pp.SMA_segment(max_proj[file])

    # extract region properties
    for file in files:
        regions[file] = props.im_properties(labels[file], max_proj[file])

    return regions

def image_summaries(files, properties, titles, regions):
    '''
    This funciton returns a final dataframe that summarizes average values and
    variance for each max projection. Dataframe includes group conditions and
    properties of interest.

    Paramters
    ---------
    files : list, str
        A list of filenames each corresponding to z-stack images.
    properties : list, str
        A list of properties of interest for further analysis between groups.
    titles : list, str
        A list of titles that are descriptors for categories between
        underscores in filenames.
    regions : output from labels_regions function

    Returns
    -------
    Pandas dataframe including average values from each max projection.

    Example
    -------
    files = glob.glob('*.tif')
    properties = ['area', 'eccentricity', 'extent',
                  'major_axis_length', 'minor_axis_length',
                  'mean_intesntiy']
    titles = ['celltype', 'ecmtype_conc', 'rgd_conc', 'mag', 'img_num']
    regions = labels_regions(files, selem=disk(3), min_size=250)

    summary = image_summaries(files, properties, titles, regions)
    '''
    # empty dictionary for summary of each image
    summary = {}

    for file in files:

        # split filename by _
        avg = file.split('_')

        for prop in properties:

            # find mean and variance of max projection property
            avg_prop = np.mean(regions[file][prop])
            var_prop = np.var(regions[file][prop])

            # add properties to list with filename details
            avg.append(avg_prop)
            avg.append(var_prop)

        # average properties with max projection into dictionary
        avg.append(len(regions[file]))
        summary[file] = avg

    for prop in properties:

        # add variance to property label
        var_i = [prop]
        var_i.append('_var')
        var = ''.join(var_i)

        # append titles to include properties with variance
        titles.append(prop)
        titles.append(var)

    titles.append('num_objects')

    # create pandas dataframe
    df_avg = pd.DataFrame.from_dict(summary, orient='index', columns=titles)
    df_r = df_avg.reset_index()
    df = df_r.drop(columns=['index'])

    return df


def mergeDict(dict1, dict2):
    '''This funciton returns a merged dictionary that summarizes average values and
    variance for each max projection. It checks if the dictionaries have the same key values and assigns
    the values corresponding to the same key to the same key.

    Paramters
    ---------
    dict1: dict
        The first dictionary to be merged containing one type of average value.
    dict2: dict
        The second dictionary to be merged containt the other type of average value.
    
    Returns
    -------
    dict3: dict
        A dictionary that combines dict1 and dict2.

    Example
    -------
    dict1 = {key1: value1, key2:value2}
    dict2 = {key1: value3, key2: value4}
    
    dict 3 = {key1: [value1, value3], key2:[value2, value4]}.
    '''
    
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = [value , dict1[key]]
 
    return dict3