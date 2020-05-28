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
    files : 2D array with str inputs

    Returns
    -------
    dictionary with filename as the key and max intensity projection as
    the entry
    '''
    # read images into a dictionary - one entry per file
    stacks = {}

    for file in files:
        stacks[file] = io.imread(file)

    # find the max projection of each stack
    max_proj = {}

    for file in files:
        max_proj[file] = np.max(stacks[file], axis=0)

    return max_proj


def labels_regions(files, selem=disk(3), min_size=250):
    '''
    This function labels objects in max projections and returns region
    properties for each image.

    Parameters
    ----------
    files : 2D array with str inputs

    Returns
    -------
    dictionary of pandas dataframes corresponding to each max projection
    '''
    # create empty dictionaries
    labels = {}
    regions = {}

    # find max projection
    max_proj = max_projection(files)

    for file in files:
        labels[file] = pp.phalloidin_labeled(max_proj[file], selem=selem,
                                             min_size=min_size)

    for file in files:
        regions[file] = props.im_properties(labels[file], max_proj[file])

    return regions


def image_summaries(files, properties, regions):
    '''
    This funciton returns a final dataframe that summarizes average values
    for each max projection.

    Paramters
    ---------
    files : 2D array

    properties : 2D array (example: properties = ['area', 'eccentricity'])

    regions : output from labels_regions function

    Returns
    -------
    pandas dataframe including average values from each max projection 
    '''
    summary = {}

    for file in files:
        avg = file.split('_')
        for prop in properties:
            avg_prop = np.mean(regions[file][prop])
            sd_prop = np.std(regions[file][prop])
            sem_prop = sd_prop/(len(regions[file]))**(1/2)

            avg.append(avg_prop)
            avg.append(sd_prop)
            avg.append(sem_prop)

        summary[file] = avg

    titles = ['celltype', 'ecmtype_conc', 'rgd_conc', 'mag', 'img_num']
    for prop in properties:

        sd_i = [prop]
        sd_i.append('_sd')
        sd = ''.join(sd_i)

        sem_i = [prop]
        sem_i.append('_sem')
        sem = ''.join(sem_i)

        titles.append(prop)
        titles.append(sd)
        titles.append(sem)

    df_avg = pd.DataFrame.from_dict(summary, orient='index', columns=titles)
    df_r = df_avg.reset_index()
    df = df_r.drop(columns=['index'])

    return df
