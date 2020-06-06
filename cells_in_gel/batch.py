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
    max_proj = cells_in_gel.batch.max_projection(files)
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
    regions = cells_in_gel.batch.labels_regions(files)
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
    This function labels objects in aSMA channel max projections and returns region
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
    regions = cells_in_gel.batch.labels_regions(files, selem=disk(3), min_size=250)

    summary = cells_in_gel.batch.image_summaries(files, properties, titles, regions)
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


def groups(summary):
    '''
    This function takes a summary dataframe input that includes all max
    projections from the experiment. It then takes the average area, area_var,
    eccentricity, eccentricity_var, extent, and extent_var for each group.
    Next, it counts total number of objects in each group. The result is a
    final dataframe with each experimental group represented by a row in the
    dataframe.

    Parameters
    ----------
    summary : dataframe
        Pandas dataframe from cells_in_gels.batch.image_summaries

    Returns
    -------
    Dataframe with each row representing an experimental group identified by
    'index' row.

    Example
    -------
    files = glob.glob('C3*.tif')
    titles = ['celltype', 'ecmtype_conc', 'rgd_conc', 'mag', 'img_num']
    properties = ['area', 'eccentricity', 'extent']
    regions = cells_in_gel.batch.labels_regions(files, selem, min_size=350)
    summary = cells_in_gel.batch.image_summaries(files, properties, titles,
                                                  regions)
    groups = cells_in_gel.batch.groups(summary)
    '''
    # extract information from summary dataframe
    all_columns = list(summary)
    categ = all_columns[0:3]
    columns = all_columns[5:12]

    cells = (summary[categ[0]].unique()).tolist()
    ecms = (summary[categ[1]].unique()).tolist()
    rgds = (summary[categ[2]].unique()).tolist()

    # empty list group_names to include cell type, ecm type, rgd concentration
    group_names = []
    # create empty dictionary where group_names are the keys for each group
    groups = {}

    # fill group_names and groups for each variation of cells, ecms, and rgds
    for rgd in rgds:
        for cell in cells:
            for ecm in ecms:
                name = cell + '_' + ecm + '_' + rgd
                group_names.append(name)
                groups[name] = summary.groupby(categ).get_group((cell, ecm,
                                                                 rgd))

    # empty list for total properties within a group
    tot = []

    # find average of most properties and sum of total objects in each group
    for name in group_names:
        gr = []
        for column in columns:
            if column == 'num_objects':
                gr.append(groups[name][column].sum())
            else:
                gr.append(groups[name][column].mean())
        tot.append(gr)

    # combine in dataframe that summarizes each group
    fin = pd.DataFrame(tot, group_names, columns)
    final = fin.reset_index()

    return final


def groupby_property(groups, properties):
    '''
    This funciton takes a summary dataframe and returns groups based on
    properties. Group 1 returns I6QTTA cells with no rgd, group 2 returns
    I6QTTA cells with rgd, group 5 includes NTG cells with no rgd, and group 4
    includes NTG cells with rgd. Each group lists five values corresponding to
    ECM variations.

    Parameters
    ----------
    groups : DataFrame
        Pandas DataFrame from cells_in_gel.batch.groups(summary).
    properties : list, str
        A list of properties extracted from the dataframe.

    Returns
    -------
    Lists of groups. Each entry in the list is a list of values for a certain
    property (in the same order as properties list).

    Examples
    --------
    files = glob.glob('C3*.tif')
    titles = ['celltype', 'ecmtype_conc', 'rgd_conc', 'mag', 'img_num']
    properties = ['area', 'eccentricity', 'extent']
    regions = cells_in_gel.batch.labels_regions(files, selem, min_size=350)
    summary = cells_in_gel.batch.image_summaries(files, properties, titles,
                                                  regions)
    groups = cells_in_gel.batch.groups(summary)
    cells_in_gel.batch.groupby_property(groups, properties)
    '''
    g1 = []  # empty list of I6QTTA no rgd group
    g2 = []  # empty list of I6QTTA rgd group
    g3 = []  # empty list of NTG no rgd group
    g4 = []  # empty list of NTG rgd group

    # get all properties from group
    for prop in properties:
        g1.append(groups[prop][0:5].values.tolist())
        g2.append(groups[prop][10:15].values.tolist())

        g3.append(groups[prop][5:10].values.tolist())
        g4.append(groups[prop][15:20].values.tolist())

    return g1, g2, g3, g4


def group_sem(groups, variance):
    '''
    This funciton takes a summary dataframe and returns standard error of mean
    from groups. Group 1 returns I6QTTA cells with no rgd, group 2 returns
    I6QTTA cells with rgd, group 5 includes NTG cells with no rgd, and group 4
    includes NTG cells with rgd. Each group lists five values corresponding to
    ECM variations.

    Parameters
    ----------
    groups : DataFrame
        Pandas DataFrame from cells_in_gel.batch.groups(summary).
    variance : list, str
        A list of variance titles extracted from the dataframe.

    Returns
    -------
    Lists of groups. Each entry in the list is a list of SEM for a certain
    property (in the same order as variance list).

    Examples
    --------
    files = glob.glob('C3*.tif')
    titles = ['celltype', 'ecmtype_conc', 'rgd_conc', 'mag', 'img_num']
    properties = ['area', 'eccentricity', 'extent']
    regions = cells_in_gel.batch.labels_regions(files, selem, min_size=350)
    summary = cells_in_gel.batch.image_summaries(files, properties, titles,
                                                  regions)
    groups = cells_in_gel.batch.groups(summary)
    cells_in_gel.batch.group_sem(groups, variance)
    '''
    g1_sem = []  # empty list of I6QTTA no rgd group
    g2_sem = []  # empty list of I6QTTA rgd group
    g3_sem = []  # empty list of NTG no rgd group
    g4_sem = []  # empty list of NTG rgd group

    # get all sem from group
    for var in variance:
        g1_var = groups[var][0:5].values.tolist()
        g1_n = groups['num_objects'][0:5].values.tolist()
        g1_sem.append(np.sqrt(g1_var)/np.sqrt(g1_n))

        g2_var = groups[var][10:15].values.tolist()
        g2_n = groups['num_objects'][10:15].values.tolist()
        g2_sem.append(np.sqrt(g2_var)/np.sqrt(g2_n))

        g3_var = groups[var][5:10].values.tolist()
        g3_n = groups['num_objects'][5:10].values.tolist()
        g3_sem.append(np.sqrt(g3_var)/np.sqrt(g3_n))

        g4_var = groups[var][15:20].values.tolist()
        g4_n = groups['num_objects'][15:20].values.tolist()
        g4_sem.append(np.sqrt(g4_var)/np.sqrt(g4_n))

    yerr_g1 = {}
    yerr_g2 = {}
    yerr_g3 = {}
    yerr_g4 = {}

    for n in range(3):
        yerr_g1[n] = [i * 2 for i in g1_sem[n].tolist()]
        yerr_g2[n] = [i * 2 for i in g2_sem[n].tolist()]
        yerr_g3[n] = [i * 2 for i in g3_sem[n].tolist()]
        yerr_g4[n] = [i * 2 for i in g4_sem[n].tolist()]

    return yerr_g1, yerr_g2, yerr_g3, yerr_g4


def graph(g1, g2, g3, g4, yerr_g1, yerr_g2, yerr_g3, yerr_g4):
    '''
    This function plots each group with 2*SEM error bars.

    Parameters
    ----------
    g1 : list
        First group from cells_in_gel.batch.groupby_property(groups, properties).
        This group corresponds to I6QTTA cells with no rgd.
    g2 : list
        Second group from cells_in_gel.batch.groupby_property(groups, properties).
        This group corresponds to I6QTTA cells with rgd.
    g3 : list
        Third group from cells_in_gel.batch.groupby_property(groups, properties).
        This group corresponds to NTG cells with no rgd.
    g4 : list
        Fourth group from cells_in_gel.batch.groupby_property(groups, properties).
        This group corresponds to NTG cells with rgd.
    yerr_g1 : list
        First group from cells_in_gel.batch.group_sem(groups, variance).
        This corresponds to error bars in I6QTTA cells with no rgd group.
    yerr_g2 : list
        Second group from cells_in_gel.batch.group_sem(groups, variance).
        This corresponds to error bars in I6QTTA cells with rgd group.
    yerr_g3 : list
        Third group from cells_in_gel.batch.group_sem(groups, variance).
        This corresponds to error bars in NTG cells with no rgd group.
    yerr_g4 : list
        Fourth group from cells_in_gel.batch.group_sem(groups, variance).
        This corresponds to error bars in NTG cells with rgd group.
    Returns
    -------
    Graphs of each group.

    Examples
    --------
    files = glob.glob('C3*.tif')
    titles = ['celltype', 'ecmtype_conc', 'rgd_conc', 'mag', 'img_num']
    properties = ['area', 'eccentricity', 'extent']
    regions = cells_in_gel.batch.labels_regions(files, selem, min_size=350)
    summary = cells_in_gel.batch.image_summaries(files, properties, titles,
                                                  regions)
    groups = cells_in_gel.batch.groups(summary)
    g1, g2, g3, g4 = cells_in_gel.batch.groupby_property(groups, properties)
    yerr_g1, yerr_g2, yerr_g3, yerr_g4 = cells_in_gel.batch.group_sem(groups, variance)

    cells_in_gel.batch.graph(g1, g2, g3, g4, yerr_g1, yerr_g2, yerr_g3, yerr_g4)
    '''
    ecms = ['I61Q1ECM', 'I61Q5ECM', 'NTG1ECM', 'NTG5ECM', 'NoECM']
    labels = ecms

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(14, 13))

    # plotting I6QTTA mean cell area
    axes[0, 0].bar(x - width/2, g1[0], width, label='No RGD', yerr=yerr_g1[0]) # without RGC
    axes[0, 0].bar(x + width/2, g2[0], width, label='RGD', yerr=yerr_g2[0]) # with RGD
    axes[0, 0].set_ylabel('Mean Cell Area (pixels)')
    axes[0, 0].set_title('I61QTTA Fibroblasts')
    # axes[0, 0].errorbar(labels, i6_norgd[0], yerr=10, fmt='--o')

    # NTG mean cell area
    axes[0, 1].bar(x - width/2, g3[0], width, label='No RGD', yerr=yerr_g3[0])
    axes[0, 1].bar(x + width/2, g4[0], width, label='RGD', yerr=yerr_g4[0])
    axes[0, 1].set_ylabel('Mean Cell Area (pixels)')
    axes[0, 1].set_title('NTG Fibroblasts')

    # I6QTTA eccentricity
    axes[1, 0].bar(x - width/2, g1[1], width, label='No RGD', yerr=yerr_g1[1])
    axes[1, 0].bar(x + width/2, g2[1], width, label='RGD', yerr=yerr_g2[1])
    axes[1, 0].set_ylabel('Eccentricity')
    axes[1, 0].set_title('I61QTTA Fibroblasts')

    # NTG eccentricity
    axes[1, 1].bar(x - width/2, g3[1], width, label='No RGD', yerr=yerr_g3[1])
    axes[1, 1].bar(x + width/2, g4[1], width, label='RGD', yerr=yerr_g4[1])
    axes[1, 1].set_ylabel('Eccentricity')
    axes[1, 1].set_title('NTG Fibroblasts')

    # I61QTTA extent
    axes[2, 0].bar(x - width/2, g1[2], width, label='No RGD', yerr=yerr_g1[2])
    axes[2, 0].bar(x + width/2, g2[2], width, label='RGD', yerr=yerr_g2[2])
    axes[2, 0].set_ylabel('Extent')
    axes[2, 0].set_title('I61QTTA Fibroblasts')

    # NTG extent
    axes[2, 1].bar(x - width/2, g3[2], width, label='No RGD', yerr=yerr_g3[2])
    axes[2, 1].bar(x + width/2, g4[2], width, label='RGD', yerr=yerr_g4[2])
    axes[2, 1].set_ylabel('Extent')
    axes[2, 1].set_title('NTG Fibroblasts')

    for ax in axes.flatten():
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='lower right', framealpha=0.9)

    fig.tight_layout()


def max_proj_wrap(files):
    '''
    This function wraps all batch processing of max intensity projections
    together for the phalloidin channel.

    Parameters
    ----------
    files : list, str
        A list of filenames each corresponding to z-stack images.

    Returns
    -------
    Returns graphs of each group and their corresponding segmented/binarized
    images.

    Examples
    --------
    files = glob.glob('C3*.tif')
    max_proj_wrap(files)
    '''
    titles = ['celltype', 'ecmtype_conc', 'rgd_conc', 'mag', 'img_num']
    properties = ['area', 'eccentricity', 'extent']
    variance = ['area_var', 'eccentricity_var', 'extent_var']
    regions = labels_regions(files, min_size=350)
    summary = image_summaries(files, properties, titles,
                                                  regions)
    grps = groups(summary)
    g1, g2, g3, g4 = groupby_property(grps, properties)
    yerr_g1, yerr_g2, yerr_g3, yerr_g4 = group_sem(grps, variance)

    graph(g1, g2, g3, g4, yerr_g1, yerr_g2, yerr_g3, yerr_g4)

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
