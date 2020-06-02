def zlink(df):
    
    def dist(a, b):
        v = ((b[0]-a[0])**2 + (b[1]-a[1])**2)**.5
        return v

    # initialize cell number to an invalid value
    df['cell'] = -1
    # sort by image num, so can iter over whole df in order easily
    df = df.sort_values('Image number', ascending = [1]).reset_index()

    # figure out what cross sections are in the same cells
    for j, r in df.iterrows():
        # get center of current cross section
        r_cent = [r['centroid-0'], r['centroid-1']]
        # get centers of cross sections in prev slice
        prev = df[df['Image number'] == r['Image number']-1]
        # get cross sections from prev slice that are "close" to current cross section
        close = prev[(dist(r_cent, [prev['centroid-0'], prev['centroid-1']]) < 20)]
        # if there cross section close to current, current probably part of same cell
        if not close.empty:
            # choose an close cell to be part of
            df['cell'].iat[j] = close['cell'].min()
        else:
            # otherwise, is a new cell, so give it a new cell number
            df['cell'].iat[j] = df['cell'].max()+1
    return df