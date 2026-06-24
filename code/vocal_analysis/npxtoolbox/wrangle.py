"""
Data wrangling functions

Specific functions for known data formats, to ease wrangling needs.


"""
# import os
import numpy as np
# import scipy.io as io
import pandas as pd


### utilities for working with nested dictionaries

def get_key_paths(d, key_name, path=None):
    if path is None: # not for user input, for iteration by function!
        path = []
    paths = [] # collect
    if isinstance(d, dict):
        for k,v in d.items():
            new_path = path + [k]
            if k == key_name: # stop when match
                paths.append(new_path)
            # inspect v recursively as its own key:
            paths.extend(get_key_paths(v, key_name, new_path))   
    elif isinstance(d, list): # when value is inspected:
        for i, item in enumerate(d):
            paths.extend(get_key_paths(item, key_name, path + [i]))
    return paths

def get_all_paths(d, path=None):
    if path is None: # not for user input, for iteration by function!
        path = []
    paths = [] # collect
    if isinstance(d, dict):
        for k, v in d.items():
            new_path = path + [k]  
            paths.append(new_path)
            paths.extend(get_all_paths(v, new_path))
    elif isinstance(d, list): # when value is inspected:
        for i, item in enumerate(d):
            paths.extend(get_all_paths(item, path + [i]))
    return paths

def extract_key_data(d, paths):
    results = []
    for i, p in enumerate(paths):
        d_sub = d
        try:
            for key in p:
                d_sub = d_sub[key]
            results.append(d_sub)
        except (KeyError, IndexError, TypeError):
            print('Skipping invalid path at position ' + str(i))
            continue
    return(results)

### utilities for lists
def unique_lists(list_of_lists):
    seen = set()
    return ([lst for lst in list_of_lists if not (tuple(lst) in seen or seen.add(tuple(lst)))])

### utilities for vector/scalar data matching
def match_scalar_df_to_vector_dat(vector_dict, scalar_df, dtype, target_key):
    if dtype == 'spontaneous':
        filt_on = ['rec_n','rec_iter']
    elif dtype == 'playback':
        filt_on = ['condition','n']
    
    key_paths = get_key_paths(vector_dict, target_key)

    if len(scalar_df)!=len(key_paths):
        # cycle through unique recs
        rec_paths = unique_lists([p[0:4] for p in key_paths])
        collect_present = []
        for p in rec_paths:
            # select N rows of paths corresponding to this rec:
            n_calls_in_rec = len([i for i in key_paths if i[0:4] == p])

            samp_df = scalar_df.loc[
                (scalar_df["anim"] == p[0]) &
                (scalar_df["sess"] == p[1]) &
                (scalar_df[filt_on[0]] == p[2]) &
                (scalar_df[filt_on[1]] == int(p[3]))
            ]
            if len(samp_df)!=n_calls_in_rec:                
                print('\tMismatch in number of calls within a rec. detected : ' + str(n_calls_in_rec) + ' calls in vector data vs. ' + str(len(samp_df)) + ' calls in scalar data(frame).' )
                print('\tLocation: ' + p)
            else:
                collect_present.append(samp_df)
        scalar_df_ = pd.concat(collect_present, ignore_index=False)
        print('\tFiltered scalar df. Original length = ' +str(len(scalar_df))+ ' | New length = '+str(len(scalar_df_)))
    else:
        print('\tNumber of rows in vector and scalar data match! Hurray!')
        scalar_df_ = scalar_df
    
    return(scalar_df_)

def filter_selection(df, stfts, timeaxis=None, filter=None):
    # filter is index of rows to keep
    if filter is None:        
        return(df, stfts, timeaxis)
    else:
        df_filt = df.iloc[filter].copy()
        stfts_filt = [stfts[i] for i in filter]
        if timeaxis is not None:
            timeaxis_filt = [timeaxis[i] for i in filter]
        else:
            timeaxis_filt = timeaxis
        return(df_filt, stfts_filt, timeaxis_filt)

### utilities for merging and managing pandas data frames
def rename_and_order_cols(df, loc, colname_old, colname_new):
    # df - df (pandas) feat. col with old name
    # loc - loc (int) of col index where new should go
    # old_col - name (str) of col in df1
    # new_col - name (str) of col in df2
    col_old = df.pop(colname_old)
    df.insert(loc, colname_new, col_old)
    return df

def move_cols_to_end(df, col_names):
    # df
    # cols - list of strings
    for col_name in col_names:
        col_dat = df.pop(col_name)
        df[col_name] = col_dat
    return df

def bin_col_values(df, colname='duration', n_bins=None, binsize=None,
                   log=False, scale=1, label_decimals=1, int_label_threshold=None, new_colname=None):
    """Bin a numerical column into discrete intervals.

    Parameters
    ----------
    df : DataFrame
    colname : str
        Column to bin.
    n_bins : int, optional
        Number of bins. Uses geomspace if log=True, else linspace.
    binsize : float, optional
        Bin width in the original units (only used when log=False).
    log : bool
        Use geometric (log-spaced) bins via np.geomspace. Requires n_bins.
    scale : float
        Multiply bin edges by this factor when generating labels (e.g. 1000 for s→ms).
    label_decimals : int
        Decimal places for bin labels after scaling (for labels below int_label_threshold).
    int_label_threshold : float, optional
        Labels with scaled values >= this are rounded to integers. None disables this.
    new_colname : str, optional
        Name for the output column. Defaults to colname + '_bin'.
    """
    df_ = df.copy()
    min_val = df_[colname].min()
    max_val = df_[colname].max()

    if log:
        if n_bins is None:
            raise ValueError("n_bins is required when log=True")
        if min_val <= 0:
            n_nonpos = (df_[colname] <= 0).sum()
            min_val = df_[colname][df_[colname] > 0].min()
            print(f'log=True: {n_nonpos} value(s) <= 0 in "{colname}" excluded from bin range. Using min = {min_val:.4g}.')
        bins = np.geomspace(min_val, max_val, n_bins)
    elif n_bins is not None:
        bins = np.linspace(min_val, max_val, n_bins)
    elif binsize is not None:
        n_bins = int((max_val - min_val) / binsize)
        bins = np.linspace(min_val, max_val, n_bins + 1)
    else:
        raise ValueError("Provide n_bins or binsize")

    scaled = bins[1:] * scale
    if int_label_threshold is not None:
        bin_labels = [str(int(round(v))) if v >= int_label_threshold
                      else str(round(float(v), label_decimals))
                      for v in scaled]
    else:
        bin_labels = [str(round(float(v), label_decimals)) for v in scaled]

    if new_colname is None:
        new_colname = colname + '_bin'

    df_[new_colname] = pd.cut(
        df_[colname],
        bins=bins,
        labels=bin_labels,
        include_lowest=True
    )

    if df_[new_colname].isna().sum() != 0:
        print('Binning missed some values. Inspect.')

    return df_

def fix_strings(df, colnames):
    def parse_values(x):
        if isinstance(x, list):
            return x
        if isinstance(x, float) and pd.isna(x):
            return []
        if isinstance(x, str):
            return np.fromstring(x.strip('[]'), sep=' ').tolist()
        return x  
    df_out = df.copy()
    for col in colnames:
        df_out[col] = (df_out[col].apply(parse_values))
    
    return(df_out)

def prep_combine_dataframes_v1(spont_df,play_df):

    ## ALL CHANGES KNOWN AND DESIRED @23/02/2026
    
    ## --- df1: spontaneous df
    # print('df1:')
    # cols1 = spont_df.columns 
    # print(cols1)

    ## Perform
    ### add cols to match df2
    # condition --> spontaneous
    colname_new = 'condition'
    colloc_new = spont_df.columns.get_loc('treatment')+1
    spont_df.insert(colloc_new, colname_new, 'spontaneous')

    ## rename cols
    # start_trig --> 'start_seconds_trig_n' | time of call onset from nth trigger
    # start_trig1 --> start_seconds_trig_1  | time of call onset from trigger 1 
    spont_df = spont_df.rename(columns={'start_trig': 'start_seconds_trig_n', 'start_trig1': 'start_seconds_trig_1'})
    
    # print(spont_df.columns)

    ## --- df2: playback df
    # print('df2:')
    # cols2 = play_df.columns
    # print(cols2)
    
    ## Perform
    ### rename cols to match df1
    # n --> rec_iter
    colname_old = 'n'; colname_new = 'rec_iter'
    colloc_new = spont_df.columns.get_loc(colname_new)
    play_df = rename_and_order_cols(play_df,colloc_new,colname_old,colname_new)
    
    ### remove cols
    # session 
    # rm_name = 'session'
    # play_df = play_df.drop(rm_name, axis=1)
    
    # print(df2.columns)

    return(spont_df, play_df)


### summary statistics
def group_summary(df, groupby, variables, observed=True):
    """Compute summary statistics for variables across groups.

    Parameters
    ----------
    df : DataFrame
    groupby : list of str
        Columns to group by.
    variables : list of str
        Columns to summarise.
    observed : bool
        Passed to groupby; True avoids FutureWarning for categoricals.

    Returns
    -------
    DataFrame with a MultiIndex column (variable, statistic).
    """
    return (
        df.groupby(groupby, observed=observed)[variables]
        .agg(['mean', 'median', 'std', 'sem', 'count'])
    )


### dfs and umaps
def label_umap_sel(df, iloc_index, label, colname='label', label_default=np.nan):
    # original df
    # col, if doesn't exist create it
    # literal index
    # label to give those items
    
    df_out = df.copy()
    if colname not in df_out.columns:
        df_out[colname] = label_default
    df_out.iloc[iloc_index,df_out.columns.get_loc(colname)] = label
    return(df_out)


