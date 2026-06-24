###
# Run UMAP on [joint] stft data
# INPUT: cache_dir + '/*acoustic_data*.npy'
# OUTPUT: cache_dir + '/stft_umap/*stft_umap_data*.npy' - stft input to umap, before transforms (e.g. zero padding)
#                      '/stft_umap/*stft_umap_annotations_data*.csv' - scalar data file + embedding and clustering
# ENV: spikey_x86 on cluster
###

import os, sys
sys.path.append(os.path.join('/cs/projects/BWFAFdeactivNpx/code') ) 
from _frontal_vocal_paper_repo.code.vocal_analysis.npxtoolbox import acoustics
from _frontal_vocal_paper_repo.code.vocal_analysis.npxtoolbox import wrangle
import numpy as np
import pandas as pd
import das_unsupervised.spec_utils
import umap
import scipy.cluster.vq


### prepare to load acoustic data 
cache_dir = '/cs/projects/BWFAFdeactivNpx/Npv1_data/Npv1_cachedata'

call_vec_data_fs = {
 'spontaneous': cache_dir + '/v1_spontaneous_calls_acoustic_data_v3.npy',
 'playback': cache_dir + '/v1_playback_calls_acoustic_data_v2.npy'
 }
call_annot_data_fs = {
 'spontaneous': cache_dir + '/v1_spontaneous_annotations_data_v3_aug.csv',
 'playback': cache_dir + '/v1_playback_annotations_data_v2_aug.csv' 
}

## params
datasets_to_load = ['spontaneous','playback']
target_dat_key = 'stft_x_cg'

remove_outliers = True
logdur = False # log the time axis to normalize durations?

# umap params
seed=24072026
nneighbors=40
mindist=0.4

# cluster params
k_desired=5
# uses same seed as umap

## load data
key_paths_datasets = []
vector_datasets = []
scalar_datasets = []
for dataset in datasets_to_load:
    print(dataset)
    # load vector data
    vector_dat = np.load(call_vec_data_fs[dataset], allow_pickle=True).all()
    # filter for desired data type
    key_paths = wrangle.get_key_paths(vector_dat, target_dat_key)
    target_data = wrangle.extract_key_data(vector_dat, key_paths)
    print('\tNumber of calls: ' + str(len(target_data)))
    # scalar data (pandas df)
    scalar_dat_raw = pd.read_csv(call_annot_data_fs[dataset])
    # filter scalar data for data present in vector data
    scalar_dat = wrangle.match_scalar_df_to_vector_dat(vector_dat, scalar_dat_raw, dataset, target_dat_key)
    # optional: inspect what was removed from scalar_dat_raw
    # scalar_dat_discrep = scalar_dat_raw.iloc[np.setxor1d(scalar_dat.index,scalar_dat_raw.index)]

    # combine data
    key_paths_datasets.extend(key_paths)
    vector_datasets.extend(target_data)

    # append but dont merge
    scalar_datasets.append(scalar_dat)

## marge dfs    
# renames some variables so that the column names are consistent
df_spont, df_play = wrangle.prep_combine_dataframes_v1(scalar_datasets[0], scalar_datasets[1])
df_sum = pd.concat([df_spont, df_play])
# convenience
df_sum = wrangle.move_cols_to_end(df_sum, ['duration','ici','n_calls'])
df_sum = wrangle.bin_col_values(df_sum, colname='duration')

## fetch time axes
stft_timeaxis_datasets = []
for dataset in datasets_to_load:
    vector_dat = np.load(call_vec_data_fs[dataset], allow_pickle=True).all()
    target_data = wrangle.extract_key_data(vector_dat, wrangle.get_key_paths(vector_dat, 'stft_t_cg'))
    stft_timeaxis_datasets.extend(target_data)

## filter for rows with stft data
stft_1_idx = np.where([len(s)!=0 for s in vector_datasets])[0]
# print('N calls to umap: ' + str(len(stft_1_idx)))
print('Percentage of data available for umap: ' + str(np.round(len(stft_1_idx)/len(vector_datasets),2)))

stft_data_1 = [vector_datasets[i] for i in stft_1_idx]
taxis_data_1 = [stft_timeaxis_datasets[i] for i in stft_1_idx]
df_sum_1 = df_sum.iloc[stft_1_idx].copy()

## remove outliers

if remove_outliers:
    # set a maximum duration limit based on time axis from stft computation (so some rounding error)
    max_dur = np.round(np.quantile([i[-1] for i in taxis_data_1],q=[0.99]),5)[0]

    # filter for calls shorter than that
    stft_2_idx = np.where([i[-1]<=max_dur for i in taxis_data_1])[0]
    print('N calls to umap: ' + str(len(stft_2_idx)))
    print('Percentage of data retained after removing excessively long calls: ' + str(np.round(len(stft_2_idx)/len(taxis_data_1),4)))

    stft_data_2 = [stft_data_1[i] for i in stft_2_idx]
    taxis_data_2 = [taxis_data_1[i] for i in stft_2_idx]
    df_sum_2 = df_sum_1.iloc[stft_2_idx].copy()

    max_len = np.max([i.shape[-1] for i in taxis_data_2]) 
else:
    max_len = np.max([i.shape[-1] for i in stft_data_1])
    stft_data_2 = stft_data_1
    taxis_data_2 = taxis_data_1
    df_sum_2 = df_sum_1

## === prepare for umap ===

if logdur:
    spec_rs = [acoustics.log_resize_spec(np.log1p(np.abs(spec)), scaling_factor=8) for spec in stft_data_2]
    stfts_pad = [das_unsupervised.spec_utils.pad_spec(spec, pad_length=max_len) for spec in spec_rs]
    stfts_T = [spec.T for spec in stfts_pad] 
else:
    stfts_pad = [das_unsupervised.spec_utils.pad_spec(spec, pad_length=max_len) for spec in stft_data_2]
    stfts_T = [np.log1p(np.abs(spec)).T for spec in stfts_pad] 

# 2d-->1d vectorize
stfts_flat = [spec.ravel() for spec in stfts_T]
stfts_flat = np.array(stfts_flat)

## --- UMAP --- 
# initialize
print('Computing umap...')

reducer = umap.UMAP(random_state=seed, 
                    n_neighbors=nneighbors, # default 15
                    min_dist=mindist, # default 0.1
                    ) 
embedding = reducer.fit_transform(stfts_flat)

k_centroid, k_labels = scipy.cluster.vq.kmeans2(embedding, k=k_desired, seed=seed)

# save to df
df_sum_2['umap_1'] = embedding[:,0]
df_sum_2['umap_2'] = embedding[:,1]
df_sum_2['k_labels'] = k_labels


# Saving
print('Saving...')

ds_hash = '_'.join(datasets_to_load)
umap_params = '_'.join([str(e) for e in [seed, nneighbors, mindist]])

stft_data_save = {'stft_data':stft_data_2, 'taxis_data': taxis_data_2}
np.save(cache_dir + '/stft_umap/v1_stft_umap_data-'+ds_hash+'-'+umap_params+'.npy', stft_data_save, allow_pickle=True)
df_sum_2.to_csv(cache_dir + '/stft_umap/v1_stft_umap_annotations_data-'+ds_hash+'-'+umap_params+'.csv', index=True, index_label='orig_row_index') #index=False)