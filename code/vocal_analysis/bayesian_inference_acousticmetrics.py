"""
call_params_inference.py
Bayesian inference of gamma distribution parameters for call acoustic metrics.

For each data slice defined in SLICES, fits a Gamma(a, b) distribution to
calls from each treatment independently, then plots:
  1. 2-D posterior P(a, b | data)  — one panel per treatment
  2. Marginal posteriors P(a | data) and P(b | data) — treatments overlaid
  3. HPD credible region summary across slices

Usage
-----
Run directly:
    python call_params_inference.py

Or import results_dict from a notebook:
    import importlib
    import call_params_inference as cpi
    importlib.reload(cpi)
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append('../../') # rel. dir. to find toolbox 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm

from bayesbox import inference, plot as bplot
from _frontal_vocal_paper_repo.code.vocal_analysis.npxtoolbox import plot

# =============================================================================
# CONFIG — edit these for each analysis
# =============================================================================

DATA_PATH = '/mnt/hpc/projects/BWFAFdeactivNpx/Npv1_data/Npv1_cachedata/v1_combined_analysis_dataset.csv'

# METRIC      = 'duration'     
# SCALE       = 1e3      
# METRIC_TITLE  = 'duration' 
# log = False
# pdist = 'gamma'

# METRIC      = 'rms'     
# SCALE       = 1    
# METRIC_TITLE  = 'norm. rms'
# log = True
# pdist = 'gamma'

# METRIC      = 'centroid_med'     
# SCALE       = 1e-3 
# METRIC_TITLE  = 'median centroid'
# log = False
# pdist = 'gaussian'

METRIC      = 'spec_peak'     
SCALE       = 1e-3      
METRIC_TITLE  = 'spec peak' 
log = True
pdist = 'gaussian'

print('Parameter inference for metric:', METRIC)

metric_lab_map = {'duration': 'duration [ms]',
                  'rms':'norm. rms [$V^{2}$]', 
                  'spec_peak': 'peak freq. [kHz]',
                  'center_f_peak': 'est. peak freq. [kHz]',
                  'f0_mean': 'mean f0 [kHz]',
                  'centroid_med': 'median centroid [kHz]',
                  'n_peaks': 'num. est. peaks',
                  'total_bw': 'sum peak bandwidth [Hz]',
                  'auc': 'auc of est. psd [a.u.]'
                  }

param_lim_default = False

CREDIBILITY = 0.95           # HPD credible region probability content

TREATMENT_COL = 'treatment'
TREATMENTS    = ['saline', 'muscimol']

TREATMENT_PAL = {'saline': plt.cm.viridis(np.linspace(0.4,0.6,1)), 
                 'muscimol': plt.cm.magma(np.linspace(0.4,0.6,1))}
line_pal = {'saline': "#235E6E", 
            'muscimol': "#75276C"}

# Data slices: each key names one analysis group.
# Filters are applied as column == value; add/remove entries as needed.
SLICES = {
    'spontaneous':   {'sess_type': 'spontaneous'},
    'playback':      {'sess_type': 'playback'},
    'playback_comm': {'sess_type': 'playback', 'call_type': 'comm'},
    'playback_echo': {'sess_type': 'playback', 'call_type': 'echo'},
    'spontaneous_comm': {'sess_type': 'spontaneous', 'call_type': 'comm'},
    'spontaneous_echo': {'sess_type': 'spontaneous', 'call_type': 'echo'},
}

slice_label_map = dict(zip(SLICES.keys(), ['spontaneous','playback','non-echo, playback','echo, playback','non-echo, spontaneous', 'echo, spontaneous']))
call_type_map = {'comm':'non-echo','echo':'echo'}

# SAVEDIR = '/mnt/hpc/projects/BWFAFdeactivNpx/results/voc_bayes'+'/'+METRIC   # set to a directory path to save all figures, e.g. './results/'
SAVEDIR = '/mnt/hpc/projects/BWFAFdeactivNpx/results/voc_bayes'+'/'+METRIC+'/'+pdist
os.makedirs(SAVEDIR, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(DATA_PATH)
print(f'Loaded {len(df)} rows from {os.path.basename(DATA_PATH)}')


# =============================================================================
# RUN INFERENCE
# =============================================================================

results = {}    # results[slice_name][treatment] = dict from inference.run_inference

for slice_name, filters in SLICES.items():
    results[slice_name] = {}

    for treatment in TREATMENTS:
        # build boolean mask
        mask = df[TREATMENT_COL] == treatment
        for col, val in filters.items():
            mask &= df[col] == val

        data = df.loc[mask, METRIC].dropna().values * SCALE
        
        # log10
        if log:
            print('Inference on log10 transformed data...')
            data = np.log10(data)

        if len(data) == 0:
            print(f'  [WARN] {slice_name}/{treatment}: no data — skipping')
            continue

        print(f'\n{slice_name} / {treatment}  (n={len(data)})')
        # Parameter grid for inference
## duration, linear, scale=1e3
        if METRIC == 'duration':
            if 'echo' in slice_name:
                print('  [NOTE] calls have different values for this metric, so we are using a different parameter grid for inference...')
                A_VALUES = np.arange(0.1, 10.01, 0.01)   # shape parameter
                B_VALUES = np.arange(0.1, 10.01, 0.001)   # scale parameter
            else:
                A_VALUES = np.arange(0.1, 10.01, 0.01)   # shape parameter
                B_VALUES = np.arange(0.1, 10.01, 0.01)   # scale parameter

## rms, log10, scale=1
        elif METRIC == 'rms':
            if 'echo' in slice_name:
                print('  [NOTE] calls have different values for this metric, so we are using a different parameter grid for inference...')
                A_VALUES = np.arange(0.1, 100.01, 0.01)   # shape parameter
                B_VALUES = np.arange(0.001, 0.21, 0.0001)   # scale parameter
            elif 'comm' in slice_name:
                print('  [NOTE] calls have different values for this metric, so we are using a different parameter grid for inference...')
                A_VALUES = np.arange(0.1, 10.01, 0.01)   # shape parameter
                B_VALUES = np.arange(0.1, 1.01, 0.001)   # scale parameter
            else:
                A_VALUES = np.arange(0.1, 10.01, 0.01)   # shape parameter
                B_VALUES = np.arange(0.1, 2.01, 0.01)   # scale parameter

## centroid, linear, scale=1e-3
        elif METRIC == 'centroid_med':
            if pdist == 'gamma':
                if 'echo' in slice_name:
                    print('  [NOTE] echo calls have very different values for this metric, so we are using a different parameter grid for inference...')
                    A_VALUES = np.arange(950.1, 3000.01, 0.01)   # shape parameter
                    B_VALUES = np.arange(0.001, 0.11, 0.00005)   # scale parameter
                else:
                    A_VALUES = np.arange(0.1, 100.01, 0.01)   # shape parameter
                    B_VALUES = np.arange(0.1, 3.01, 0.001)   # scale parameter
            elif pdist == 'gaussian':
                A_VALUES = np.arange(40.0, 100.01, 0.01)   # mean parameter
                B_VALUES = np.arange(0.01, 50.01, 0.01)   # standard deviation parameter
## spec_peak, linear, scale=1e-3
        elif METRIC == 'spec_peak':
            if pdist == 'gamma':
                if 'echo' in slice_name:
                    print('  [NOTE] calls have different values for this metric, so we are using a different parameter grid for inference...')
                    A_VALUES = np.arange(0.1, 500.01, 0.01)   # shape parameter
                    B_VALUES = np.arange(0.1, 2.01, 0.001)   # scale parameter
                else:
                    A_VALUES = np.arange(0.1, 10.01, 0.01)   # shape parameter
                    B_VALUES = np.arange(0.1, 25.01, 0.01)   # scale parameter
            elif pdist == 'gaussian':
                A_VALUES = np.arange(0.01, 4.01, 0.001)   # mean parameter
                B_VALUES = np.arange(0.01, 2.01, 0.0001)   # standard deviation parameter
                
        res = inference.run_inference(data, A_VALUES, B_VALUES, CREDIBILITY, dist=pdist)
        results[slice_name][treatment] = res

        print(f'  P({CREDIBILITY}) for a: [{res["bounds_a"][0]:.3f}, {res["bounds_a"][1]:.3f}]')
        print(f'  P({CREDIBILITY}) for b: [{res["bounds_b"][0]:.3f}, {res["bounds_b"][1]:.3f}]')
        
np.save(os.path.join(SAVEDIR, 'inference_results.npy'), results, allow_pickle=True)

# is save object too large?
# import pickle
# with open(os.path.join(SAVEDIR, 'inference_results.pkl'), 'wb') as f:
#     pickle.dump(results, f, protocol=4)

# =============================================================================
# PLOT
# =============================================================================

plt.style.use('default')

for slice_name, slice_results in results.items():
    if not slice_results:
        continue
    
    if param_lim_default:
        a_lim = None; b_lim = None
    else:
# ## duration 
#         if 'echo' in slice_name or 'comm' in slice_name:
#             a_lim = [1,8]; b_lim = [0,2]
#         else:
#             a_lim = [1,2]; b_lim = [0.5,2] 

## RMS
        # if 'echo' in slice_name:
        #     a_lim = [1,100]; b_lim = [0,0.2] 
        # else:
        #     a_lim = [1.5,6]; b_lim = [0,1] 

## centroid
        if METRIC == 'centroid_med':
            if pdist == 'gamma':
                if 'echo' in slice_name:
                    a_lim, b_lim = bplot.auto_lims(slice_results, nice=True) 
                else:
                    a_lim = [20,70]; b_lim = [0,3] 
            elif pdist == 'gaussian':
                a_lim, b_lim = bplot.auto_lims(slice_results, nice=True)
## spec_peak 
        elif METRIC == 'spec_peak':
            a_lim, b_lim = bplot.auto_lims(slice_results, nice=True) 

    label = f'{metric_lab_map[METRIC]} | {slice_label_map[slice_name]}'

    # --- 2-D posterior (one panel per treatment) ---
    fig_2d, _ = bplot.posterior_2d_grid(
        slice_results,
        # a_values=A_VALUES, b_values=B_VALUES,
        a_values=list(slice_results.values())[0]['a_values'],
        b_values=list(slice_results.values())[0]['b_values'],
        treatment_order=TREATMENTS,
        title=label,
        cmap='BuPu',
        a_lim=a_lim, b_lim=b_lim,
        savedir=SAVEDIR,
        filename=f'posterior_2d_{slice_name}.png',
    )

    # --- Marginal posteriors (treatments overlaid) ---
    fig_m, _ = bplot.marginal_posteriors(
        # A_VALUES, B_VALUES,
        list(slice_results.values())[0]['a_values'],
        list(slice_results.values())[0]['b_values'],
        slice_results,
        treatment_palette=TREATMENT_PAL,
        credibility=CREDIBILITY,
        treatment_order=TREATMENTS,
        title=label, legend_bbox=(1.15, 1.0),
        a_lim=a_lim, b_lim=b_lim,
        savedir=SAVEDIR,
        filename=f'marginals_{slice_name}.png',
    )


# --- Credible region summary across all slices ---
rows = []
for slice_name, slice_results in results.items():
    for treatment, res in slice_results.items():
        for param, bounds_key, mode_key in [('a', 'bounds_a', 'mode_a'), ('b', 'bounds_b', 'mode_b')]:
            b = res[bounds_key]
            rows.append(dict(
                slice=slice_name,
                treatment=treatment,
                param=param,
                lower=b[0],
                upper=b[1],
                center=res[mode_key],
            ))

summary_df = pd.DataFrame(rows)

for param in ['a', 'b']:
    fig_s, ax_s = bplot.credible_region_summary(
        summary_df[summary_df['param'] == param],
        param=param, ylabel_map=slice_label_map,
        treatment_palette=TREATMENT_PAL, dist=pdist,
        figsize=(5,4),
        # savedir=SAVEDIR,
        # filename=f'credible_summary_{param}.png',
    )
    ax_s.invert_yaxis()
    fig_s.savefig(os.path.join(SAVEDIR, f'credible_summary_{param}.png'), bbox_inches='tight', transparent=True) 

# --- Overlay pdf on data for all slices ---
if log:
    bs = 0.1
else:
    bs = 1
fig, axes = plot.hist_hue(df, x=METRIC, 
                        hue_palette=TREATMENT_PAL,
                        xlabel=metric_lab_map[METRIC], scale=SCALE, 
                        log=log, binsize=bs,                        
                        split_by='sess_type', legend_bbox=(1.2,0.9),
                        title=METRIC_TITLE,
                        density=True, sharey=True, 
                        )
for i, sess_type in enumerate(['playback','spontaneous']):
    slice_results = results[sess_type]

    for treat in TREATMENTS:
        d = np.array(df.loc[(df['treatment']==treat) & 
                            (df['sess_type']==sess_type)
                            ][METRIC]*
                            SCALE)
        if log:
            d = np.log10(d)
        x = np.linspace(d.min(), d.max(), len(d))
        if pdist=='gaussian':
            f = norm(loc=slice_results[treat]['mode_a'], scale=slice_results[treat]['mode_b'])
        elif pdist=='gamma':
            f = gamma(a=slice_results[treat]['mode_a'], scale=slice_results[treat]['mode_b'])
        else:
            f = gamma(a=slice_results[treat]['mode_a'], scale=slice_results[treat]['mode_b'])

        pdf = f.pdf(x)
        axes[i].plot(x, pdf, c=line_pal[treat], linestyle='solid', alpha=0.8, linewidth=2, label='gamma pdf '+treat)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc='upper left', bbox_to_anchor=(0.94,0.73))
fig.tight_layout()
fig.savefig(SAVEDIR+'/_'+METRIC+'_linear_overlay_pdf.png',
            transparent=True, bbox_inches='tight')


for i, sess_type in enumerate(['playback', 'spontaneous']):

    df_ = df.loc[df['sess_type']==sess_type]
    fig, axes = plot.hist_hue(df_, x=METRIC,
                            hue_palette=TREATMENT_PAL,
                            xlabel=metric_lab_map[METRIC], scale=SCALE,
                            log=log, binsize=bs,
                            split_by='call_type', legend_bbox=(1.2, 0.9),
                            title=f'{METRIC_TITLE} | {sess_type}',
                            density=True, sharey=True)        

    for i, call_type in enumerate(['comm','echo']):
        slice_name = sess_type+'_'+call_type
        slice_results = results[slice_name]

        for treat in TREATMENTS:

            d = df.loc[(df['treatment'] == treat) & 
                        (df['sess_type'] == sess_type) &
                        (df['call_type'] == call_type),
                        METRIC].dropna().values * SCALE
            if log:
                d = np.log10(d)
            x = np.linspace(d.min(), d.max(), 300)
            if pdist=='gaussian':
                f = norm(loc=slice_results[treat]['mode_a'], scale=slice_results[treat]['mode_b'])
            elif pdist=='gamma':
                f = gamma(a=slice_results[treat]['mode_a'], scale=slice_results[treat]['mode_b'])
            else:
                f = gamma(a=slice_results[treat]['mode_a'], scale=slice_results[treat]['mode_b'])

            axes[i].plot(x, f.pdf(x), c=line_pal[treat],
                            linestyle='solid', alpha=0.8, linewidth=2, label='gamma pdf ' + treat)
            axes[i].set_title(call_type_map[call_type])
            
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper left', bbox_to_anchor=(0.94, 0.73))
    fig.tight_layout()
    fig.savefig(os.path.join(SAVEDIR, f'_{METRIC}_{slice_name}_linear_overlay_pdf.png'),
                transparent=True, bbox_inches='tight')


# --- Plot generated pdfs from MAP across all slices ---
fig, ax = plt.subplots(1,2, figsize=(2*1.5,2), sharey=True)

for i, sess_type in enumerate(['playback','spontaneous']):
    for treat in TREATMENTS:
        d = np.array(df.loc[(df['treatment']==treat) & 
                            (df['sess_type']==sess_type)
                            ][METRIC]*
                            SCALE)
        if log:
            d = np.log10(d)
        x = np.linspace(d.min(), d.max(), len(d))
        if pdist=='gaussian':
            f = norm(loc=results[sess_type][treat]['mode_a'], scale=results[sess_type][treat]['mode_b'])
        elif pdist=='gamma':
            f = gamma(a=results[sess_type][treat]['mode_a'], scale=results[sess_type][treat]['mode_b'])
        else:
            f = gamma(a=results[sess_type][treat]['mode_a'], scale=results[sess_type][treat]['mode_b'])
    
        pdf = f.pdf(x)
        ax[i].plot(x, pdf, c=line_pal[treat], linestyle='solid', alpha=0.8, linewidth=2, label='gamma pdf '+treat)
        ax[i].set_title(sess_type)
        ax[i].spines[['right', 'top']].set_visible(False)
        ax[i].set_ylabel('density')
        ax[i].set_xlabel(metric_lab_map[METRIC])

ax[i].label_outer()
fig.suptitle(METRIC_TITLE)
fig.legend(handles, labels, frameon=False, loc='upper left', bbox_to_anchor=(1,0.95))
fig.tight_layout()
fig.savefig(SAVEDIR+'/_'+METRIC+'_pdf.png',
            transparent=True, bbox_inches='tight')

for i, sess_type in enumerate(['playback', 'spontaneous']):    
    fig, ax = plt.subplots(1,2, figsize=(2*1.5,2), sharey=True)   
    for i, call_type in enumerate(['comm','echo']):
        slice_name = sess_type+'_'+call_type
        for treat in TREATMENTS:

            d = df.loc[(df['treatment'] == treat) & 
                        (df['sess_type'] == sess_type) &
                        (df['call_type'] == call_type),
                        METRIC].dropna().values * SCALE
            if log:
                d = np.log10(d)
            x = np.linspace(d.min(), d.max(), len(d))
            if pdist=='gaussian':
                f = norm(loc=results[slice_name][treat]['mode_a'], scale=results[slice_name][treat]['mode_b'])
            elif pdist=='gamma':
                f = gamma(a=results[slice_name][treat]['mode_a'], scale=results[slice_name][treat]['mode_b'])
            else:
                f = gamma(a=results[slice_name][treat]['mode_a'], scale=results[slice_name][treat]['mode_b'])
            
            pdf = f.pdf(x)
            ax[i].plot(x, pdf, c=line_pal[treat], linestyle='solid', alpha=0.8, linewidth=2, label='gamma pdf '+treat)
            ax[i].set_title(call_type_map[call_type])
            ax[i].spines[['right', 'top']].set_visible(False)
            ax[i].set_ylabel('density')
            ax[i].set_xlabel(metric_lab_map[METRIC])

    ax[i].label_outer()
    fig.suptitle(f'{METRIC_TITLE} | {sess_type}')
    # handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper left', bbox_to_anchor=(1,0.95))
    fig.tight_layout()
    fig.savefig(os.path.join(SAVEDIR, f'_{METRIC}_{slice_name}_pdf.png'),
                transparent=True, bbox_inches='tight')