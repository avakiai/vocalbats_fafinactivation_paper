import os, sys, pickle
sys.path.append('/cs/projects/BWFAFdeactivNpx/Npv0/v1/code')
from toolbox import npv0_lfp as lfp, npv0_cfc as cfc, npv0_power as power
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))

# input
neural_data_dir = '/cs/projects/BWFAFdeactivNpx/Npv0/v0/data/data'
metadata_dir = '/cs/projects/BWFAFdeactivNpx/Npv0/v0/data/exp_metadata'
# output
cache_dir = '/cs/projects/BWFAFdeactivNpx/Npv0/v1/data/cache_data/LFP_parallel'

## params
CHANNEL = 0
# For phase/amplitude:
phase_bands = cfc.make_phase_bands(start=1, stop=8, step=1, width=2)
amp_bands   = cfc.make_amp_bands(start=15, stop=120, step=5, width=10)
# For windowed power spectrum:
win_s = 2
step_s = 0.01
nfft = 8192


## run
# load
annot_data = pd.read_csv('/cs/projects/BWFAFdeactivNpx/Npv0/v1/data/cache_data/call_data/v0_spontaneous_analysis_dataset.csv')

root, dirs, files = next(os.walk(neural_data_dir))
dirs = list(np.sort([folders for folders in dirs if 'Test' in folders if 'Test1' not in folders]))

def _extract_one_band(lfp_1d, lo, hi, Fs, kind, order=4):
    import scipy.signal as sg
    nyq = 0.5 * Fs
    sos = sg.butter(order, [lo / nyq, hi / nyq], btype='band', output='sos')
    analytic = sg.hilbert(sg.sosfiltfilt(sos, lfp_1d))
    return np.angle(analytic) if kind == 'phase' else np.abs(analytic)


def _compute_mi_grid(periods, phase_dict, amp_dict):
    task_keys = [(label, fpha, fAmp)
                 for label in periods
                 for fpha in phase_dict
                 for fAmp in amp_dict]
    results = Parallel(n_jobs=N_JOBS)(
        delayed(cfc.tort_mi)(phase_dict[fpha][periods[label]], amp_dict[fAmp][periods[label]], n_bins=18)
        for label, fpha, fAmp in task_keys
    )
    MI_mat = {}
    for (label, fpha, fAmp), (mi, p) in zip(task_keys, results):
        MI_mat.setdefault(label, {}).setdefault(fpha, {})[fAmp] = {'mi': mi, 'p': p}
    return MI_mat


for dir in dirs:
    folder, _, files = next(os.walk(root+'/'+dir)) # dirs[1]
    # find files
    h5_files = [f'{folder}/{f}' for f in files if 'silence' in f and '.h5' in f] # neural data
    annot_files = [f'{folder}/{f}' for f in files if 'silence' in f and '.csv' in f] # annotation data
    probe_file = [f'{folder}/{f}' for f in files if '.prb' in f][0]

    # initialize - load lfp and call data for control/muscimol for this anim.
    call_data = {'control':None, 'muscimol':None}
    lfp_data = {'control':None, 'muscimol':None}

    for f in h5_files:
        if 'pre_' in f:
            treat = 'control'
        elif 'post_' in f:
            treat = 'muscimol'

        print(f)
        print('\t' + treat)
        print("\n")

        annot_data_f = (annot_data.loc[(annot_data['anim']==f.rsplit('/')[-1].rsplit('_')[2]) & (annot_data['treatment']==treat)]).copy()
        if len(annot_data_f)==0 or len(annot_data_f)==1:
            continue # skip to next | note: F2/post (no pre) has data here that could be useful (N=209), M2/post also has data

        ## --= 1. get neural data ---
        channel_lfp, meta = lfp.prepare_lfp(f, probe_file)
        Fs = meta['Fs']
        _, trigs_sp, _ = lfp.get_triggers(f)

        print(f"\t{meta['n_bad_channels']} bad channels, {meta['duration_s']:.1f}s")

        lfp_data[treat] = {'lfp':channel_lfp, 'trigs_sp':trigs_sp}

        ## --- 2. get call data --
        af = [af for af in annot_files if f.rsplit('/')[-1].rsplit('_')[4] in af][0]
        annot_data_local = pd.read_csv(af)
        annot_data_f["start_seconds_neuralsamples"] = np.floor((annot_data_f.start_seconds - annot_data_local.query('name == "trigger"').start_seconds[0])*Fs)+trigs_sp[0]
        annot_data_f["stop_seconds_neuralsamples"] = np.floor((annot_data_f.stop_seconds - annot_data_local.query('name == "trigger"').start_seconds[0])*Fs)+trigs_sp[0]

        call_data[treat] = {'df': annot_data_f}

    if f.rsplit('/')[-1].rsplit('_')[2] not in ['M3','M4','F3','F4']:
        continue

    # z-score lfp
    print('\n\tZ-normalizing lfp data between treatments...')
    vec_lfp = np.concatenate([lfp_data[treat]['lfp'] for treat in lfp_data.keys()], axis=1)
    chan_mean = vec_lfp.mean(axis=1, keepdims=True); chan_std  = vec_lfp.std(axis=1, keepdims=True)
    for treat in lfp_data:
        lfp_data[treat]['lfp_z'] = (lfp_data[treat]['lfp'] - chan_mean) / chan_std

    # call start times --> call vector with len = lfp len
    print('\n\tCreating call vectors...')
    for treat in lfp_data.keys():
        print('\t'+treat)
        call_vector_01 = np.zeros(len(lfp_data[treat]['lfp'][0,:]),dtype=int)
        for c in call_data[treat]['df'].itertuples():
            call_vector_01[int(c.start_seconds_neuralsamples):int(c.stop_seconds_neuralsamples)] = 1
        idx_vocal  = np.where(call_vector_01 == 1)[0]
        idx_nonvocal = np.where(call_vector_01 == 0)[0]

        call_data[treat].update({'call_vector_01':call_vector_01,'idx_voc':idx_vocal, 'idx_nonvoc':idx_nonvocal})

        print('\tNumber of vocal samples: ' + str(len(idx_vocal)))
        print('\tNumber of non-vocal samples: '+ str(len(idx_nonvocal)))

    ## compute phase/amplitude vectors — all bands × treatments in parallel
    print('\n\tCreating phase/amplitude vectors...')
    pa_tasks = [(treat, 'phase', phase_bands[0,i], phase_bands[1,i])
                for treat in lfp_data.keys() for i in range(phase_bands.shape[1])] + \
               [(treat, 'amp', amp_bands[0,i], amp_bands[1,i])
                for treat in lfp_data.keys() for i in range(amp_bands.shape[1])]
    pa_results = Parallel(n_jobs=N_JOBS)(
        delayed(_extract_one_band)(lfp_data[treat]['lfp_z'][CHANNEL,:], lo, hi, Fs, kind)
        for treat, kind, lo, hi in pa_tasks
    )
    for (treat, kind, lo, _), arr in zip(pa_tasks, pa_results):
        lfp_data[treat].setdefault('phase_dict' if kind == 'phase' else 'amp_dict', {})[lo] = arr
    for treat in lfp_data.keys():
        lfp_data[treat].update({'params':{'cfc':{'channel':CHANNEL}}})

    ## compute power spectrum (windowed) — both treatments in parallel
    # (for time-resolved power; autocorrelating power in specific bands)
    print('\n\tComputing time-resolved power spectrum...')
    spec_results = Parallel(n_jobs=min(2, N_JOBS))(
        delayed(power.welch_windowed)(lfp_data[treat]['lfp_z'][CHANNEL,:], Fs,
                                      nfft=nfft, nperseg=nfft, win_len_s=win_s,
                                      win_step_s=step_s, scaling='spectrum')
        for treat in lfp_data.keys()
    )
    for treat, (freq_ax, welch_spec_ts) in zip(lfp_data.keys(), spec_results):
        lfp_data[treat].update({'spectrum':{'welch_spec_ts':welch_spec_ts, 'freq_ax':freq_ax}})
        lfp_data[treat].update({'params':{'spectrum':{'win_s':win_s, 'step_s':step_s, 'nfft':nfft, 'channel':CHANNEL}}})

## Compute CFC mutual information
### --- 1. MI over all phase/amp pairs (all samples) ---

    for treat in lfp_data.keys():
        print('\n\tComputing CFC MI for all samples in '+treat+'...')
        MI_mat = cfc.compute_cfc_tort(lfp_data[treat]['phase_dict'], lfp_data[treat]['amp_dict'], n_bins=18)
        n_samples = len(lfp_data[treat]['lfp_z'][CHANNEL,:])
        lfp_data[treat].update({'all': {'MI_mat':MI_mat, 'n':n_samples}})

### -- 2. MI for vocal vs non-vocal periods (equalize N within treatment) ---
        print('\n\tComputing CFC MI for vocal vs non-vocal periods (equal N) in '+treat+'...')
        periods = dict(zip(['vocal',
                            'non-vocal'],
                        [call_data[treat]['idx_voc'],
                        np.random.choice(call_data[treat]['idx_nonvoc'],len(call_data[treat]['idx_voc']), replace=False)
                        ]))

        phase_dict = lfp_data[treat]['phase_dict']; amp_dict = lfp_data[treat]['amp_dict']
        MI_mat = _compute_mi_grid(periods, phase_dict, amp_dict)
        lfp_data[treat].update({'within': {'MI_mat':MI_mat,'n':len(list(periods.values())[0])}})

### -- 3. MI for vocal, non-vocal periods between treatments (equalize N between periods/treatments) ---
    print('\n\tComputing CFC MI for vocal and non-vocal periods between treatments...')
    min_len = {
        'vocal':min(len(call_data['control']['idx_voc']),len(call_data['muscimol']['idx_voc'])),
        'non-vocal':min(len(call_data['control']['idx_nonvoc']),len(call_data['muscimol']['idx_nonvoc']))
        }

    for treat in lfp_data.keys():
        # equalize within treatment
        periods = dict(zip(['vocal','non-vocal'],
                       [np.random.choice(call_data[treat]['idx_voc'], min_len['vocal'], replace=False),
                        np.random.choice(call_data[treat]['idx_nonvoc'], min_len['non-vocal'], replace=False)
                        ]))

        phase_dict = lfp_data[treat]['phase_dict']; amp_dict = lfp_data[treat]['amp_dict']
        MI_mat = _compute_mi_grid(periods, phase_dict, amp_dict)
        lfp_data[treat].update({'across':{'MI_mat':MI_mat,'n':min_len}})


    # np.save(cache_dir + '/' + f.rsplit('/')[-1].rsplit('_')[2] + '_lfp_data.npy', lfp_data)
    # np.save(cache_dir + '/' + f.rsplit('/')[-1].rsplit('_')[2] + '_call_data.npy', call_data)
    anim = f.rsplit('/')[-1].rsplit('_')[2]
    with open(cache_dir + '/' + anim + '_lfp_data.pkl', 'wb') as fh:
        pickle.dump(lfp_data, fh, protocol=4)
    with open(cache_dir + '/' + anim + '_call_data.pkl', 'wb') as fh:
        pickle.dump(call_data, fh, protocol=4)
