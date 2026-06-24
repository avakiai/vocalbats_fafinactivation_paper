"""
utils_npv0.py
"""

from glob import glob
import os
import numpy as np

## Helpers for matching annot_dat rows to audio files
# when running from 1 subfolder down from NPv0/v1/code, i.e. /acoustic
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../v0/data/data'))

# mappings from annot_dat values to filename tokens
TREATMENT_MAP = {'saline': 'pre', 'muscimol': 'post'}
CONDITION_MAP  = {'spontaneous': 'silence'}   # others pass through unchanged


def find_audio_file(anim, treatment, condition, rec_iter, data_dir=DATA_DIR):
    """
    Return the audio (.wav) file path matching a row in annot_dat.

    Parameters
    ----------
    anim      : str   e.g. 'F1', 'M2'
    treatment : str   'saline' or 'muscimol'
    condition : str   'spontaneous' or playback condition (e.g. 'natsyll')
    rec_iter  : int   recording iteration number
    data_dir  : str   root data directory (contains FAF_Deactive_Test* subdirs)

    Returns
    -------
    str or None   — matched file path, or None if not found
    """
    treatment_code = TREATMENT_MAP.get(treatment, treatment)
    condition_code = CONDITION_MAP.get(condition, condition)

    pattern = os.path.join(
        data_dir,
        'FAF_*',
        f'VocalCircuits*{anim}*{treatment_code}*{condition_code}*n{rec_iter}*.wav'
    )
    matches = glob(pattern)
    return matches[0] if matches else None


def get_pre_onset_windows(
    call_vector_01,
    Fs,
    ts_step_s,
    pre_win_s,
    mode=2,
    post_onset_s=0.0,
    ts_len=None,
):
    """
    Return pre-onset window indices into any derived timeseries aligned to call onsets.

    Parameters
    ----------
    call_vector_01 : 1-D int array
        Binary call vector at LFP rate (1 = vocalization, 0 = silence).
    Fs : float
        LFP sampling rate (Hz).
    ts_step_s : float
        Step size of the target timeseries in seconds.
        Use 1/Fs to index directly into the LFP; use the windowing step_s for
        power/xcorr timeseries.
    pre_win_s : list of (float, float)
        (start_s, stop_s) pairs in seconds before onset, stop_s > start_s.
        (0.0, 0.5) → 0–0.5 s before onset; (1.0, 2.0) → 1–2 s before onset.
    mode : int
        1 = all onsets (no filtering)
        2 = per-window quiet filter: no vocal activity within each window's lookback
        3 = max-lookback filter: filter once using the longest window, apply to all
    post_onset_s : float
        Extend each window past the onset by this many seconds (0 = end at onset).
    ts_len : int, optional
        Length of the target timeseries; windows whose end exceeds it are dropped.

    Returns
    -------
    dict
        {(start_s, stop_s): list of np.ndarray}
        Each array holds indices into the target timeseries for one onset.
    """
    ts_Fs = 1.0 / ts_step_s
    post_samp = int(post_onset_s * ts_Fs)

    onsets_lfp  = np.where(np.diff(call_vector_01) ==  1)[0] + 1
    offsets_lfp = np.where(np.diff(call_vector_01) == -1)[0] + 1
    onsets  = (onsets_lfp  / Fs * ts_Fs).astype(int)
    offsets = (offsets_lfp / Fs * ts_Fs).astype(int)

    def _is_quiet(onset, lookback):
        return (
            not np.any((onsets  > onset - lookback) & (onsets  < onset)) and
            not np.any((offsets > onset - lookback) & (offsets < onset))
        )

    if mode == 3:
        max_lookback = int(max(stop_s for _, stop_s in pre_win_s) * ts_Fs)
        clean_all = np.array([o for o in onsets if _is_quiet(o, max_lookback)])

    windows = {}
    for start_s, stop_s in pre_win_s:
        start_samp = int(start_s * ts_Fs)
        stop_samp  = int(stop_s  * ts_Fs)

        if mode == 1:
            use_onsets = onsets
        elif mode == 2:
            use_onsets = np.array([o for o in onsets if _is_quiet(o, stop_samp)])
        else:
            use_onsets = clean_all

        idxs = []
        for onset in use_onsets:
            lo = max(0, onset - stop_samp)
            hi = onset - start_samp + post_samp + 1
            if ts_len is not None and hi > ts_len:
                continue
            idxs.append(np.arange(lo, hi))

        windows[(start_s, stop_s)] = idxs

    return windows, use_onsets


# loading in processed data
def load_cache_data(data_dir):
    import pickle
    data = {}
    for file in os.listdir(data_dir):
        key = file[:-4]
        if file.endswith('.npy'):
            ext = '.npy'
        elif file.endswith('.pkl'):
            ext = '.pkl'
        anim = key.split('_')[0]
        dkey = '_'.join(key.split('_')[1:])
        if anim not in data:
            data[anim] = {}
        if ext == '.npy':
            data[anim][dkey] = np.load(os.path.join(data_dir, file), allow_pickle=True).item()
        else:
            with open(os.path.join(data_dir, file), 'rb') as f:
                data[anim][dkey] = pickle.load(f)
    return data