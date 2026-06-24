"""
npv0_acoustics.py — acoustic analysis helpers for Npv0
"""

import numpy as np
from scipy import io


def get_powerof2(x):
    return 1 << x.bit_length()-1

def find_baseline_window(file_df, min_dur=10.0,
                         onset_col='start_seconds', offset_col='stop_seconds',
                         rec_dur_col='rec_dur'):
    """
    Find the first call-free window of at least min_dur seconds in a recording.

    Parameters
    ----------
    file_df     : DataFrame   rows from annot_dat for a single recording file
    min_dur     : float       minimum required quiet duration in seconds (default 10)
    onset_col   : str         column with call start times (seconds)
    offset_col  : str         column with call stop times (seconds)
    rec_dur_col : str         column with total recording duration (seconds)

    Returns
    -------
    (t_start, t_end) : tuple of floats, or None if no window found
    """
    rec_dur = float(file_df[rec_dur_col].iloc[0])

    if file_df.empty:
        return (0.0, min_dur) if rec_dur >= min_dur else None

    onsets  = np.sort(file_df[onset_col].values.astype(float))
    offsets = np.sort(file_df[offset_col].values.astype(float))

    # gap before first call
    if onsets[0] >= min_dur:
        return (0.0, min_dur)

    # gaps between consecutive calls
    for i in range(len(offsets) - 1):
        gap_start = offsets[i]
        gap_end   = onsets[i + 1]
        if gap_end - gap_start >= min_dur:
            return (gap_start, gap_start + min_dur)

    # gap after last call
    if rec_dur - offsets[-1] >= min_dur:
        return (offsets[-1], offsets[-1] + min_dur)

    return None

def get_audio(filepath, channel=-1):
    """
    Parameters
    ----------
    filepath : str        
    channel : int, optional
        for wav files

    Returns
    -------
    audio
    sampling rate

    """


    if filepath.split('.')[-1]=='npz':
        audio_data = np.load(filepath, mmap_mode='r')['data']    
        sr = np.load(filepath, mmap_mode='r')['samplerate'].astype(int)[0]
    
    if filepath.split('.')[-1]=='wav':
        audio_data_nd = io.wavfile.read(filepath, mmap=True)[1]
        audio_data = audio_data_nd[:,channel]
        sr = io.wavfile.read(filepath, mmap=True)[0]

    return (audio_data, sr)