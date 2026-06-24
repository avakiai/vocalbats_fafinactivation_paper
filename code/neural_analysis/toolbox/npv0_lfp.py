"""
npv0_lfp.py — Neural data I/O and LFP preparation for NPv0 recordings.

Wraps McsPy loading, probe-based channel sorting, bad-channel masking,
and bandpass filtering into a clean prepare_lfp() pipeline.

Module design rationale
-----------------------
npv0_preprocess.py  — kept for the IBL bad-channel QC algorithm (complex, borrowed).
npv0_utils.py       — kept for file-finding helpers and sort_channels.
npv0_lfp.py (this)  — orchestrates I/O: load → sort → clean → filter.
                       Import this in notebooks instead of calling McsPy directly.
"""

import numpy as np
from scipy import signal as sg


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_mcs_recording(h5_path, probe_file):
    """
    Load a McsPy .h5 file and return depth-sorted channel data.

    Parameters
    ----------
    h5_path : str
    probe_file : str
        .prb file; columns 0-2 are (software_id, depth_rank, ...).

    Returns
    -------
    channel_data : ndarray, shape (n_channels, n_samples), float
    channel_ids  : ndarray of int
    Fs           : float   sampling frequency in Hz
    duration_s   : float
    units        : str     voltage unit string from McsPy
    """
    import McsPy.McsData
    data = McsPy.McsData.RawData(h5_path)
    rec = data.recordings[0]
    stream = data.recordings[0].analog_streams[0]
    channel_ids = np.array(list(stream.channel_infos.keys()))

    Fs = stream.channel_infos[channel_ids[0]].sampling_frequency.magnitude
    duration_s = rec.duration / 1e6
    units = stream.channel_infos[channel_ids[0]].info['Unit']

    channel_data = np.array(stream.channel_data).astype(float)
    channel_mapping = np.loadtxt(probe_file, skiprows=1, delimiter=',',
                                 usecols=(0, 1, 2), dtype=int)
    channel_data, channel_ids = sort_channels(channel_data, channel_mapping[:, 1])

    return channel_data, channel_ids, Fs, duration_s, units


def get_triggers(h5_path):
    """
    Extract trigger timestamps from the event stream of a McsPy recording.

    Returns
    -------
    trig_s  : ndarray  trigger times in seconds
    trig_sp : ndarray  trigger times in samples (relative to recording start)
    Fs      : float
    """
    import McsPy.McsData
    data = McsPy.McsData.RawData(h5_path)
    rec = data.recordings[0]
    stream = data.recordings[0].analog_streams[0]
    channel_ids = np.array(list(stream.channel_infos.keys()))
    Fs = stream.channel_infos[channel_ids[0]].sampling_frequency.magnitude

    trig_ts = rec.event_streams[0].event_entity[0].get_event_timestamps()[0]
    trig_s = trig_ts / 1e6
    trig_sp = trig_s * Fs
    return trig_s, trig_sp, Fs

# ----------------------------------------------------------------------------------------------
# IBL Detect Bad Channels
# ----------------------------------------------------------------------------------------------
def detect_bad_channels_ibl(
    raw,
    fs,
    psd_hf_threshold=0.02,
    dead_channel_thr=-0.5,
    noisy_channel_thr=1.0,
    outside_channel_thr=-0.75,
    n_neighbors=11,
    nyquist_threshold=0.8,
    welch_window_ms=0.3,
):
    """
    Bad channels detection for Neuropixel probes developed by IBL

    Parameters
    ----------
    raw : traces
        (num_samples, n_channels) raw traces
    fs : float
        sampling frequency
    psd_hf_threshold : float
        Threshold for high frequency PSD. If mean PSD above `nyquist_threshold` * fn is greater than this
        value, channels are flagged as noisy (together with channel coherence condition).
    dead_channel_thr : float, optional
        Threshold for channel coherence below which channels are labeled as dead, by default -0.5
    noisy_channel_thr : float
        Threshold for channel coherence above which channels are labeled as noisy (together with psd condition),
        by default -0.5
    outside_channel_thr : float
        Threshold for channel coherence above which channels
    n_neighbors : int, optional
        Number of neighbors to compute median fitler, by default 11
    nyquist_threshold : float, optional
        Threshold on Nyquist frequency to calculate HF noise band, by default 0.8
    welch_window_ms: float
        Window size for the scipy.signal.welch that will be converted to nperseg, by default 10ms
    Returns
    -------
    1d array
        Channels labels: 0: good,  1: dead low coherence / amplitude, 2: noisy, 3: outside of the brain
    """
    _, nc = raw.shape
    raw = raw - np.mean(raw, axis=0)[np.newaxis, :]
    nperseg = int(welch_window_ms * fs / 1000)
    import scipy.signal

    fscale, psd = scipy.signal.welch(raw, fs=fs, axis=0, window="hann", nperseg=nperseg)

    # compute similarities
    ref = np.median(raw, axis=1)
    xcorr = np.sum(raw * ref[:, np.newaxis], axis=0) / np.sum(ref**2)

    # compute coherence
    xcorr_neighbors = detrend(xcorr, n_neighbors)
    xcorr_distant = xcorr - detrend(xcorr, n_neighbors) - 1

    # make recommendation
    psd_hf = np.mean(psd[fscale > (fs / 2 * nyquist_threshold), :], axis=0)

    ichannels = np.zeros(nc, dtype=int)
    idead = np.where(xcorr_neighbors < dead_channel_thr)[0]
    inoisy = np.where(np.logical_or(psd_hf > psd_hf_threshold, xcorr_neighbors > noisy_channel_thr))[0]

    ichannels[idead] = 1
    ichannels[inoisy] = 2

    # the channels outside of the brains are the contiguous channels below the threshold on the trend coherency
    # the chanels outide need to be at either extremes of the probe
    ioutside = np.where(xcorr_distant < outside_channel_thr)[0]
    if ioutside.size > 0 and (ioutside[-1] == (nc - 1) or ioutside[0] == 0):
        a = np.cumsum(np.r_[0, np.diff(ioutside) - 1])
        ioutside = ioutside[a == np.max(a)]
        ichannels[ioutside] = 3

    return ichannels

def detrend(x, nmed):
    """
    Subtract the trend from a vector
    The trend is a median filtered version of the said vector with tapering
    :param x: input vector
    :param nmed: number of points of the median filter
    :return: np.array
    """
    ntap = int(np.ceil(nmed / 2))
    xf = np.r_[np.zeros(ntap) + x[0], x, np.zeros(ntap) + x[-1]]

    import scipy.signal

    xf = scipy.signal.medfilt(xf, nmed)[ntap:-ntap]
    return x - xf

# ---------------------------------------------------------------------------
# QC and filtering
# ---------------------------------------------------------------------------

def mask_bad_channels(channel_data, Fs):
    """
    Detect bad channels with the IBL method and NaN them out in-place.

    Returns
    -------
    channel_data : ndarray  bad rows replaced with NaN
    bad_mask     : ndarray of int  0=good, 1=dead, 2=noisy, 3=outside-brain
    """
    bad_mask = detect_bad_channels_ibl(channel_data.T, fs=Fs)
    channel_data[bad_mask != 0, :] = np.nan
    return channel_data, bad_mask


def filter_lfp(channel_data, Fs, low=0.1, high=300, order=3):
    """
    Bandpass-filter channel_data to the LFP band.

    Operates row-wise (channels × samples). NaN rows are preserved as NaN.

    Parameters
    ----------
    channel_data : ndarray, shape (n_channels, n_samples)
    Fs           : float
    low, high    : float   band edges in Hz
    order        : int     Butterworth order
    """
    nyq = 0.5 * Fs
    sos = sg.butter(order, np.array([low, high]) / nyq, btype='band', output='sos')
    return sg.sosfiltfilt(sos, channel_data, axis=1)

def sort_channels(channel_dat, channel_idx, desc = True):
    """[summary]

    Args:
        channel_dat (matrix[channels, samples]): raw channel data
        channel_idx (list[int]): list of indices for channel data ordered top (superficial) to bottom (deep)
        desc (bool): whether superficial to deep should be outputted (default) or deep to superficial (false)
    Returns:
        channel_data_sorted (matrix[channels, samples]): raw channel data sorted by depth of electodes
        channel_ids (list[int]): list of channel numbers [1:N]
    """
    # if full conversion not made, just software channels are mapped onto probe, would look like this and would require
    # sorting, below:
    # neuronexus_pinout = [19, 29, 20, 26, 16, 30, 18, 28, 17, 31, 22, 24, 21, 27, 23, 25]  # deep to superficial - known
    # source: https://neuronexus.com/files/probemapping/16-channel/A16-Maps.pdf + visual inspection of hardware
    # note: software gives range of 16:31 for channels, but on hardware we had 17:32 - I am assuming I just need to -1 to correct.
    # i honestly have no idea what this is:
    # deep_to_sup_idx = [9, 5, 1, 7, 13, 3, 15, 11, 14, 10, 12, 2, 0, 6, 8, 4]#[::-1]
    # deep_to_sup_idx = []
    # for p in np.arange(len(channel_idx)):
    #    deep_to_sup_idx.append(np.where(neuronexus_pinout == channel_idx[p])[0][0])

    channel_ids = np.arange(1, len(channel_idx))

    if desc:
        # row 0 = shallowest, row max = deepest
        channel_data_sorted = channel_dat.take(channel_idx, 0)
    else:
        # row 0 = deepest, row max = shallowest
        channel_data_sorted = channel_dat.take(channel_idx[::-1], 0)
        channel_ids = channel_ids[::-1]

    return (channel_data_sorted, channel_ids)

def signal_win(win, Fs):
    # win : time in seconds relative to trigger when to cut data
    #       if scalar, symmetrical time window around trigger, if len==2, asymmetrical cut
    # Fs : Fs
    win = np.array(win)
    if len(win) == 1:
        win_pre = win * Fs
        win_post = win * Fs
    elif len(win) == 2:
        win_pre = np.abs(win[0]) * Fs
        win_post = np.abs(win[1]) * Fs

    return (int(win_pre), int(win_post))

# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def prepare_lfp(h5_path, probe_file, lfp_band=(0.1, 300), order=3):
    """
    Full pipeline: load → sort channels → mask bad channels → LFP filter.

    Parameters
    ----------
    h5_path    : str
    probe_file : str
    lfp_band   : (float, float)  bandpass edges in Hz
    order      : int             Butterworth order

    Returns
    -------
    channel_lfp : ndarray, shape (n_channels, n_samples)
        NaN rows are bad channels.
    meta : dict
        Fs, duration_s, units, channel_ids, bad_channel_mask, n_bad_channels
    """
    channel_data, channel_ids, Fs, duration_s, units = load_mcs_recording(h5_path, probe_file)
    channel_data, bad_mask = mask_bad_channels(channel_data, Fs)
    channel_lfp = filter_lfp(channel_data, Fs, low=lfp_band[0], high=lfp_band[1], order=order)

    meta = dict(
        Fs=Fs,
        duration_s=duration_s,
        units=units,
        channel_ids=channel_ids,
        bad_channel_mask=bad_mask,
        n_bad_channels=int(np.sum(bad_mask != 0)),
    )
    return channel_lfp, meta
