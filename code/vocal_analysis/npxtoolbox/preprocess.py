import pandas as pd
import numpy as np
from ephystoolbox import unpack


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


# ----------------------------------------------------------------------------------------------
# IBL Helpers
# ----------------------------------------------------------------------------------------------


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


def preprocess_calldata(df, file_mdat, contrast, application):
    """[summary]

    :param df: a pandas dataframe with fields name, start_seconds, stop_seconds;
                the first or first triggers should be annotated call events should be labelled either comm or echo
            file_mdat: a dictionary with fields (below)
            contrast: pre/post
            application (dict): needed only if running post conditional
    :return: calls: a df with addition information

    requirements: conv_dt() from module unpack
    """
    calls = df.query('name != "trigger"')  # call events
    if df.query('name == "trigger"').start_seconds.shape[0] == 1:
        trig1_s = float(df.query('name == "trigger"').start_seconds)
    else:
        trig1_s = df.query('name == "trigger"').start_seconds[0]
    calls["start_from_t1"] = calls.start_seconds - trig1_s  # call timing from first detected trigger

    # call timing from musicmol application
    if contrast == 'post':
       # application = kwargs
        calls["t_from_musc_dt"] = unpack.conv_dt(file_mdat['dattime']) + pd.to_timedelta(calls.start_seconds, unit="s") - application[file_mdat['anim#']]['time']
        calls["t_from_musc_s"] = calls.t_from_musc_dt.dt.total_seconds()

    # calculate offsets
    calls["last_offset_to_onset"] = calls.start_seconds - calls.stop_seconds.shift(1)
    calls["offset_to_next_onset"] = calls.stop_seconds.shift(-1)-calls.start_seconds

    return(calls)
