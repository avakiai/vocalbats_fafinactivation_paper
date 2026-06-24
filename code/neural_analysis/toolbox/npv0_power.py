import numpy as np
from scipy import signal


def fft_windowed(array_1d, 
                 Fs,
                 nfft=2048,
                 window_type='hann',
                 scaling='spectrum', 
                 win_len_s=5, 
                 win_step_s=1
                 ):
    """
    """

    win_samp = int(win_len_s * Fs)
    step_samp = int(win_step_s * Fs)
    n = len(array_1d)
    starts = np.arange(0, n, step_samp)
    starts = starts[starts + nfft <= n]  # keep only windows with enough samples

    # get freq. axis once
    freq_ax, _ = signal.periodogram(array_1d[starts[0] : min(starts[0]+win_samp, n)], Fs, nfft=nfft, window=window_type, scaling=scaling)
    
    power_spec_ts = np.array([signal.periodogram(array_1d[s : min(s+win_samp, n)], 
                                                Fs, nfft=nfft, window=window_type, scaling=scaling)[1]
                             for s in starts])

    return freq_ax, power_spec_ts

def welch_windowed(array_1d, 
                 Fs,
                 nfft=2048,
                 nperseg = None,
                 ovlp = None,
                 window_type='hann',
                 scaling='density', 
                 win_len_s=5, 
                 win_step_s=1
                 ):
    """
    """
    if nperseg is None:
        nperseg = nfft//2
    if ovlp is None:
        ovlp = nperseg//2

    win_samp = int(win_len_s * Fs)
    step_samp = int(win_step_s * Fs)
    n = len(array_1d)
    starts = np.arange(0, n, step_samp)
    starts = starts[starts + nperseg <= n]  # keep only windows with enough samples

    # get freq. axis once
    freq_ax, _ = signal.welch(array_1d[starts[0] : min(starts[0]+win_samp, n)], 
                              Fs, nfft=nfft, nperseg=nperseg, noverlap=ovlp, window=window_type, scaling=scaling)

    welch_psd_ts = np.array([signal.welch(array_1d[s : min(s+win_samp, n)], 
                                          Fs, nfft=nfft, nperseg=nperseg, noverlap=ovlp, window=window_type, scaling=scaling)[1]
                            for s in starts])


    return freq_ax, welch_psd_ts

def bandpass_filter(array_nd, 
                    Fs,
                    bound, 
                    order=4
                    ):
    """
    bandpass_bounds: list of lists
    """
    sos = signal.butter(4, np.array([bound[0], bound[1]])/(0.5*Fs), btype='band', output='sos') 
    sig = signal.sosfiltfilt(sos, array_nd)

    return sig                   
