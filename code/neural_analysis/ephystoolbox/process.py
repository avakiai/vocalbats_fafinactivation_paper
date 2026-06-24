import numpy as np
from scipy import signal
from fooof.utils import interpolate_spectrum

def compute_psd(x, Fs, nperseg, ovlp, nfft, remove_linenoise=True):
    # requires from fooof.utils import interpolate_spectrum
    freq_ax, psd = signal.welch(x, fs=Fs, nperseg=nperseg, noverlap=ovlp, nfft=nfft, return_onesided=True)

    if remove_linenoise:
        powers_i = []

        for i in range(psd.shape[0]):  # for passing multiple trials/segments at once, in first dimension
            interp_range1 = [48, 52]
            freqs_int, powers_int = interpolate_spectrum(freq_ax, psd[i], interp_range1)
            interp_range2 = [58, 62]
            freqs_int2, powers_int2 = interpolate_spectrum(freqs_int, powers_int, interp_range2)
            powers_i.append(powers_int2)

        freq_ax = freqs_int2
        psd = np.array(powers_i)

    return (psd, freq_ax)