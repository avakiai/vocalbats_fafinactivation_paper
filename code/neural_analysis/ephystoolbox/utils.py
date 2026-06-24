import numpy as np
import random

def ranks(sample):
    """
    Return the ranks of each element in an integer sample.
    """
    indices = sorted(range(len(sample)), key=lambda i: sample[i])
    return sorted(indices, key=lambda i: indices[i])

def sample_with_minimum_distance(n=40, k=4, d=5):
    """
    Sample of k elements from range(n), with a minimum distance d.
    """
    sample = random.sample(range(n - (k - 1) * (d - 1)), k)
    return [s + (d - 1) * r for s, r in zip(sample, ranks(sample))]

def gen_bins(arr, min=False, binsize=0.001):
    if min==False:
        bins = np.arange(np.min(arr), np.max(arr), binsize)
    else:
        bins = np.arange(min, np.max(arr), binsize)
    return(bins)

def zero_pad_calls(audio, fs=384e3, dur=0.002):
    # audio should be 1d [nsamples,]
    if audio.shape[0]/fs < dur:
        pad_total = dur-(audio.shape[0]/fs)
        padded_audio = np.concatenate((np.zeros(np.floor((pad_total/2)*fs).astype(int)), audio, 
                        np.zeros(np.floor((pad_total/2)*fs).astype(int))))
    else:
        padded_audio = audio        
    return(padded_audio)
    