"""Utility functions"""
import os
import numpy as np
import scipy.io as io


## General utilities
def gen_bins(arr, min=False, binsize=0.001, max=False):
    if min==False and max==False:
        bins = np.arange(np.nanmin(arr), np.nanmax(arr), binsize)
    elif min==False and max!=False:
        bins = np.arange(np.nanmin(arr), max, binsize)
    elif min!=False and max==False:
        bins = np.arange(min, np.nanmax(arr), binsize)
    else:
        bins = np.arange(min, max, binsize)
    return(bins)


## Audio utilities
def convert_audio(dir, format = 'npz', compress = True, keep_channels = True, save_dir = False):
    """
    Converts audio files between formats .wav, .npz, and .npz (compressed). Optionally allows saving individual
    channels.

    Parameters
    ----------
    dirs: str, list
        Path to an audio file.
    format: str
        Desired output format, 'npz' or 'wav'. Default = 'npz' (compressed).
    compress: bool (optional)
        Whether to use np.savez_compressed if format == 'npz'. Default = True.
    keep_channels: bool, int (optional)
        If bool (True), keeps all channels. If int (usually 0 or 1), will save only requested channel. Will also check
        that the channel exists in the file.
    save_dir: str (optional)
        Save to a different directory.

    Returns
    -------
        Saves file with requested format and channels.
    """
    assert type(dir)==list or type(dir)==str, "Path must be either a list or a string."
    if type(dir)==list:
        dir = dir.pop(0)
        assert type(dir)==str and len(dir)==1, "Number of elements in list are non-1 or element is not a string."
    assert os.path.isfile(dir) and ('wav' in dir.rsplit('.')[-1] or 'npz' in dir.rsplit('.')[-1]), "Path must be a file of type 'wav' or 'npz'."
    assert os.path.exists(dir), "404: File not found."
    assert format=='npz' or format=='wav', "Requested format must be 'wav' or 'npz'."
    if format=='npz':
        assert bool(compress), "Compress must be True or False if requested format is 'npz'."
    assert keep_channels is True or keep_channels==0 or keep_channels==1, "Keep channels must be either True (keep all), 0, or 1."
    assert save_dir is False or type(save_dir)==str, "Save directory must be string."

    # fetch existing format
    file_ext = dir.rsplit('.')[-1]

    # read file
    if file_ext == 'wav':
        sr, y = io.wavfile.read(dir)  # y.shape = [Nsamples, Nchannels]
    if file_ext == 'npz':
        npzfile = np.load(dir)
        y = npzfile['data']
        sr = npzfile['samplerate']

    samplerate = np.array([sr], dtype='float')

    # channel selection
    if keep_channels is True:
        data = y
    else:
        if y.shape[-1] >= int(keep_channels):
            data = y[:, int(keep_channels)]
        else:
            raise IndexError('Requested channels not in audio.')

    # filename to save, without ext, but with .
    filename = dir.removesuffix(file_ext)

    # save to given path, otherwise modify in place
    if save_dir is not False: # doesn't work
        filename = os.path.join(save_dir, filename.rsplit('\\')[-1])
    else:
        assert file_ext != format, "Requested format and existing format are the same, and no separate save directory has been requested. Stopping to prevent overwriting."

    filename = filename + format

    if format == 'npz' and compress is True:
        # save as per format requested by DAS: https://janclemenslab.org/das/technical/data_formats.html
        np.savez_compressed(filename, data=data, samplerate=samplerate)
    if format == 'npz' and compress is False:
        np.savez(filename, data=data, samplerate=samplerate)
    # don't give option to save as .npy, stupidly loses samplerate information
    if format == 'wav':
        io.wavfile.write(filename, int(samplerate), data)

    print('Saving ' + filename + ' with channel(s): ' + str(keep_channels) + '...')

def zero_pad_calls(audio, fs=384e3, dur=0.002):
    # audio should be 1d [nsamples,]
    if audio.shape[0]/fs < dur:
        pad_total = dur-(audio.shape[0]/fs)
        padded_audio = np.concatenate((np.zeros(np.floor((pad_total/2)*fs).astype(int)), audio, 
                        np.zeros(np.floor((pad_total/2)*fs).astype(int))))
    else:
        padded_audio = audio        
    return(padded_audio)

