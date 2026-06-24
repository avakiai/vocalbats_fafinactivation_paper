"""
ephys toolbox for python 3.

A module for unpacking various data used in neuroethology,
including electrophysiology data (via multichannel systems),
audio recording, and possibly video recording.

Functions provided for sorting data files by conditions in
order to ease preprocessing.

Ava Kiai
2023/07/07

Initialized in conda environment "spikeinterface" with,
minimally, the following packages:
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


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


def sort_files_bydate(files, conds, conts):
    """[summary]
    Arrange MC (h5) files in temporal order and calculate time since last application of substances.
    Somewhat dangerously, relies silently on the presence in the workspace of a dictionary:
        application = {'pre':{'kz':[]},'post':{'kz':[], 'musc':[]}}
    At present, only can be done for only playback condition.

    Args:
        files (list[char]):
        conds (list[char]):
        conts (list[char]):
    Returns:
        sort_to (dict[contrasts]):
    """
    sort_to = {}
    # sort files by date and filter
    for co in conds:
        subgroup = [x for n, x in enumerate(files) if co in x]
        for cont in conts:  # pre/post
            subsubgroup = [x for n, x in enumerate(subgroup) if cont in x]
            # get rec time from .h5 filename and created sorted index for reading in files in order:
            lsin = []
            for f in subsubgroup:
                lsin.append(conv_dt(extract_metadata(f)['dattime']))
            sort_idx = np.argsort(lsin, )
            lsout = np.array(subsubgroup)[sort_idx]
            # get sorted rec times to calculate time deltas
            times = np.array(lsin)[sort_idx]

            sort_to[cont] = {'h5': lsout,
                             # time from first file in minutes
                             'delta_first': np.array([x.total_seconds() for x in (times - times[0])]) / 60,
                             # time from kz app in minutes
                             'delta_kz': np.array([x.total_seconds() for x in (times - application[cont]['kz'])]) / 60}
            if cont == 'post':
                # time from last muscimol app in minutes
                sort_to[cont]['delta_musc'] = np.array(
                    [x.total_seconds() for x in (times - application[cont]['musc'][-1])]) / 60
    return (sort_to)


def sort_files(files, conditions, conts):
    """[summary]
    Arrange audio (wav), annotation (csv), and MC (h5) data into a nested (pre/post) dict. for easier access.

    Args:
        files (list[char]):
        conditions (list[char]):
        conts (list[char]):
    Returns:
        sort_to (dict[contrasts]):
    """
    if conditions is None:
        sort_to = {}
        for cont in conts:  # pre/post
            sort_to[cont] = {'wav': [x for n, x in enumerate(files) if 'wav' in x and cont in x],
                             'trig': [x for n, x in enumerate(files) if 'trig' in x and cont in x],
                             'npy': [x for n, x in enumerate(files) if 'npy' in x and cont in x]}
    else:
        sort_to = dict.fromkeys(conditions)
        for cond in conditions:
            subgroup = [x for n, x in enumerate(files) if cond in x]
            catch_to = {}
            for cont in conts:  # pre/post
                subsubgroup = [x for n, x in enumerate(subgroup) if cont in x]
                catch_to[cont] = {'wav': [x for n, x in enumerate(subsubgroup) if 'wav' in x],
                                  'csv': [x for n, x in enumerate(subsubgroup) if 'csv' in x],
                                  'h5': [x for n, x in enumerate(subsubgroup) if 'h5' in x],
                                  'trig': [x for n, x in enumerate(subsubgroup) if '.trig' in x],
                                  'npy': [x for n, x in enumerate(subsubgroup) if 'npy' in x]}
                if cond == 'ft':
                    catch_to[cont]['bin'] = [x for n, x in enumerate(subsubgroup) if 'MCSstim' in x]

            sort_to[cond] = catch_to
    return (sort_to)


def n_order(list_of_files):
    """[summary]
    If multiple iterations of a condition are labelled (n1, n2...) but not zero-padded so as to be read
    in order by python, create an index array that can be used to re-order the list of files to be analyzed.

    Args:
        list_of_files (list[char]):
    Returns:
        ordnew (list[char]): indices by which to reorder list_of_files
    """
    ordold = list()
    for old in list_of_files:
        t = extract_metadata(old)
        t['n'] = 'n' + t['n'].split('n')[1].zfill(2)
        ordold.append(t['n'])

    ordnew = np.argsort(ordold)

    return(ordnew)


def unpack_ft(filename):
    """[summary]
    extract frequency tuning stimulus presentation info from binary .MCSstim files
    tone timing relative to trigger: |t_pre|tone|t_post|ISI
    Args:
        filename (char):
    Returns:
        ft_data (dict[]):
    """
    ft_bin = np.fromfile(filename, dtype='float32')
    ft_data = {}
    ft_data['n_stims'] = int(ft_bin[0])  # n elements in stim matrix = number of unique stims * n_avgs * n_SPLs
    ft_data['n_SPLs'] = int(ft_bin[ft_data['n_stims'] * 3 + 1])  # number of SPL levels
    # ft_data['stim_vec'] = np.array(ft_bin[1:n_stims+1],dtype=int) # stim vector - each index corresp. to a combination of file x SPL level
    # ^ numbers should reflect the logic: repelem(1:n_stims,n_SPLs) such that 1,1,... = 1@SPL1, 1@SPL2, etc... then repeated 10 times and permuted
    ft_data['stim_num'] = np.array(ft_bin[ft_data['n_stims'] + 1:ft_data['n_stims'] * 2 + 1], dtype=int)  # stim [file] number
    ft_data['stim_SPL'] = np.array(ft_bin[ft_data['n_stims'] * 2 + 1:ft_data['n_stims'] * 3 + 1], dtype=int)  # real SPL
    ft_data['stim_freq'] = ft_map(ft_data)  # lcl function
    ft_data['SPL_vec'] = np.array(ft_bin[ft_data['n_stims'] * 3 + 2:(ft_data['n_stims'] * 3 + 2) + ft_data['n_SPLs']], dtype=int)  # vec of SPL values
    ft_data['freq_vec'] = np.unique(ft_data['stim_freq'])
    ft_data['n_avgs'] = int(ft_bin[ft_data['n_stims'] * 3 + 2 + ft_data['n_SPLs']])  # -4
    ft_data['ISI'] = ft_bin[ft_data['n_stims'] * 3 + 2 + ft_data['n_SPLs'] + 1] / 1000  # ms to s # -3
    ft_data['t_pre'] = ft_bin[ft_data['n_stims'] * 3 + 2 + ft_data['n_SPLs'] + 2] / 1000  # ms to s # -2
    ft_data['t_post'] = ft_bin[ft_data['n_stims'] * 3 + 2 + ft_data['n_SPLs'] + 3] / 1000  # ms to s # -1
    return (ft_data)


def extract_metadata(filename, post = False):
    """[summary]
    extract metadata about file from filename

    :param filename:
    :return: file_mdat:

    """
    fsplit = filename.split('/')[-1].replace('.', '_').split('_')

    file_mdat = {}
    file_mdat['expname'] = fsplit[0]
    file_mdat['exp#'] = fsplit[1]
    file_mdat['anim#'] = fsplit[2]
    file_mdat['animstate'] = fsplit[3]
    file_mdat['contrast'] = fsplit[4]
    file_mdat['condition'] = fsplit[5]
    file_mdat['n'] = fsplit[6]
    if fsplit[-1] == 'h5': # raw recording data
        file_mdat['dattime'] = fsplit[7]
    elif fsplit[-1] == 'npy' and fsplit[-2] == 'spikes': # LFP data or spiking data
        file_mdat['dattime'] = fsplit[7]
        file_mdat['sorting'] = fsplit[8]  # sorter hash
        file_mdat['dattype'] = fsplit[9]  # spiking label
    elif fsplit[-1] == 'npy': # LFP data or spiking data
        file_mdat['dattime'] = fsplit[7]
        file_mdat['dattype'] = fsplit[8]  # processing stage/label
    elif fsplit[-1] == 'csv' and post == True:
        file_mdat['dattime'] = fsplit[7]  # a bit of a stretch
        file_mdat['dattype'] = fsplit[8]  # processing stage
    file_mdat['ftype'] = fsplit[-1]

    return (file_mdat)


def conv_dt(datstr):
    dt = datetime.strptime(datstr, "%Y-%m-%dT%H-%M-%S")
    return (dt)


def ft_map(ft_data, tone_bank = np.arange(10, 95, 5)):
    """[summary]
    :param ft_data: use vector of filenumbers (stim_num)
    :param tone_bank: and tone information
    :return: tone_vec: frequency and time content of tones
    """

      # known, also known that this linearly maps onto 1:n_stims!
    tone_vec = np.zeros([len(ft_data['stim_num']), ], dtype=int)
    for n in np.arange(len(ft_data['stim_num'])):
        tone_vec[n] = tone_bank[ft_data['stim_num'][n] - 1]
    # ft_seq = np.stack([np.arange(1,len(tone_vec)+1), tone_vec, ft_data['stim_SPL']],1) # trigger #, tone (kHz), dB SPL
    return (tone_vec)


def plot_trig_dt(trigs_s):
    """[summary]
    plot a histogram of diffs between consecutive triggers
    :param trigs_s:
    :return:
    """
    plt.hist(np.diff(trigs_s), density=False, color=[0, .67, 0, 0.25])
    plt.axvline(np.median(np.diff(trigs_s)), color='green', label='median delta')
    plt.xlabel('Delta t between triggers [s]')
    plt.ylabel('count')
    # plt.xlim(0, 4)
    # plt.xticks(np.arange(0, 4))
    plt.legend()
    plt.show()


def rescale_channels(channel_dat2d, scale=1):
    """[summary]
    add matrix values of 1:n channels to data itself, to separate them for mc plotting
    :param channel_dat2d:
    :param scale:
    :return: channel_datsc
    """
    channel_datsc = channel_dat2d + (
                np.tile((np.arange(1, channel_dat2d.shape[0] + 1))[::-1], [channel_dat2d.shape[1], 1]).T * scale)
    return (channel_datsc)


