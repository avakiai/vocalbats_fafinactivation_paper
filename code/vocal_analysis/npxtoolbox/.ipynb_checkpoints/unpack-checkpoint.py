"""
neuropixel toolbox for python 3.

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
import os
from ast import literal_eval as ale
from pathlib import Path
import numpy as np
from datetime import datetime


def get_npx_metadata(filename):
    """[summary]
    extract metadata about file from file or directory name

    :param filename:
    :return: file_mdat:

    """

    if os.path.isfile(filename):

        fsplit = filename.split('/')[-1].replace('.', '_').split('_')
    
        file_mdat = {}
        file_mdat['exp'] = fsplit[0]
        file_mdat['anim'] = fsplit[1]
        file_mdat['sess'] = {'n': fsplit[2].split('-')[1], "label": fsplit[2]}
        file_mdat['treatment'] = fsplit[3]
        file_mdat['date'] = fsplit[4]
        file_mdat['condition'] = fsplit[5]
        file_mdat['n'] = {'n': fsplit[6].split('-')[1], "label": fsplit[6]}
        file_mdat['gseries'] = fsplit[7]
        file_mdat['tseries'] = fsplit[8]
        file_mdat['imec'] = fsplit[9]
        file_mdat['filt'] = fsplit[10]
        file_mdat['ext'] = fsplit[11]
        
    elif os.path.isdir(filename):
        
        fsplit = filename.split('/')[-1].replace('.', '_').split('_')
    
        file_mdat = {}
        file_mdat['exp'] = fsplit[0]
        file_mdat['anim'] = fsplit[1]
        file_mdat['sess'] = {'n': fsplit[2].split('-')[1], "label": fsplit[2]}
        file_mdat['treatment'] = fsplit[3]
        file_mdat['date'] = fsplit[4]
        file_mdat['condition'] = fsplit[5]
        file_mdat['n'] = {'n': fsplit[6].split('-')[1], "label": fsplit[6]}
        file_mdat['gseries'] = fsplit[7]
        # will not read final _imec0 extension, which is named automatically 
        
    return file_mdat


def metadata(dp):
    '''
    Copied from https://github.com/m-beau/NeuroPyxels/ - npyx doesn't play well with spikeinterface env,
    here a way of porting just this function

    Read spikeGLX (.ap/lf.meta) or openEphys (.oebin) metadata files
    and returns their contents as dictionnaries.

    The 'highpass' or 'lowpass' nested dicts correspond to Neuropixels 1.0 high or low pass filtered metadata.
    2.0 recordings only have a 'highpass' key, as they are acquired as a single file matched with a .ap.meta file.
        for spikeGLX, corresponds to metadata of .ap.meta and .lf.meta files.
        for OpenEphys, .oebin metadata relating to the first and second dictionnaries in 'continuous' of the .oebin file
                       which match the /continuous/Neuropix-PXI-100.0 or .1 folders respectively.

    Arguments:
        - dp: str, datapath to spike sorted dataset

    Returns:
        - meta: dictionnary containing contents of meta file.
        the structure of meta is as follow:
        {
        'probe_version': either of '3A', '1.0_staggered', '2.0_1shank', '2.0_4shanks', 'ultra_high_density';
        'highpass':
            {
            'binary_relative_path':relative path to binary file from dp,
            'sampling_rate':int, # sampling rate
            'n_channels_binaryfile':int, # n channels saved on file, typically 385 for .bin and 384 for .dat
            'n_channels_analysed':int, # n channels used for spikesorting. Will set the shape of temp_wh.daat for kilosort.
            'datatype':str, # datatype of binary encoding, typically int16
            'binary_relative_path':relative path to binary file from dp,
            'key1...': all other keys present in meta file, that you must be familiar with!
                       e.g. 'fileSizeBytes' for spikeGLX or 'channels' for OpenEphys...
            },
        'lowpass': {...}, # same as high for low pass filtered data (not existing in 2.0 recordings)
        'events': {...}, # only for openephys recordings, contents of oebin file
        'spikes': {...} # only for openephys recordings, contents of oebin file
        }
    '''
    dp = Path(dp)
    assert dp.exists(), "Provided path does not exist!"
    assert dp.is_dir(), f"Provided path {dp} is a filename!"

    probe_versions = {
        'glx': {3.0: '3A',  # option 3
                0.0: '1.0',
                21: '2.0_singleshank',
                24: '2.0_fourshanks',
                1123: 'ultra_high_density',
                1030: 'NHP_1.0'},
        'oe': {"Neuropix-3a": '3A',  # source_processor_name keys
               "Neuropix-PXI": '1.0',
               '?1': '2.0_singleshank',  # do not know yet
               '?2': '2.0_fourshanks'},  # do not know yet
        'int': {'3A': 1,
                '1.0': 1,
                'NHP_1.0': 1,
                '2.0_singleshank': 2,
                '2.0_fourshanks': 2,
                'ultra_high_density': 3}
    }

    # import params.py data
    # params_f = dp / 'params.py'
    # if params_f.exists():
    #     params = read_pyfile(dp / 'params.py')

    # find meta file
    glx_ap_files = list_files(dp, "ap.meta", True)
    glx_lf_files = list_files(dp, "lf.meta", True)
    oe_files = list_files(dp, "oebin", True)
    glx_found = np.any(glx_ap_files)
    assert glx_found, \
        f'WARNING no .ap/lf.meta (spikeGLX) or .oebin (OpenEphys) file found at {dp}.'
    assert len(glx_ap_files) == 1 or len(glx_lf_files) == 1, \
        'WARNING more than 1 .ap.meta or 1 .oebin files found!'

    # Formatting of openephys meta file
    meta = {}
    meta['path'] = os.path.realpath(dp)

    # Formatting of SpikeGLX meta file
    if glx_found:
        meta['acquisition_software'] = 'SpikeGLX'
        # Load SpikeGLX metadata
        meta_glx = {}
        for metafile in glx_ap_files + glx_lf_files:
            if metafile in glx_ap_files:
                filtkey = 'highpass'
            elif metafile in glx_lf_files:
                filtkey = 'lowpass'
            metafile = Path(metafile)
            meta_glx[filtkey] = {}
            with open(metafile, 'r') as f:
                for ln in f.readlines():
                    tmp = ln.split('=')
                    k, val = tmp[0], ''.join(tmp[1:])
                    k = k.strip()
                    val = val.strip('\r\n')
                    if '~' in k:
                        meta_glx[filtkey][k] = val.strip('(').strip(')').split(')(')
                    else:
                        try:  # is it numeric?
                            meta_glx[filtkey][k] = float(val)
                        except:
                            meta_glx[filtkey][k] = val

        # find probe version
        if 'imProbeOpt' in meta_glx["highpass"]:  # 3A
            glx_probe_version = meta_glx["highpass"]["imProbeOpt"]
        elif 'imDatPrb_type' in meta_glx["highpass"]:  # 1.0 and beyond
            glx_probe_version = meta_glx["highpass"]["imDatPrb_type"]
        else:
            glx_probe_version = 'N/A'

        if glx_probe_version in probe_versions['glx']:
            meta['probe_version'] = probe_versions['glx'][glx_probe_version]
            meta['probe_version_int'] = probe_versions['int'][meta['probe_version']]
        else:
            print(
                f'WARNING probe version {glx_probe_version} not handled - post an issue at www.github.com/m-beau/NeuroPyxels and provide your .ap.meta file.')
            meta['probe_version'] = glx_probe_version
            meta['probe_version_int'] = 0

        # Based on probe version,
        # Find the voltage range, gain, encoding
        # and deduce the conversion from units/bit to uV
        Vrange = (meta_glx["highpass"]['imAiRangeMax'] - meta_glx["highpass"]['imAiRangeMin']) * 1e6
        if meta['probe_version'] in ['3A', '1.0', 'ultra_high_density', 'NHP_1.0']:
            if Vrange != 1.2e6: print(
                f'\u001b[31mHeads-up, the voltage range seems to be {Vrange}, which is not the default (1.2*10^6). Might be normal!')
            bits_encoding = 10
            ampFactor = ale(meta_glx["highpass"]['~imroTbl'][1].split(' ')[3])  # typically 500
            # if ampFactor!=500: print(f'\u001b[31mHeads-up, the voltage amplification factor seems to be {ampFactor}, which is not the default (500). Might be normal!')
        elif meta['probe_version'] in ['2.0_singleshank', '2.0_fourshanks']:
            if Vrange != 1e6:
                print(
                    f'\u001b[31mHeads-up, the voltage range seems to be {Vrange}, which is not the default (10^6). Might be normal!')
            bits_encoding = 14
            ampFactor = 80  # hardcoded
        else:
            raise ValueError(f"Probe version unhandled - bits_encoding unknown.")
        meta['bit_uV_conv_factor'] = (Vrange / 2 ** bits_encoding / ampFactor)

        # find everything else
        for filt_key in ['highpass', 'lowpass']:
            if filt_key not in meta_glx.keys(): continue
            meta[filt_key] = {}

            # binary file
            filt_suffix = {'highpass': 'ap', 'lowpass': 'lf'}[filt_key]
            # binary_rel_path = get_binary_file_path(dp, filt_suffix, False)
            # if binary_rel_path != 'not_found':
            #     meta[filt_key]['binary_byte_size'] = os.path.getsize(dp / binary_rel_path)
            #     meta[filt_key]['binary_relative_path'] = './' + binary_rel_path
            # else:
            meta[filt_key]['binary_byte_size'] = 'unknown'
            #     meta[filt_key]['binary_relative_path'] = binary_rel_path
                # print(f"\033[91;1mWARNING binary file .{filt_suffix}.bin not found at {dp}\033[0m")

            # sampling rate
            if meta_glx[filt_key]['typeThis'] == 'imec':
                meta[filt_key]['sampling_rate'] = float(meta_glx[filt_key]['imSampRate'])
            else:
                meta[filt_key]['sampling_rate'] = float(meta_glx[meta_glx['typeThis'][:2] + 'SampRate'])

            meta[filt_key]['n_channels_binaryfile'] = int(meta_glx[filt_key]['nSavedChans'])
            # if params_f.exists():
            #     meta[filt_key]['n_channels_analysed'] = params['n_channels_dat']
            #     meta[filt_key]['datatype'] = params['dtype']
            # else:
            meta[filt_key]['n_channels_analysed'] = meta[filt_key]['n_channels_binaryfile']
            meta[filt_key]['datatype'] = 'int16'
            meta[filt_key] = {**meta[filt_key], **meta_glx[filt_key]}

    # Calculate length of recording
    high_fs = meta['highpass']['sampling_rate']

    if meta['highpass']['binary_byte_size'] == 'unknown':
        if (dp / 'spike_times.npy').exists():
            t_end = np.load(dp / 'spike_times.npy').ravel()[-1]
            meta['recording_length_seconds'] = t_end / high_fs
        else:
            meta['recording_length_seconds'] = 'unknown'
    else:
        file_size = meta['highpass']['binary_byte_size']
        item_size = np.dtype(meta['highpass']['datatype']).itemsize
        nChans = meta['highpass']['n_channels_binaryfile']
        meta['recording_length_seconds'] = file_size / item_size / nChans / high_fs

    return meta

def list_files(directory, extension, full_path=False):
    """
    List files with extension "extension" in directory "directory"."""
    directory=str(directory)
    if extension[0]!='.': extension = '.'+extension
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    files.sort()
    if full_path:
        return [Path('/'.join([directory,f])) for f in files]
    return files

def parse_dt(datstr):
    parsed = datetime.strptime(datstr, "%Y-%m-%dT%H:%M:%S")

    return parsed

# def sort_files(files, conditions, conts):
#     """[summary]
#     Arrange audio (wav), annotation (csv), and MC (h5) data into a nested (pre/post) dict. for easier access.

#     Args:
#         files (list[char]):
#         conditions (list[char]):
#         conts (list[char]):
#     Returns:
#         sort_to (dict[contrasts]):
#     """
#     if conditions is None:
#         sort_to = {}
#         for cont in conts:  # pre/post
#             sort_to[cont] = {'wav': [x for n, x in enumerate(files) if 'wav' in x and cont in x],
#                              'trig': [x for n, x in enumerate(files) if 'trig' in x and cont in x],
#                              'npy': [x for n, x in enumerate(files) if 'npy' in x and cont in x]}
#     else:
#         sort_to = dict.fromkeys(conditions)
#         for cond in conditions:
#             subgroup = [x for n, x in enumerate(files) if cond in x]
#             catch_to = {}
#             for cont in conts:  # pre/post
#                 subsubgroup = [x for n, x in enumerate(subgroup) if cont in x]
#                 catch_to[cont] = {'wav': [x for n, x in enumerate(subsubgroup) if 'wav' in x],
#                                   'csv': [x for n, x in enumerate(subsubgroup) if 'csv' in x],
#                                   'h5': [x for n, x in enumerate(subsubgroup) if 'h5' in x],
#                                   'trig': [x for n, x in enumerate(subsubgroup) if '.trig' in x],
#                                   'npy': [x for n, x in enumerate(subsubgroup) if 'npy' in x]}
#                 if cond == 'ft':
#                     catch_to[cont]['bin'] = [x for n, x in enumerate(subsubgroup) if 'MCSstim' in x]

#             sort_to[cond] = catch_to
#     return (sort_to)


# def n_order(list_of_files):
#     """[summary]
#     If multiple iterations of a condition are labelled (n1, n2...) but not zero-padded so as to be read
#     in order by python, create an index array that can be used to re-order the list of files to be analyzed.

#     Args:
#         list_of_files (list[char]):
#     Returns:
#         ordnew (list[char]): indices by which to reorder list_of_files
#     """
#     ordold = list()
#     for old in list_of_files:
#         t = extract_metadata(old)
#         t['n'] = 'n' + t['n'].split('n')[1].zfill(2)
#         ordold.append(t['n'])

#     ordnew = np.argsort(ordold)

#     return(ordnew)

# def plot_trig_dt(trigs_s):
#     """[summary]
#     plot a histogram of diffs between consecutive triggers
#     :param trigs_s:
#     :return:
#     """
#     plt.hist(np.diff(trigs_s), density=False, color=[0, .67, 0, 0.25])
#     plt.axvline(np.median(np.diff(trigs_s)), color='green', label='median delta')
#     plt.xlabel('Delta t between triggers [s]')
#     plt.ylabel('count')
#     # plt.xlim(0, 4)
#     # plt.xticks(np.arange(0, 4))
#     plt.legend()
#     plt.show()
