###
# Get psd, psd fitting, and stft for all calls, from playback condition.
# INPUT: cache_dir + '/v1_playback_annotations_data_v2.csv'
# OUTPUT: cache_dir + '/v1_playback_calls_acoustic_data_v2.py'
# ENV: spikey_x86 on cluster
###
import os, sys
from glob import glob
sys.path.append(os.path.join('/mnt/cs/projects/BWFAFdeactivNpx/code') ) # dir. to find toolbox 
from _frontal_vocal_paper_repo.code.vocal_analysis.npxtoolbox import acoustics
from _frontal_vocal_paper_repo.code.vocal_analysis.npxtoolbox import utils
import numpy as np
import pandas as pd
from scipy import signal
import librosa
from fooof import FOOOF

silence_dat = {'F1':{'1_saline':{'rec1':{'1':[0,10]}}}, 
              'F2':{'1_muscimol':{'rec1':{'1':[0,10]}}},  
              'M1':{'1_saline':{'rec1':{'1':[10,20]}}},  
              'M2':{'1_muscimol':{'rec1':{'1':[10,20]}}}
              } 

# log data and audio recordings
root_aud_dir = '/mnt/cs/projects/BWFAFdeactivNpx/Npv1_data/Npv1_audio_log_data'
cache_dir = '/mnt/cs/projects/BWFAFdeactivNpx/Npv1_data/Npv1_cachedata'


data_df = pd.read_csv(cache_dir + '/v1_playback_annotations_data_v2.csv')

data_df_sorted = data_df.sort_values(['anim','sess'])

# new
acoustic_data = {}

for anim in data_df_sorted['anim'].unique():
    # new
    acoustic_data[anim] = {}
    for sess in data_df_sorted.loc[data_df_sorted['anim']==anim]['sess'].unique():
        # new
        acoustic_data[anim][sess] = {}
        for condition in data_df_sorted.loc[(data_df_sorted['anim']==anim) & (data_df_sorted['sess']==sess)]['condition'].unique():
            # new
            acoustic_data[anim][sess][condition] = {}
            for n in data_df_sorted.loc[(data_df_sorted['anim']==anim) & (data_df_sorted['sess']==sess) & (data_df_sorted['condition']==condition)]['n'].unique():
                print(anim, sess, condition, n)
                
                data_in_file = data_df_sorted.loc[(data_df_sorted['anim']==anim) & 
                                                  (data_df_sorted['sess']==sess) & 
                                                  (data_df_sorted['condition']==condition) &
                                                  (data_df_sorted['n']==n)].reset_index(drop=True) ## THIS LAST ALLOWS CALL_N BELOW TO MATCH Nth CALL IN REC!
                                                
                d = data_in_file.iloc[0]
                rel_audio_sess = root_aud_dir + '/Npv1_'+d['anim'] + '/Npv1_'+d['anim']+'_sess-'+d['sess']
                rel_audio_f_ca = d['anim']+'-'+str(d['sess_n'])+'-'+d['treatment']+'-playback-'+d['condition']+'-*-'+str(d['n'])+'.npz'
                audio_f = glob(rel_audio_sess+ '/'+ rel_audio_f_ca)
                if len(audio_f)!=1:
                    print('Problem searching for ', audio_f)
                    continue
                else:
                    audio_f = audio_f[0]                  
                    print(audio_f)

                audio_data = np.load(audio_f,mmap_mode='r')['data']    
                sr = np.load(audio_f,mmap_mode='r')['samplerate'].astype(int)[0]

                # --- highpass filter ---
                hp_filt_at = 10 # in Npv0.1, you hp filtered at 10kHz, I think 10 might be high... [5?]
                sos = signal.butter(4, int(hp_filt_at*1e3)/(0.5*sr), btype='highpass', output='sos') 
                audio_data_hp = signal.sosfiltfilt(sos, audio_data)
                
                # new
                # baseline slice
                if d['sess_n']==1: # only reset baseline on first sess/rec for each anim
                    print('\tSetting baseline...')
                    # get rel. baseline file
                    rel_audio_sess = root_aud_dir + '/Npv1_'+ d['anim'] + '/Npv1_'+d['anim']+'_sess-'+d['sess']
                    baseline_f = glob(rel_audio_sess + '/' + d['anim'] + '-' + str(list(silence_dat[d['anim']].keys())[0].replace('_','-')) + '-*-rec1-1.wav')
                    print('\t'+ baseline_f[0])
                    # load and hp filt:
                    baseline_audio, _ = acoustics.get_audio(baseline_f[0], channel=1)
                    baselineaudio_data_hp = signal.sosfiltfilt(sos, baseline_audio)
                    base_start_stop = silence_dat[anim][d['sess']]['rec1']['1']
                    baseline_cut = baselineaudio_data_hp[int(base_start_stop[0]*sr):int(base_start_stop[1]*sr)]
                else:
                    baseline_cut = baseline_cut

                # acoustic_data[anim][sess][condition][n] = {}
                acoustic_data[anim][sess][condition].setdefault(n, {})
                call_data = {}                
                ## ATTENTION! As of resetting index, index follows nth call in rec
                for call_n in data_in_file.index:
                    # new
                    arr_data = {}
                    call_n_data = data_in_file.loc[call_n]

                    # cut call audio segment
                    call_audio = audio_data_hp[int(call_n_data['start_seconds']*sr):
                                               int(call_n_data['stop_seconds']*sr)]


                    ## --- get PSD --- 
                    nfft = acoustics.get_powerof2(len(call_audio))
                    if np.log2(nfft)>=8:
                        nfft = 2**np.log2(nfft)-1 # 8-->7, 9-->8, 10-->9
                        if np.log2(nfft)>11:
                            nfft = np.min((nfft,2**11)) # max 2048
                    nfft = int(nfft)

                    nperseg = nfft//2
                    ovlp = nperseg//2
                    freq_ax, psd_w = signal.welch(call_audio, sr, nfft=nfft, nperseg=nperseg, noverlap=ovlp, window='hann', scaling="density")

                    # normalize psd
                    _, baseline_psd_w = signal.welch(baseline_cut, sr, nfft=nfft, nperseg=nperseg, noverlap=ovlp, window='hann', scaling="density") # [V**2/Hz]
                    constant = 1
                    psd_w_norm = psd_w - baseline_psd_w + constant

                    # compute
                    spec_peak = freq_ax[psd_w_norm.argmax()]

                    _, power_spec = signal.periodogram(call_audio, sr, nfft=nfft, window='hann', scaling='spectrum') # squared amplitude spectrum [V**2]
                    _, baseline_power_spec = signal.periodogram(baseline_cut, sr, nfft=nfft, window='hann', scaling='spectrum') # squared amplitude spectrum [V**2]
                    rms = np.sqrt(np.nanmean((power_spec-baseline_power_spec)**2)) # REAL ONE

                    ### ADD DATA
                    arr_data['psd'] = {'psd_norm': psd_w_norm, 
                                    'freq_ax': freq_ax, 
                                    'meta': {'nfft': nfft, 'baseline_psd': baseline_psd_w}}
                    arr_data['psd'].update(dict(zip(['spec_peak','rms'],[spec_peak, rms]))) # sca, sca


                    ## --- spectral fitting --- 
                    specfit = FOOOF(peak_width_limits=[1e3, 20e3], 
                                    max_n_peaks=4, 
                                    # peak_threshold=1, # default 2 relative units of the power spectrum (standard deviation)
                                    # min_peak_height=1 # default 0 absolute units of the power spectrum (log power)
                                    verbose=False,
                                    )
                    freq_range = [8, 180e3]
                    # specfit.set_check_data_mode(False)
                    specfit.fit(freq_ax, psd_w_norm, freq_range)

                    # extract
                    freqs = specfit.freqs
                    # specfit.get_model('full','log')
                    # specfit.get_model('peak','log')
                    # specfit.get_data('peak','log')
                    # specfit.get_model('aperiodic','log')

                    # peak values
                    center_fs = specfit.peak_params_[:,0]
                    # normalized power
                    norm_powers = specfit.peak_params_[:,1]
                    # bandwidth
                    bws = specfit.peak_params_[:,2]

                    # compute
                    bw_total = np.sum(bws)
                    auc = np.trapz(specfit._peak_fit) # same as get_model('peak','log')

                    ### ADD DATA
                    arr_data['specfit'] = {'fullfit_log10': specfit.get_model('full','log'), 
                                        'peakfit_log10': specfit.get_model('peak','log'),
                                        'peakdat_log10': specfit.get_data('peak','log'),
                                        'apfit_log10': specfit.get_model('aperiodic','log'),
                                        'freq_ax': freqs, 
                                        'meta': {'r2': specfit.r_squared_,}}
                    arr_data['specfit'].update(dict(zip(['center_fs','norm_powers','bandwidths','total_bw','auc'],[center_fs, norm_powers, bws, bw_total, auc]))) # arr, arr, arr, sca, sca

                    ## --- stft ---

                    # !! attention, using different nfft, here called n_fft !!
                    if np.log2(nfft)>=8:
                        n_fft = int(2**np.log2(nfft)-1) # go one down from whatever the psd one is
                    else:
                        n_fft = int(nfft) 

                    win_len = n_fft//2
                    hop_len = win_len//2
                    stft_f, stft_t, stft_x = signal.stft(call_audio, fs=sr, nfft=n_fft, nperseg=win_len, noverlap=hop_len, scaling='spectrum')

                    # ALSO SAVE ONE COMMON NFFT-LENGTH STFT, for UMAP!
                    common_ground_nfft = 256
                    if len(call_audio)<common_ground_nfft//2:
                        stft_x_cg = []; stft_t_cg = []; stft_f_cg = []
                    else:
                        # cg_nperseg = common_ground_nfft #//2 # defaults to 256
                        # cg_ovlp = cg_nperseg//2 # defaults to nperseg/2
                        stft_f_cg, stft_t_cg, stft_x_cg = signal.stft(call_audio, fs=sr, nfft=common_ground_nfft, nperseg=common_ground_nfft//2, scaling='spectrum')

                    ## compute
                    centroid = librosa.feature.spectral_centroid(y=call_audio, sr=sr, n_fft=n_fft, win_length=win_len, hop_length=hop_len)
                    bandwidth = librosa.feature.spectral_bandwidth(y=call_audio, sr=sr, n_fft=n_fft, win_length=win_len, hop_length=hop_len)
                    flatness = librosa.feature.spectral_flatness(y=call_audio, n_fft=n_fft, win_length=win_len, hop_length=hop_len)

                    ### ADD DATA
                    arr_data['stft'] = {'stft_x': stft_x, 
                                        'stft_t': stft_t,
                                        'stft_f': stft_f,
                                        'stft_x_cg': stft_x_cg, 
                                        'stft_t_cg': stft_t_cg,
                                        'stft_f_cg': stft_f_cg,
                                        'meta': {'nfft': n_fft, 'common_ground_nfft': common_ground_nfft}}
                    arr_data['stft'].update(dict(zip(['centroid','bandwidth','flatness'],[centroid, bandwidth, flatness]))) # arr, arr, arr
                    
                    try:
                        f0,_,_ = librosa.pyin(y=call_audio, sr=sr, fmin=10e3, fmax=192e3, 
                                            frame_length=np.max((2**10, n_fft)), # 1024, or the true power of 2 just under seg length 
                                            hop_length=hop_len
                                            ) 

                        best_harm_peak_energy = np.max(f0) # max stft value at best harmonic
                        best_harm_peak_samp = np.argmax(f0) # sample at which highest energy occurs
                        best_harm_peak_time = stft_t[best_harm_peak_samp] # time in s. at which highest energy occurs

                        arr_data['stft'].update(dict(zip(['f0','f0_peaksp','f0_peakt'],[f0, best_harm_peak_samp, best_harm_peak_time]))) # arr, sca, sca

                    except Exception as e:
                        print(f"Skipping f0 estimation on {call_n}: {e}")
                        # continue

                    call_data[call_n] = arr_data

                print('\tProcessed N calls: '+ str(len(call_data)))
                acoustic_data[anim][sess][condition][n] = call_data

            # save at each sess:
            np.save(cache_dir + '/v1_playback_calls_acoustic_data_v2.npy', acoustic_data, allow_pickle=True)
            print('\tSaving...')    