###
# Get psd, psd fitting, and stft for all calls, from spontaneous condition.
# INPUT: /das/analysis_annotations
# OUTPUT: cache_dir + '/v1_spontaneous_calls_acoustic_data_v3.npy'
# ENV: spikey_x86 on cluster
###
import os, sys
from glob import glob
sys.path.append(os.path.join('/mnt/cs/projects/BWFAFdeactivNpx/code') ) # dir. to find toolbox 
from _frontal_vocal_paper_repo.code.vocal_analysis.npxtoolbox import acoustics
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


acoustic_data = {}

anim_dirs = [f'{root_aud_dir}/{d}' for d in next(os.walk(root_aud_dir))[1] if 'Npv1' in d]
anim_dirs.sort()

for anim_dir in anim_dirs: # animals
    anim_tag = anim_dir.split('/')[-1].split('_')[-1]
    print(anim_tag)

    sess_dirs = (np.sort([f'{anim_dir}/{d}' for d in os.listdir(anim_dir) if 'sess' in d]))

    acoustic_data[anim_tag] = {}
    for sess_dir in sess_dirs:
        if sess_dir == '/mnt/cs/projects/BWFAFdeactivNpx/Npv1_data/Npv1_audio_log_data/Npv1_F2/Npv1_F2_sess-3-1_muscimol':
            continue # skip
        sess_tag = sess_dir.split('/')[-1].split('-')[-1]
        sess_n = sess_tag.split('_')[0]
        sess_treat = sess_tag.split('_')[-1]
        print('\t' + sess_tag)
        # get file paths    
        pred_annot_fs = np.sort([d for d in glob(sess_dir+'/das/analysis_annotations/*vocalizations*annotations*.csv') if 'manual' not in d]) 
        
        acoustic_data[anim_tag][sess_tag] = {}
        for pred_annot_f in pred_annot_fs:

            rec_n = pred_annot_f.split('/')[-1].split('-')[4]
            rec_iter = pred_annot_f.split('/')[-1].split('-')[5].split('_')[0]
            annot_dat = pd.read_csv(pred_annot_f)
            
            print('\t' + pred_annot_f)
            print('\tProcessing N calls: '+ str(len(annot_dat)))

            ## get audio
            rel_audio_2d = glob(sess_dir+'/'+pred_annot_f.split('/')[-1].split('_')[0]+'.wav')[0]
            audio_data, sr = acoustics.get_audio(rel_audio_2d, channel=1)

            ## --- highpass filter ---
            hp_filt_at = 10 # in Npv0.1, you hp filtered at 10kHz, I think 10 might be high... [5?]
            sos = signal.butter(4, int(hp_filt_at*1e3)/(0.5*sr), btype='highpass', output='sos') 
            audio_data_hp = signal.sosfiltfilt(sos, audio_data)#,dtype=int)

            # baseline slice
            if sess_n=='1' and rec_n=='rec1': # only reset baseline on first sess/rec for each anim
                print('\tSetting baseline...')
                base_start_stop = silence_dat[anim_tag][sess_tag][rec_n][rec_iter]
                baseline_cut = audio_data_hp[int(base_start_stop[0]*sr):int(base_start_stop[1]*sr)]
            else:
                baseline_cut = baseline_cut

            # acoustic_data[anim_tag][sess_tag][rec_n] = {}
            acoustic_data[anim_tag][sess_tag].setdefault(rec_n, {})
            call_data = {}
            ## ATTENTION: INDIVIDUAL FILES SO HERE THE BASIC INDEX == NTH CALL
            for call_n in annot_dat.index: #range(11,12): 
                arr_data = {}
                call_n_data = annot_dat.loc[call_n]
                # print(call_n)
                call_audio = audio_data_hp[int(call_n_data['start_seconds']*sr):int(call_n_data['stop_seconds']*sr)]

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

                    # harmonics = np.arange(1, 4)
                    # f0_harm = librosa.f0_harmonics(np.abs(stft_x), freqs=stft_f, f0=np.nanmean(f0), harmonics=harmonics)
                    # f0_harm_db = librosa.amplitude_to_db(np.abs(f0_harm)**2, ref=np.max)
                    # best_harm = np.argmax(np.max(f0_harm, axis=1)) # harmonic with most energy
                    best_harm_peak_energy = np.max(f0) # max stft value at best harmonic
                    best_harm_peak_samp = np.argmax(f0) # sample at which highest energy occurs
                    best_harm_peak_time = stft_t[best_harm_peak_samp] # time in s. at which highest energy occurs

                    arr_data['stft'].update(dict(zip(['f0','f0_peaksp','f0_peakt'],[f0, best_harm_peak_samp, best_harm_peak_time]))) # arr, sca, sca

                except Exception as e:
                    print(f"Skipping f0 estimation on {call_n}: {e}")
                    # continue

                call_data[call_n] = arr_data
                
            print('\tProcessed N calls: '+ str(len(call_data)))            
            # acoustic_data[anim_tag][sess_tag][rec_n][rec_iter] = call_data # overwrites!
            acoustic_data[anim_tag][sess_tag][rec_n][rec_iter] = call_data #.update({rec_iter: call_data})
        # save at each sess:
        np.save(cache_dir + '/v1_spontaneous_calls_acoustic_data_v3.npy', acoustic_data, allow_pickle=True)
        print('\tSaving...')            

