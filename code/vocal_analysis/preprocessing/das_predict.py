import os
from glob import glob
import numpy as np
import pandas as pd
import das.predict

print('Starting prediction...')
# allow as parsed input? if decide to vary, via slurm
model_dir = '/mnt/cs/projects/BWFAFdeactivNpx/Npv1_data/Npv1_audio_log_data/das/v1.1/dataset_v1.1.3.res/20250419_174131'

# allow as parsed input? if decide to vary, via slurm
predict_params = dict(
               batch_size=128, 
               segment_minlen=0.0001, 
               segment_fillgap=0.0009,
               segment_thres=0.5)

# for current run, this is ok, but for spont. vocs and for parameter/model tuning, let bash sort through this and run only predict/save through py
base_dir = '/mnt/cs/projects/BWFAFdeactivNpx/Npv1_data/Npv1_audio_log_data'
dirs = [f'{base_dir}/{d}' for d in next(os.walk(base_dir))[1] if 'Npv1' in d]

for anim_dir in dirs[3:4]: #dirs[0:1] + dirs[3:4]:
    print('\t Starting on files in ' + anim_dir +'\n')
    sess_dirs = np.sort([f'{anim_dir}/{d}/das' for d in os.listdir(anim_dir) if 'Npv1' in d])
    for sess_dir in sess_dirs: # sess_dirs[1:2]
        # if sess_dir =='/mnt/cs/projects/BWFAFdeactivNpx/Npv1_data/Npv1_audio_log_data/Npv1_M1/Npv1_M1_sess-1_saline/das': # no playback vocal data
        #     continue        
        # files to generate predictions for
        # files_to_detect_in_d = np.sort(glob(sess_dir+'/*responses.npz')) # playback trials
        files_to_detect_in_d = np.sort(glob(sess_dir+'/*vocalizations*.npz')) # spontaneous voc recordings
        # where to save predictions
        pred_dir = sess_dir + '/pred_annotations'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)

        for file_to_detect in files_to_detect_in_d:
            print('\t'+file_to_detect)
            
            ## load audio
            audio_data = np.load(file_to_detect,allow_pickle=True)

            # print(f"DAS requires [T, channels], but single-channel wave files are loaded with shape [T,] (data shape is {audio.shape}).")
            Fs = int(audio_data['samplerate'][0])
            audio = audio_data['data']
            audio = np.atleast_2d(audio).T            

            ## predict
            _, segments, class_probabilities, class_names = das.predict.predict(x=audio, 
                                                                                model_save_name=model_dir,
                                                                                verbose=1,
                                                                                batch_size=predict_params['batch_size'],
                                                                                segment_minlen=predict_params['segment_minlen'],
                                                                                segment_fillgap=predict_params['segment_fillgap'],
                                                                                segment_thres=predict_params['segment_thres'])

            ## collect
            onsets = segments['onsets_seconds']
            offsets = segments['offsets_seconds']
            names = segments['sequence']

            predicted = pd.DataFrame({'name':segments['sequence'], 
                        'start_seconds':segments['onsets_seconds'], 
                        'stop_seconds':segments['offsets_seconds']})
            
            ## save            
            save_to_filename = pred_dir + '/' + file_to_detect.split('/')[-1].removesuffix('.npz') + '_annotations_' + model_dir.split('/')[-1] + '.csv'
            print('\tSaving predictions as: ' + save_to_filename)

            predicted.to_csv(save_to_filename, index=False)
