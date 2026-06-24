import das.train
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir',type=str,required=True,help='Enter directory to .npy dataset directory.')
args = parser.parse_args()
dataset_dir = args.dataset_dir

print('Start DAS training...')

models_dir = dataset_dir.removesuffix('.npy') + '.res'


print('Training model on: ')
print(dataset_dir)

print('Saving model to: ')
print(models_dir)

train_params = dict(nb_conv = 4, 
                    nb_filters = 64,
                    kernel_size = 32,
                    nb_hist = 4096, 
                    pre_nb_conv = 0, # default = 0
                    
                    nb_epoch = 400,
                    reduce_lr = True, 
                    reduce_lr_patience = 5, # default = 5
                    learning_rate = 0.0001, # default
                    seed = 42) 

model, params = das.train.train(data_dir=dataset_dir, 
                                save_dir=models_dir,

                                nb_hist=train_params['nb_hist'], # chunk size
                                pre_nb_conv=train_params['pre_nb_conv'], # downsample 16x STFT
                                nb_filters=train_params['nb_filters'], # kernels, default = 16
                                kernel_size=train_params['kernel_size'], # default = 16
                                learning_rate=train_params['learning_rate'], # default = 0.0001
                                nb_conv=train_params['nb_conv'], # TCN blocks, default = 3
                                nb_epoch=train_params['nb_epoch'], # default = 400
                                log_messages=True,
                                reduce_lr = train_params['reduce_lr'],
                                reduce_lr_patience = train_params['reduce_lr_patience'],
                                seed = train_params['seed'])
