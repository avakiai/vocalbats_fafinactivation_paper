import numpy as np
from scipy import io
from PIL import Image

def get_powerof2(x):
    return 1 << x.bit_length()-1

def get_audio(filepath, channel=-1):
    """
    Parameters
    ----------
    filepath : str        
    channel : int, optional
        for wav files

    Returns
    -------
    audio
    sampling rate

    """


    if filepath.split('.')[-1]=='npz':
        audio_data = np.load(filepath, mmap_mode='r')['data']    
        sr = np.load(filepath, mmap_mode='r')['samplerate'].astype(int)[0]
    
    if filepath.split('.')[-1]=='wav':
        audio_data_nd = io.wavfile.read(filepath, mmap=True)[1]
        audio_data = audio_data_nd[:,channel]
        sr = io.wavfile.read(filepath, mmap=True)[0]

    return (audio_data, sr)


def log_resize_spec(spec: np.ndarray, scaling_factor=10) -> np.ndarray:
    """Log resize time axis. SCALING_FACTOR determines nonlinearity of scaling."""
    #from https://github.com/timsainb/avgn_paper
    resize_shape = [int(np.log(spec.shape[1]) * scaling_factor), spec.shape[0]]
    resize_spec = np.array(Image.fromarray(spec).resize(resize_shape, Image.LANCZOS))
    return resize_spec