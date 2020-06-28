import os
from librosa.core import load, stft, istft, magphase
from librosa.output import write_wav
from concurrent.futures import ThreadPoolExecutor
from time import time
import asyncio
import os,glob
import numpy as np
from multiprocessing import cpu_count
#Thanks to https://github.com/jnzhng/keras-unet-vocal-separation

SAMPLE_RATE = 8192
WINDOW_SIZE = 1024
HOP_LENGTH = 768

def downsample(input_path, output_path):
    wav, _ = load(input_path, sr=SAMPLE_RATE)
    write_wav(output_path, wav, SAMPLE_RATE, norm=True)
    print(f"Saving {output_path}")

def load_as_mag(file):
    wav, _ = load(file, sr=None)
    spectrogram = stft(wav, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH)
    mag, _ = magphase(spectrogram)
    return mag.astype(np.float32)

def save_to_npz(base, sample):
    nps = {}
    mix = load_as_mag(f'{base}/{sample}/mix.wav')
    vocal = load_as_mag(f'{base}/{sample}/vocal.wav')
    inst = load_as_mag(f'{base}/{sample}/inst.wav')
    
    mix_max = mix.max()
    mix_norm = mix / mix_max
    vocal_norm = vocal / mix_max
    inst_norm = inst / mix_max
    #print(f"Saving {sample}")
    try:
        np.savez_compressed(f'MIR-1K_resized/{sample}.npz', mix=mix_norm, vocal=vocal_norm, inst=inst_norm)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    voise = 'MIR-1K/voise'
    bg = 'MIR-1K/bg'
    mix = 'MIR-1K/mix'
    name = 0
    resampled_data = 'MIR-1K_resampled_data'
    base = 'MIR-1K'

    foldernames = []
    for filename in sorted(glob.glob(os.path.join(voise, '*.wav'))):
        foldernames.append(os.path.split(filename)[-1].replace('.wav',''))
    dirs = foldernames
    
    with ThreadPoolExecutor(max_workers=cpu_count() * 2) as pool:
        for i in range(len(dirs)):
            target_dir = 'MIR-1K_resampled_data/{}_{:0>2d}/'.format(base, i+1)
            os.makedirs(target_dir, exist_ok=True)
            pool.submit(downsample, f'{mix}/{dirs[i]}.wav', target_dir + 'mix.wav')
            pool.submit(downsample, f'{bg}/{dirs[i]}.wav', target_dir + 'inst.wav')
            pool.submit(downsample, f'{voise}/{dirs[i]}.wav', target_dir + 'vocal.wav')
    
    # ## Save wav files to npz
    # 1. Load wave files from `corpus_resized`.
    # 2. Apply Short-time Fourier transform (STFT) to audio trios
    # 3. Apply normalization to magnitudes and save as npz dict in `numpy/`
    dirs = sorted(list(os.walk('MIR-1K_resampled_data'))[0][1])
    print(dirs)
    with ThreadPoolExecutor(max_workers=cpu_count() * 2) as pool:
        #print("!!!")
        for i in range(len(dirs)):
            #print("!!!")
            pool.submit(save_to_npz, 'MIR-1K_resampled_data', dirs[i])