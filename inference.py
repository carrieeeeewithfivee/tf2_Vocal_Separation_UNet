import os
import cv2
import time
import numpy as np
import tensorflow as tf
import pylab
import librosa
from absl import app, flags, logging
from librosa.core import istft, load, stft, magphase
from librosa.output import write_wav
from Unet import Unet
import mir_eval

SAMPLE_RATE = 8192
WINDOW_SIZE = 1024
HOP_LENGTH = 768
START = 0
END = START + 128

FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt_path', './checkpoints/checkpoints_corpus_mir1k/tf_ckpt/ckpt-156000', 'Link to the file of TensorFlow checkpoint.')
flags.DEFINE_string('original_wav', './data/MIR-1K_resampled_data/MIR-1K_01/mix.wav', 'Link to the wav')
flags.DEFINE_string('original_gt', './data/MIR-1K_resampled_data/MIR-1K_01/vocal.wav', None)
flags.DEFINE_string('output_dir', './predictions/predict_wav_corpus_mir1k156000/', None)
flags.DEFINE_bool('gt',False,'Whether there is ground truth provided')

def spectogram_librosa(_wav_file_,flag):
    (sig, rate) = librosa.load(_wav_file_, sr=None, mono=True,  dtype=np.float32)
    pylab.specgram(sig, Fs=rate)
    if flag==0:
        pylab.savefig(_wav_file_+'spec_input.png')
    else:
        pylab.savefig(_wav_file_+'spec_output.png')

def restore(net, ckpt_path):
    checkpoint = tf.train.Checkpoint(net=net)
    if os.path.isdir(ckpt_path):
        latest_checkpoint = tf.train.latest_checkpoint(ckpt_path)
        status = checkpoint.restore(latest_checkpoint).expect_partial()
        logging.info("Restored from {}".format(latest_checkpoint))
    elif os.path.exists('{}.index'.format(ckpt_path)):
        status = checkpoint.restore(ckpt_path).expect_partial()
        logging.info("Restored from {}".format(ckpt_path))
    else:
        logging.info("Nothing to restore.")

def main(argv):
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    ''' Initialize model '''
    unet = Unet()
    restore(net=unet, ckpt_path=FLAGS.ckpt_path)

    ''' Load data '''
    mix_wav, _ = load(FLAGS.original_wav, sr=SAMPLE_RATE)
    mix_wav_mag, mix_wav_phase = magphase(stft(mix_wav, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH))
    mix_wav_mag= mix_wav_mag[:, START:END]
    mix_wav_phase= mix_wav_phase[:, START:END]

    '''Load gt '''
    if FLAGS.gt == True:
        gt_wav, _ = load(FLAGS.original_gt, sr=SAMPLE_RATE)
        gt_wav_mag, gt_wav_phase = magphase(stft(gt_wav, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH))
        gt_wav_mag= gt_wav_mag[:, START:END]
        gt_wav_phase= gt_wav_phase[:, START:END]

    '''Save input spectrogram image and gt'''
    write_wav(FLAGS.output_dir+'original_mix.wav', 
                istft(mix_wav_mag * mix_wav_phase,win_length=WINDOW_SIZE,hop_length=HOP_LENGTH),
                SAMPLE_RATE, norm=True)
    spectogram_librosa(FLAGS.output_dir+'original_mix.wav',0)
    if FLAGS.gt == True:
        write_wav(FLAGS.output_dir+'gt.wav', 
                    istft(gt_wav_mag * gt_wav_phase,win_length=WINDOW_SIZE,hop_length=HOP_LENGTH),
                    SAMPLE_RATE, norm=True)
        spectogram_librosa(FLAGS.output_dir+'gt.wav',0)

    ''' run data '''
    inputs = mix_wav_mag[1:].reshape(1, 512, 128, 1)
    mask = unet(inputs).numpy().reshape(512, 128)
    predict = inputs.reshape(512, 128)*mask

    ''' evaluation metrics '''
    if FLAGS.gt == True:
        expand_pre = np.expand_dims(predict.flatten(), axis=0)
        expand_gt = np.expand_dims(gt_wav_mag[1:].flatten(), axis=0)
        expand_input = np.expand_dims(inputs.flatten(), axis=0)
        (SDR, SIR, SAR, _) = mir_eval.separation.bss_eval_sources(expand_gt,expand_pre)
        (SDR2, _, _, _) = mir_eval.separation.bss_eval_sources(expand_gt,expand_input)
        NSDR = SDR - SDR2 #SDR(Se, Sr) − SDR(Sm, Sr)

        fout = open(FLAGS.output_dir+'metrics.txt','a')
        print('*****SDR = '+ str(SDR) + ', SIR = '+ str(SIR) + ', SAR = '+ str(SAR) + ', NSDR = '+ str(NSDR) + '*****')
        fout.write('*****SDR = '+ str(SDR) + ', SIR = '+ str(SIR) + ', SAR = '+ str(SAR) + ', NSDR = '+ str(NSDR) + '*****')
        fout.close()

    ''' Convert model output to target magnitude '''
    target_pred_mag = np.vstack((np.zeros((128)), predict))

    ''' Write vocal prediction audio files '''
    write_wav(FLAGS.output_dir+'pred_vocal.wav', 
                istft(target_pred_mag * mix_wav_phase,win_length=WINDOW_SIZE,hop_length=HOP_LENGTH),
                SAMPLE_RATE, norm=True)

    spectogram_librosa(FLAGS.output_dir+'pred_vocal.wav',1)
    
if __name__ == '__main__':
    app.run(main)
