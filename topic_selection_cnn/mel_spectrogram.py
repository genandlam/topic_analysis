#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 20:31:17 2018

@author: genevieve
"""
# Packages we're using
import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import pylab
import librosa
import librosa.display
import numpy as np

#fft size / sampling rate = __hz (resolution of each spectral line)
def mel_spec(working_dir):
    
    y, sr = librosa.load(working_dir)       
    
#    S = librosa.feature.melspectrogram(y=y, sr=44100)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S =  librosa.amplitude_to_db(S, ref=np.max)
    t_mel=np.transpose(log_S)
    
#    print(np.transpose(log_S).shape)
    #   librosa.power_to_db(S, ref=np.max)
 #  librosa.logamplitude(S, ref_power=np.max)

    return t_mel

def save_image(log_S,png_name):
    
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    librosa.display.specshow(log_S )
    pylab.savefig(png_name, bbox_inches=None, pad_inches=0)
    pylab.close()
    
if __name__ == '__main__':
# directory containing participant folders with segmented wav files
    dir_name = '/media/hdd1/genfyp/raw_data/interim_selected/'
    
    # walks through wav files in dir_name and creates pngs of the spectrograms.
    # This is a visual representation of what is passed to the CNN before
    # normalization, although a cropped matrix representation is actually
    # passed.
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith('no_silence.wav'):
                partic_id = int(file.split('_')[0][1:])
                wav_file = os.path.join(subdir, file)
                  # make pictures name 
                png_name = subdir + '/' + str(partic_id) + '.png'
                print('Processing ' + file + '...')
                log_S  = mel_spec(wav_file )
                save_image(log_S,png_name)
                    
            