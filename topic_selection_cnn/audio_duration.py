#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:10:44 2018

@author: genevieve
"""

import wave
import os
import contextlib

def audio_duration(filename):
   
    with contextlib.closing(wave.open(filename,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        print(duration)


if __name__ == "__main__":
    dir_name='/media/hdd1/genfyp/raw_data/selected/'
    for file in os.listdir(dir_name):
            if file.endswith('.wav'):
                filename = os.path.join(dir_name, file)
                print(file)
                audio_duration(filename)
                