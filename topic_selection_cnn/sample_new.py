#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 21:34:26 2018

@author: genevieve
"""

import numpy as np
import pandas as pd
import os
import random
from sklearn.utils import resample
from spectrogram_dicts import build_class_dictionaries
np.random.seed(15)  # for reproducibility


def rand_samp_train_test_split():
    """
    Utilizes the above function to return two dictionaries, depressed
    and normal. Each dictionary has only participants in the specific class,
    with participant ids as key, a values of a list of the cropped samples
    from the spectrogram matrices. The lists are vary in length depending
    on the length of the interview clip. The entries within the list are
    numpy arrays with dimennsion (513, 125).
    """
    # build dictionaries of participants and segmented audio matrix
    depressed_dict, normal_dict = build_class_dictionaries('/media/hdd1/genfyp/raw_data/interim_selected/')
    scores_train = pd.read_csv('/media/hdd1/genfyp/depression_data/train_split_Depression_AVEC2017.csv')
#    scores_test = pd.read_csv('/media/hdd1/genfyp/depression_data/dev_split_Depression_AVEC2017.csv')
  
    
    train_samples_dep = []
    train_samples_nor = []
    train_labels_dep = []
    train_labels_nor = []
    
    test_samples= []
    test_labels = []
  
    
    # Depressed participants
    for key, spect in depressed_dict.iteritems():
        if key in set(scores_train["Participant_ID"].values):
 #           print("yes")
            train_samples_dep.append(spect)
            train_labels_dep.append(1)
#        if key in set(scores_test["Participant_ID"].values):
        else:
#            print("no")
            test_samples.append(spect)
            test_labels.append(1)
    
    # Normal participants
    for key, spect in normal_dict.iteritems():
#        if key in scores_train["Participant_ID"]:
        if key in set(scores_train["Participant_ID"].values):    
            train_samples_nor.append(spect)
            train_labels_nor.append(0)
        else:
            test_samples.append(spect)
            test_labels.append(0)
    # Sampling
    max_samples = min(len(train_samples_dep), len(train_samples_nor))  
    print(max_samples)
#    print(train_samples_dep)
#    print(train_labels_dep )
 #   d = {'col1': [1, 2], 'col2': [3, 4]}
#    train_dep={'samples':train_samples_dep,'labels':train_labels_dep}
#    train_dep= pd.DataFrame(data=train_dep)
    
    train_samples_dep, train_labels_dep = resample(train_samples_dep, train_labels_dep, n_samples=max_samples, random_state=15)
    train_samples_nor, train_labels_nor = resample(train_samples_nor, train_labels_nor, n_samples=max_samples, random_state=15)

    
   
    # Create final arrays
    train_samples = np.concatenate((train_samples_dep, train_samples_nor),  axis=0)
    train_labels = np.concatenate((train_labels_dep, train_labels_nor),axis=0)
   
    return train_samples, train_labels, test_samples, test_labels
#    
#    for key, _ in depressed_dict.iteritems():
#        path = '/media/hdd1/genfyp/processed/'
#        filename = 'D{}.npz'.format(key)
#        outfile = path + filename
#        np.savez(outfile, *depressed_dict[key])
#    
#    for key, _ in normal_dict.iteritems():
#        path = '/media/hdd1/genfyp/processed/'
#        filename = '/N{}.npz'.format(key)
#        outfile = path + filename
#        np.savez(outfile, *normal_dict[key])



if __name__ == '__main__':
    # build participant's cropped npz files
    # this is of the whole no_silence particpant's no_silence interview,
    # but each array in the npz files was width of crop_width
    

    # random sample from particpants npz files to ensure class balance
    train_samples, train_labels, test_samples, \
        test_labels = rand_samp_train_test_split()

    # save as npz locally
    print("Saving npz file locally...")
    np.savez('/media/hdd1/genfyp/processed/train_samples.npz', train_samples)
    np.savez('/media/hdd1/genfyp/processed/train_labels.npz', train_labels)
    np.savez('/media/hdd1/genfyp/processed/test_samples.npz', test_samples)
    np.savez('/media/hdd1/genfyp/processed/test_labels.npz', test_labels)

