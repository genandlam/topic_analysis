#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 18:54:38 2018

@author: genevieve
"""
import os.path
import os
import pandas as pd
import csv
from topic_analysis import data_retrieve
import gensim
from gensim.models import ldamodel
import numpy as np



model =   gensim.models.LdaModel.load('lda_model')
model.show_topics(topics=20, topn=20)

#data =np.load('lda.model.expElogbeta.npy')
for i in range(0, model.num_topics-1):
    print model.print_topic(i)


def loop_transcript():
    d={}

    working_dir = "/Users/genevieve/Desktop/raw_data/transcript_data"
    transcripts,participants= data_retrieve(working_dir)
    
    return transcripts

   






# print all topics
#model.show_topics(topics=20, topn=20)



if __name__ == '__main__':
    transcripts=loop_transcript()
    
