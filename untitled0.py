#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 20:34:41 2018

@author: genevieve
"""

# Plotting tools

import csv
import gensim
from gensim.models import wrappers
from gensim.models.wrappers import LdaMallet
from gensim import corpora

mallet_path = '../Documents/mallet-2.0.8/bin/mallet' # update this path
   
    
with open("corpus.csv", 'rb') as f:
    reader = csv.reader(f)
    corpus = list(reader)

with open("data_lemmatized.csv", 'rb') as f:
    reader = csv.reader(f)
    data_lemmatized = list(reader)

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)


#reader = csv.reader(open('id2word.csv', 'rb'))
#reader = csv.DictReader(open('id2word.csv', 'rb'))
#id2word = []
#for line in reader:
#  id2word.append(line)
#
#new_dict = {}
#for item in id2word:
#  print item



#with open("id2word.csv", 'rb') as f:
#    reader = csv.reader(f)
#    id2word= list(reader)    
  
ldamallet = LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)