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
from topic_analysis import data_retrieve, stop_word,remove_stopwords,lemmatization
import gensim
from gensim.models import ldamodel
from gensim import corpora 
import numpy as np
import pandas as pd


lda_model =   gensim.models.LdaModel.load('lda_model')
id2word =  corpora.Dictionary.load('lda_model.id2word')
#print(type(id2word))



def topic_to_dic(lda_model):
    topic = {}
    for idx in range(lda_model.num_topics):
        tt = lda_model.get_topic_terms(idx,20)
        topic[idx]=([id2word[pair[0]] for pair in tt])
        

#     print model.print_topic(idx)
    return topic    
print(lda_model.print_topics())
working_dir = "/Users/genevieve/Desktop/raw_data/transcript_data"
transcripts,participants= data_retrieve(working_dir)

def clean_text(text):
    
    stop_words=stop_word()
    
    text=remove_stopwords(text,stop_words)
    cleaned_text = lemmatization(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    return cleaned_text
    
for participant in participants:
        for index, row in transcripts[participant].iterrows():
         
            ques_vec = []
            topic_vec = []
            topic_vec = lda_model[ques_vec]
            if row.speaker =='Ellie':    
                text=row.value
                bow = id2word.doc2bow(clean_text(text))
                a=lda_model[bow]
                
                print a[0][1]

               
                
                

     

def loop_transcript():
    d=[]
    #calling another
    working_dir = "/Users/genevieve/Desktop/raw_data/transcript_data"
    transcripts,participants= data_retrieve(working_dir)
    
    for participant in participants:
        for index, row in transcripts[participant].iterrows():
            if row.speaker =='Ellie':
                query=row.value
                query = query.split()
                query = id2word.doc2bow(query)
                for i in query:
                   print(query[i][0])
                   if query[i][0]>0.5:
                       print(query[i][0])
                       d.append(participant)
    return d            

   
#new_topics = new_lda[corpus]
#
#for topic in new_topics:
#
#      print(topic)





# print all topics
#model.show_topics(topics=20, topn=20)



if __name__ == '__main__':
      topic=topic_to_dic()
#    d=loop_transcript()
#    
    