#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:13:32 2018

@author: genevieve
"""

import pandas as pd
import numpy as np
from random import sample,randrange
from spectrogram_dicts import get_depression_label
import os
from pydub import AudioSegment
from itertools import combinations

def train_index(working_dir,out_dir):    
         df = pd.read_csv(working_dir,sep='\t')
         participant_id=df.participant.unique()
         participants=participant_id.tolist()
         path='/media/hdd1/genfyp/raw_data/audio/'
#         participant_df=df[(df['participant']== 321)]
         for participant_id in participants:
             participant_df=df[(df['participant']== participant_id)]
             print(participant_id)
             
             combined=segment_want(participant_df,participant_id,path)
             filename=str(participant_id)+'_AUDIO.wav'
#             partic_id =  str(participant_id)
 #            filename=str(participant_id)+'_AUDIO.wav'
#             participant_dir = os.path.join(out_dir, partic_id)
#             if not os.path.exists(participant_dir):
#                os.makedirs(participant_dir)
#             print(participant_dir)
#             combined.export(os.path.join(participant_dir, filename),format="wav")
             combined.export(os.path.join(out_dir, filename),format="wav")
         
             
#         return participant_df
def segment_want(df,participant_id,path):
    combined = AudioSegment.empty()
   
    for index, row in df.iterrows():
        t1=row.start_time
        t2=row.stop_time
    # TIMESTAMP IS IN SECONDS DF 
        t1 = t1 * 1000 #Works in milliseconds
        t2 = t2 * 1000 
        file_name=str(participant_id)+"_AUDIO.wav"
        
        newAudio = AudioSegment.from_wav(os.path.join(path, file_name))
        newAudio = newAudio[t1:t2]
        
             
        combined+=newAudio
        
#        file_name='./audio/491_audio'+str(index)+'.wav'
#        newAudio.export(file_name, format="wav") #Exports to a wav file in the current path.
   # combined.export('491_combine.wav', format="wav")
    return combined
        
def data_aug(trans_dir,audio_dir):
    count=0
    min_topic=5
    combined = AudioSegment.empty()
    df_topic = pd.read_csv(trans_dir,sep='\t')
    
    participants=df_topic.participant.unique()
    
    for participant_id in participants:
        
        participant_df=df_topic[(df_topic['participant']== participant_id)]
        topic=participant_df['topic'].unique()
        
        
        topic_dic=topic_start_end(participant_df,topic)
        
        if len(topic) >= min_topic:
            
            depressed = get_depression_label(participant_id) # 1 if True
            
            if depressed:
                 t_lens = np.random.randint(low=min_topic, 
                                       size=10)
                 for t_len in t_lens:
                
                    # array of topics selected for augmentation
                    combs = list(combinations(list(topic_dic.values()), t_len))
                    # Select a random combination
                    t_comb = list(combs[np.random.randint(len(combs))])
                    print(t_comb)
                    # Shuffle the topic texts in selected combination
                    np.random.shuffle(t_comb)
                    # Select a random combination
                    
#                    for index, row in topic_segment.iterrows():
#                        
#                        
#                        new_par_file=str(participant_id)+"_"+str(count)+"_AUDIO.wav"
#                        out_dir ='/media/hdd1/genfyp/raw_data/data_aug'
#                        newAudio.export(os.path.join(out_dir, new_par_file), format="wav")
#                        count+=1
#             
        
def topic_start_end(participant_df,topics):
    topic_dic={}
    topics=topics.astype(float).tolist()
  
    for topic in topics:
        topic_segment=participant_df[(participant_df['topic']== topic)]
        
        time_stap= topic_segment[['start_time','stop_time']]
        
        time_stap = time_stap.values.tolist()
        print(time_stap)
        topic_dic[str(topic)] = time_stap
    return topic_dic
    
    
    


         
if __name__ =='__main__':
#    train_dir='/Users/genevieve/Documents/GitHub/topic_analysis/topic_selection_cnn/transcripts_topic.csv'
#    out_dir = '/media/hdd1/genfyp/raw_data/interim_selected'   
#    participant_df,participant_id=train_index(train_dir,out_dir)
    train_dir='/media/hdd1/genfyp/topic_analysis/transcripts_topic.csv'
    audio_dir='/media/hdd1/genfyp/raw_data/audio/'

    out_dir='/media/hdd1/genfyp/raw_data/selected/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
 
#    train_index(train_dir,out_dir)
    data_aug(train_dir,audio_dir)


   
   
   
   