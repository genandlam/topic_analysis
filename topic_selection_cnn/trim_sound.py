#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:13:32 2018

@author: genevieve
"""

import pandas as pd

import os
from pydub import AudioSegment


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
        



         
if __name__ =='__main__':
#    train_dir='/Users/genevieve/Documents/GitHub/topic_analysis/topic_selection_cnn/transcripts_topic.csv'
   
    train_dir='/media/hdd1/genfyp/topic_analysis/transcripts_topic.csv'
    audio_dir='/media/hdd1/genfyp/raw_data/audio/'
#    out_dir = '/media/hdd1/genfyp/raw_data/interim_selected'
    out_dir='/media/hdd1/genfyp/raw_data/selected/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
#    participant_df,participant_id=train_index(train_dir,out_dir)
#    
    train_index(train_dir,out_dir)


   
   
   
   