#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:55:02 2018

@author: genevieve
"""
import pandas as pd
import os.path
import os
import numpy as np
import csv

def data_retrieve(working_dir ):
    for root, dirs, files in os.walk(working_dir):
        file_list = []
        participants= []
        transcripts = {}
        
        for filename in files:
            if filename.endswith('_TRANSCRIPT.csv'):
               participant_id = int(filename.split('_')[0][0:3])
               if(participant_id != 342 | participant_id !=394 | participant_id !=398 | participant_id !=460  ):
                   participants.append(participant_id )
                   file_list.append(os.path.join(root, filename)) 
                    
        for file in file_list:
            for participant in participants:
                transcripts[participant] = pd.read_csv(file.format(participant), sep='\t')
                transcripts[participant]['topic']= np.nan  
                
    return transcripts,participants

    
def convert_df(transcripts,participants):
  
    e_list = []
    
    for participant in participants:
      for index, row in transcripts[participant].iterrows():
        if (row.speaker =='Ellie' and (row.value !="hi i'm ellie thanks for coming in today" or
            row.value !="i was created to talk to people in a safe and secure environment" or
           row.value !="i'm not a therapist but i'm here to learn about people and would love to learn about you" or
           row.value !="i'll ask a few questions to get us started" or
           row.value !="and please feel free to tell me anything your answers are totally confidential " or
           row.value !="i don't judge i can't i'm a computer" or
           row.value !='think of me as a friend')):
           e_list.append(row.value)
    
    e_list=list(set(e_list))
    print(len(e_list))
    e_list = [x for x in e_list if str(x) != 'nan'] 
#    e_list.remove(np.nan)
    
    # removing single word or 2 word sentences     
    for sentence in e_list:
      if len(sentence.split())== 1 or len(sentence.split())== 2 :
        e_list.remove(sentence)
        
    print (len(e_list))
    
    return e_list 


if __name__ == '__main__':
    
    working_dir = "/media/hdd1/genfyp/raw_data/transcript_data/"

    transcripts,participants =data_retrieve(working_dir)
    #list ofEllie dictionary of all the unique sentences Ellie said 
    e_list=convert_df(transcripts,participants)

    with open("e_list.csv", 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(e_list)
  
   
            
            
            
            
            
 