#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 12:29:20 2018

@author: genevieve
"""

import pandas as pd
import os 
import numpy as np 
#import math





def train_index(train_dir,dev_dir):
     df_train = pd.read_csv(train_dir)
     print(df_train.shape)
     
     
     df_dev = pd.read_csv(dev_dir)
    
     print(df_dev.shape)
     df=pd.concat([df_train,df_dev])
     df = df[["Participant_ID",'Gender']].copy()  
     print(df_dev.shape)
     print(df.shape)
     
     
     
     
     return df

def data_retrieve(working_dir,train_id):
    
    


       
        participants=train_id
        transcripts = pd.DataFrame()
        
        
        

        for index, row in participants.iterrows():
                filename= str(row.Participant_ID)+'_TRANSCRIPT.csv'
                location=os.path.join(working_dir, filename)
                temp = pd.read_csv(location, sep='\t')
                temp['topic']= np.nan  
                temp['topic_value']= np.nan 
                temp['sub_topic']= np.nan 
                temp['participant']=row.Participant_ID
                temp['Gender']=row.Gender
                
                for index, row in temp.iterrows():
                    if row.speaker == 'Ellie':
                      topic = topic_selection(row.value)
                      
                      if topic!=[]:
                       
                        if len(topic)!=1:
                            df_try=row
                            temp.append([df_try]*len(topic),ignore_index=True)
                            print(temp)
                            
                      for words in topic:    
                          temp['topic'][index+1]=words[0]                
                          temp['sub_topic'][index+1]=words[1] 
                          temp['topic_value'][index+1]=row.value
                          
                          if temp['speaker'][index+2]=='Participant':
                              temp['topic'][index+2]=words[0]
                              temp['sub_topic'][index+2]=words[1]
                              temp['topic_value'][index+2]=row.value
                              
                          if temp['speaker'][index+3]=='Participant':
                              temp['topic'][index+3]=words[0]
                              temp['sub_topic'][index+3]=words[1]
                         # print (row.sub_topic)     
#                transcripts.append(temp)  
                     
                temp.dropna(inplace=True)          
                transcripts = pd.concat([transcripts, temp], axis=0)
                   
               
                        
        return transcripts


def topic_selection(transcript):    
    interest=['recently that you really enjoy','traveling','travel alot','family','fun','best friend','weekend']
    sleep=["good night's sleep", "don't sleep well"]
    feeling_depressed=['really happy','behavior',' disturbing thought','feel_lately']
    failure=['regret','guilty','proud','being_parent','best_quality']
    personality=['introvert','shyoutgoing']
    dignose=['ptsd','depression','therapy is useful']
    parent=['hard_parent','best_parent','easy_parent','your_kid','differnet_parent']
    
    
    ques =[interest,sleep,feeling_depressed,failure,personality,dignose,parent]
    
    topic_name=[]
    for topic_count,topic in enumerate(ques):
        for sub_topic_count, sub_topic in enumerate(topic):
#            if type(transcript) == float:
#                    print(topic)
#                    print(topic_count)
#                    print(transcript)
#                    print(sub_topic)
#                    print(math.isnan(transcript))
            if type(transcript) != float:
                
                if sub_topic in transcript: 
                    
                    topic_name.append([topic_count,sub_topic_count])
        
    return topic_name




    
if __name__ == '__main__':

    train_dir='/media/hdd1/genfyp/depression_data/train_split_Depression_AVEC2017.csv'
    dev_dir ='/media/hdd1/genfyp/depression_data/dev_split_Depression_AVEC2017.csv'
    train_id=train_index(train_dir,dev_dir)
    
    working_dir = "/media/hdd1/genfyp/raw_data/transcript_data/"   
    transcripts=data_retrieve(working_dir,train_id)
    
    transcripts.to_csv('transcripts_topic.csv',index=False,sep='\t', encoding='utf-8')
    
#    with open ("sementic.csv",'wb') as myfile:
#        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#                        wr.writerow(sementic_feature)
       
            