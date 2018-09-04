#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 17:56:43 2018

@author: genevieve
"""

import pandas as pd
import numpy as np
import word_dic


class topic_selection(object):
    
     def __init__(self, topic,sub_topic):    
         
        self.topic = topic
        self.sub_topic=sub_topic

        
        
     def train_index(self,working_dir):
    
         df = pd.read_csv(working_dir,sep='\t')
         df.topic=df.topic.astype(int)
         df.sub_topic=df.sub_topic.astype(int)
         df['yes/no']= np.nan 
         
         if self.topic =='sleep':
            
              responds=self.sleep_info(df)     
              
          
        
         if self.topic =='personality':
            
              responds=self.personality_info(df) 
        
         if self.topic =='dignosis':
            
              responds=self.dignosis_info(df)   
             
         return responds
     
     def sleep_info(self,df):
        
         
         for index, row in df.iterrows():
              if row.topic== 1 and row.sub_topic==0:
                  sentence=row.value
                  not_easy,easy=self.dic()
                  if df['participant'][index-1]==row.participant and (df['yes/no'][index-1] == 0 or df['yes/no'][index-1]== 1 ) :
                      df['yes/no'][index]=df['yes/no'][index-1]
            
                  else:
                    
                    for word_not in not_easy:
                        if word_not in sentence: 
                            df['yes/no'][index]=0
                            
                    for word in easy:
                        if word in sentence:
                            df['yes/no'][index]=1
                            
                  if df['participant'][index-1]==row.participant and (df['yes/no'][index+1] != 0 or df['yes/no'][index+1]!= 1 ) :
                          df['yes/no'][index-1]=df['yes/no'][index]
                  if row.participant==423:
                      df['yes/no'][index]=1
                  if row.participant==441:
                      df['yes/no'][index]=0    
                      
            
                      #    sleep=df[(df['topic']== 1)&(df['sub_topic']== 0) & (df['yes/no']!= 1 )&(df['yes/no']!= 0 )] 
         sleep=df[(df['topic']== 1)&(df['sub_topic']== 0)]
    #   sleep.dropna(inplace=True)          
         respond =sleep[["participant",'topic','yes/no','value','topic_value']].copy()
        
         print(respond.shape)
            
         return respond 

     def personality_info(self,df):
        
         
         for index, row in df.iterrows():
              if row.topic== 4:
                  sentence=row.value
                  introvert,not_introvert=self.dic()
                  if index !=0: 
                      if df['participant'][index-1]==row.participant and (df['yes/no'][index-1] == 0 or df['yes/no'][index-1]== 1 ) :
                          df['yes/no'][index]=df['yes/no'][index-1]
                
                      else:
                        
                          for word_not in not_introvert:
                                if word_not in sentence: 
                                    df['yes/no'][index]=0
                                if row.value=='no'or row.value=='no ':
                                     df['yes/no'][index]=0    
                                    
                          for word in introvert:
                                if word in sentence:
                                    df['yes/no'][index]=1
                                    
                  
                  if index !=0:       
                      if df['participant'][index-1]==row.participant and (df['yes/no'][index+1] != 0 or df['yes/no'][index+1]!= 1 ) :
                              df['yes/no'][index-1]=df['yes/no'][index]
                  
                  
                  if row.participant==315:
                      df['yes/no'][index]=0     
                  if row.participant==318:
                      df['yes/no'][index]=1 
                 
              
                 
                 
                  if df['participant'][index]==350:
                      df['yes/no'][index]=0 
                  if df['participant'][index]==353:
                      df['yes/no'][index]=0 
                  if df['participant'][index]==358:
                      df['yes/no'][index]=0
                  if df['participant'][index]==364:
                      df['yes/no'][index]=1
                  if df['participant'][index]==464:
                      df['yes/no'][index]=0
            
            
  #       personality=df[(df['topic']== 4) & (df['yes/no']!= 1 )&(df['yes/no']!= 0 )] 
         personality=df[df['topic']== 4]
         #remove invalid reply or reply with little info
         personality.dropna(inplace=True)          
         respond =personality[["participant",'topic','yes/no','value','topic_value']].copy()
        
         print(respond.shape)
            
         return respond  


     def dignosis_info(self,df):
        
         
         for index, row in df.iterrows():
              if row.topic== 5 & row.sub_topic==0:
                  sentence=row.value
                  introvert,not_introvert=self.dic()
                  if index !=0: 
                      if df['participant'][index-1]==row.participant and (df['yes/no'][index-1] == 0 or df['yes/no'][index-1]== 1 ) :
                          df['yes/no'][index]=df['yes/no'][index-1]
                
                      else:
                        
                          for word_not in not_introvert:
                                if word_not in sentence: 
                                    df['yes/no'][index]=0
                                    
                          for word in introvert:
                                if word in sentence:
                                    df['yes/no'][index]=1
                  
                  if index !=0:       
                      if df['participant'][index-1]==row.participant and (df['yes/no'][index+1] != 0 or df['yes/no'][index+1]!= 1 ) :
                              df['yes/no'][index-1]=df['yes/no'][index]
                              
                  
                  
                
               
                      
            
  #       personality=df[(df['topic']== 4) & (df['yes/no']!= 1 )&(df['yes/no']!= 0 )] 
         personality=df[df['topic']== 5]
         print(df.dtypes)
    #   sleep.dropna(inplace=True)          
         respond =personality[["participant",'topic','yes/no','value','topic_value']].copy()
        
         print(respond.shape)
            
         return respond                                    
        
     def dic(self):
         
         if self.topic =='sleep':
             
             not_easy,easy = word_dic.sleep()
  

             return not_easy,easy
      
         if self.topic=='personality':
             introvert,not_introvert= word_dic.personality()
             return introvert,not_introvert
             
        
        
        
 
             
        
        
        
        
train_dir='transcripts_topic.csv'
#topic='sleep'
#sub_topic="good night's sleep"
#sleep=topic_selection(topic,sub_topic)
#sleep_responds=sleep.train_index(train_dir)

     

#topic='dignose'
#sub_topic="therapy"
#dignose=topic_selection(topic,sub_topic)
#dignose.train_index(train_dir)
#
topic='personality'
sub_topic='null'
personality=topic_selection(topic,sub_topic)
personality_responds=personality.train_index(train_dir)
