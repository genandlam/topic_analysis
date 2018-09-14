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
            
             respond_depression, respond_ptsd  =self.dignosis_info(df) 
             return respond_depression, respond_ptsd 
         
         if self.topic =='therapy':
            
             responds=self.therapy_info(df) 
         
         if self.topic =='emotion':
            
             responds=self.emotion_info(df)
        
         if self.topic =='behaviour':
            
             respond_behaviour,respond_thoughts=self.change_info(df)
             return respond_behaviour,respond_thoughts
             
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
                            df['yes/no'][index]=1
                            
                    for word in easy:
                        if word in sentence:
                            df['yes/no'][index]=0
                            
                  if df['participant'][index-1]==row.participant and (df['yes/no'][index+1] != 0 or df['yes/no'][index+1]!= 1 ) :
                          df['yes/no'][index-1]=df['yes/no'][index]
                  if row.participant==423:
                      df['yes/no'][index]=0
                  if row.participant==441:
                      df['yes/no'][index]=1    
                      
            
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
                  
                  for word_not in not_introvert:
                        if word_not in sentence: 
                            df['yes/no'][index]=0
                         
                            
                  for word in introvert:
                        if word in sentence:
                            df['yes/no'][index]=1 
                  if row.value=='no'or row.value=='no ':
                             df['yes/no'][index]=0            
                  if index !=0:
                      if df['participant'][index-1]==row.participant and (df['yes/no'][index-1] == 0 or df['yes/no'][index-1]== 1 ) :
                          df['yes/no'][index]=df['yes/no'][index-1]
                          
                      if df['participant'][index-1]==row.participant and (df['yes/no'][index+1] != 0 or df['yes/no'][index+1]!= 1 ) :
                              df['yes/no'][index-1]=df['yes/no'][index]
                

                  if row.participant==315:
                      df['yes/no'][index]=0     
                  if row.participant==318:
                      df['yes/no'][index]=1 
                  if row.participant==350:
                      df['yes/no'][index]=0 
                  if row.participant==353:
                      df['yes/no'][index]=0 
                  if row.participant==358:
                      df['yes/no'][index]=0
                  if row.participant==364:
                      df['yes/no'][index]=1
                  if row.participant==464:
                      df['yes/no'][index]=0
            
            

         personality=df[df['topic']== 4]
         #remove invalid reply or reply with little info
         personality.dropna(inplace=True)          
         respond =personality[["participant",'topic','yes/no','value','topic_value']].copy()
        
         print(respond.shape)
            
         return respond  

     def dignosis_info(self,df):
        
         
         for index, row in df.iterrows():
              if row.topic== 5 and row.sub_topic!=2:
                  sentence=row.value
                  illness, no_illness=self.dic()
                  
                  for word_not in no_illness:
                        if word_not in sentence: 
                            df['yes/no'][index]=0
                         
                            
                  for word in illness:
                        if word in sentence:
                            df['yes/no'][index]=1 
                            
                  if row.value=='no'or row.value=='no 'or row.value=='nah':
                             df['yes/no'][index]=0        
                  if row.value=='i have 'or row.value=='i have':
                             df['yes/no'][index]=1       
#                  if index !=0:
#                      if df['participant'][index-1]==row.participant and (df['yes/no'][index-1] == 0 or df['yes/no'][index-1]== 1 ) :
#                          df['yes/no'][index]=df['yes/no'][index-1]
#                          
#                      if df['participant'][index-1]==row.participant and (df['yes/no'][index+1] != 0 or df['yes/no'][index+1]!= 1 ) :
#                              df['yes/no'][index-1]=df['yes/no'][index]
                 
         dignosis_depression=df[(df['topic']== 5)&(df['sub_topic']== 1)]
         dignosis_ptsd=df[(df['topic']== 5)&(df['sub_topic']== 0)]
         #take note depression
         #333, 352, 423, 478
         #ptsd
         #423, 457
         #remove invalid reply or reply with little info
         dignosis_ptsd.dropna(inplace=True)          
         respond_depression =dignosis_depression[["participant",'topic','yes/no','value','topic_value']].copy()
         respond_ptsd =dignosis_ptsd[["participant",'topic','yes/no','value','topic_value']].copy()
        
         
            
         return respond_depression, respond_ptsd 
                  
     def therapy_info(self,df):
        
         
         for index, row in df.iterrows():
              if row.topic== 5 and row.sub_topic==2:
                  sentence=row.value
                  useful,useless=self.dic()
                  
                  for word_not in useful:
                        if word_not in sentence: 
                            df['yes/no'][index]=0
                         
                            
                  for word in useless:
                        if word in sentence:
                            df['yes/no'][index]=1 
                            
                  if row.value=='no'or row.value=='no 'or row.value=='nah':
                             df['yes/no'][index]=1        
                      
                             df['yes/no'][index-1]=df['yes/no'][index]
                  if row.participant==364:
                      df['yes/no'][index]=0
         therapy=df[(df['topic']== 5)&(df['sub_topic']== 2)]
        
        
         #remove invalid reply or reply with little info
         therapy.dropna(inplace=True)          
         respond =therapy[["participant",'topic','yes/no','value','topic_value']].copy()
         
         return respond
        
     def emotion_info(self,df):
         
          for index, row in df.iterrows():
              if row.topic== 2 and ( row.sub_topic==3 ):
                  sentence=row.value
                  alright,depressed=self.dic()
                  
                  for word_not in alright:
                        if word_not in sentence: 
                            df['yes/no'][index]=0
                         
                            
                  for word in depressed:
                        if word in sentence:
                            df['yes/no'][index]=1 
                            
                  if row.value=='no'or row.value=='no 'or row.value=='nah':
                             df['yes/no'][index]=0        
                  if row.value=='i have 'or row.value=='i have':
                             df['yes/no'][index]=1       
                          
                 
                 
          feeling_lately=df[(df['topic']== 2)&(df['sub_topic']== 3)]
 #         happy_recently=df[(df['topic']== 2)&(df['sub_topic']== 0)]
         
#         dignosis_ptsd.dropna(inplace=True)          
          respond_feeling =feeling_lately[["participant",'topic','yes/no','value','topic_value']].copy()
#          respond_happy=happy_recently[["participant",'topic','yes/no','value','topic_value']].copy()
        
          return respond_feeling
      
        
     def change_info(self,df):
        
         
         for index, row in df.iterrows():
              if row.topic== 2 and (row.sub_topic==2 or row.sub_topic==1 ):
                  sentence=row.value
                  change,no_change=self.dic()
                  
                  for word_not in no_change:
                        if word_not in sentence: 
                            df['yes/no'][index]=0
                         
                            
                  for word in change:
                        if word in sentence:
                            df['yes/no'][index]=1 
                            
                  if row.value=='no'or row.value=='no 'or row.value=='nah':
                             df['yes/no'][index]=0        
                  if row.value=='i have 'or row.value=='i have':
                             df['yes/no'][index]=1       
         
         dignosis_depression=df[(df['topic']== 2)&(df['sub_topic']== 1)]
         dignosis_ptsd=df[(df['topic']== 2)&(df['sub_topic']== 2)]
         
         #remove invalid reply or reply with little info
#         dignosis_ptsd.dropna(inplace=True)          
         respond_depression =dignosis_depression[["participant",'topic','yes/no','value','topic_value']].copy()
         respond_ptsd =dignosis_ptsd[["participant",'topic','yes/no','value','topic_value']].copy()
        
         
            
         return respond_depression, respond_ptsd  
    
     def dic(self):
         
         if self.topic =='sleep':
             
             not_easy,easy = word_dic.sleep()
             
             return not_easy,easy
      
         if self.topic=='personality':
             
             introvert,not_introvert= word_dic.personality()
             
             return introvert,not_introvert
             
         if self.topic=='dignosis':
             
             illness, no_illness= word_dic.illness()
             
             return illness, no_illness
        
         if self.topic=='therapy':
             
             useful,useless= word_dic.therapy()
             
             return useful,useless
         
         if self.topic=='emotion':
             
             alright,depressed= word_dic.emotion()
             
             return alright,depressed
        
         if self.topic=='behaviour':
             
             change,no_change= word_dic.behaviour()
             
             return change,no_change
             
        
        
        
        
train_dir='transcripts_topic.csv'
#topic='sleep'
#sub_topic="good night's sleep"
#sleep=topic_selection(topic,sub_topic)
#sleep_responds=sleep.train_index(train_dir)

 
#topic='personality'
#sub_topic='null'
#personality=topic_selection(topic,sub_topic)
#personality_responds=personality.train_index(train_dir)    

#topic='dignosis'
#sub_topic="null"
#dignosis=topic_selection(topic,sub_topic)
#respond_depression, respond_ptsd  =dignosis.train_index(train_dir)

#topic='therapy'
#sub_topic="therapy"
#therapy=topic_selection(topic,sub_topic)
#therapy_respond=therapy.train_index(train_dir)

#topic='emotion'
#sub_topic="emotion"
#feeling=topic_selection(topic,sub_topic)
#feeling_respond=feeling.train_index(train_dir)

topic='behaviour'
sub_topic="behaviour"
behaviour=topic_selection(topic,sub_topic)
respond_behaviour,respond_thoughts=behaviour.train_index(train_dir)

