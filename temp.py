#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 11:19:04 2018

@author: genevieve
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 17:56:43 2018

@author: genevieve
"""

import pandas as pd
import numpy as np

def train_index(working_dir):
    
     df = pd.read_csv(working_dir,sep='\t')
     df.topic=df.topic.astype(int)
     df.sub_topic=df.sub_topic.astype(int)
     df['yes/no']= np.nan  
     
     return df 
     
def acess_info(df):   
    
    for index, row in df.iterrows():
        if row.topic== 1 and row.sub_topic==0:
            sentence=row.value
            not_easy,easy=dic()
            if df['participant'][index-1]==row.participant and (df['yes/no'][index-1] == 0 or df['yes/no'][index-1]== 1 ) :
                      df['yes/no'][index]=df['yes/no'][index-1]
            
            else:
                
                for word_not in not_easy:
                    if word_not in sentence: 
                        df['yes/no'][index]=0
                        
                for word in easy:
                    if word in sentence:
                        df['yes/no'][index]=1
                        
            if df['participant'][index-1]==row.participant and (df['yes/no'][index-1] != 0 or df['yes/no'][index+1]!= 1 ) :
                      df['yes/no'][index-1]=df['yes/no'][index]
                      
                  
#    sleep=df[(df['topic']== 1)&(df['sub_topic']== 0) & (df['yes/no']!= 1 )&(df['yes/no']!= 0 )] 
    sleep=df[(df['topic']== 1)&(df['sub_topic']== 0)]
#    sleep.dropna(inplace=True)          
    respond =sleep[["participant",'topic','yes/no','value','topic_value']].copy()
    
    print(respond.shape)
        
    return respond 
            
    
def dic():
    
   not_easy, easy= word_dic.sleep()
  

   return not_easy, easy


if __name__ == '__main__':

    train_dir='transcripts_topic.csv'
    df=train_index(train_dir)
    respond =acess_info(df)