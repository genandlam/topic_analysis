#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 17:37:42 2018

@author: genevieve
"""

def sleep():
    
      not_easy=["don't",'difficult','hard','tough','trouble','troublesome','problem','problematic','challenging','burdensome',
              'not easy',"haven't",'seldom','awful','not that easy','impossible','affect','never easy','insomniac',"it's not",
              'not too easy','wake up',"can't sleep",'waking up','not really',"couldn't sleep",'really bad','not very easy']
      easy=['pretty easy','very easy','good sleep','pretty good','fairly easy','no problem','really easy','sometimes','somewhat',
          'depend','extremely easy','sleep well','yeah','reasonably','heavy sleeper']
      return not_easy,easy
  
def personality():   
    intro= ['yes','yeah','more shy','pretty shy','probably shy','guess so','fairly','kind of shy','most part shy','i am shy']
    not_intro=["i'm not",'perhaps','maybe not','somewhat','both','depends','outgoing','middle','more outgoing',
               'pretty outgoing','mixture','at times','extrovert','occassional','mm no','definitely not shy']
    
    return intro,not_intro

def illness():
    illness=['yes','I have depression','yes i have','i am dignosed with',"i'm dignosed with",'yeah','dignosed with depression',]
    not_illness=['do not have','I am not','nope','never' ,'i have not',"i haven't",'no i have not',"no i haven't","no i have not",
                 "no i haven't","i don't think ",'no not at all','no not','um no','no <laughter>','no i','no mm',
                 'diagnosed by a doctor no','uh no','have not']
    
    return illness,not_illness



def therapy():
    useful=['useful','yes','yeah','i think so','miss it','really useful','really useful','i do ']
    useless=['useless',"nope",'no use',"i don't",'i do not','never','deceitful']
    
    
    return useful,useless


def emotion():
    alright=['alright','happy','great','pretty good','fine','okay','pretty great','good','optimistic','positive',' calm',
             'better','feeling well','relatively well','so so']
    depressed=['bad','depressed','disatisfied','bad thoughts','horrific','feeling down','hopeless','upsad','pessimistic','blue',
                'low-spirited','dispirited','down and out','dispirited','tired','low energy','stressed out']
    
    
    return alright,depressed

def behaviour():
    yes=['yes','yeah','hopelessness','overeat','sleep alot','not positive','no interest','poor appetite','no appetite','depress',
         'sleep a lot more']
    no=['no not','have not','not lately','normal','not really',"i don't",'i have not','not lately','no no','very seldom']
    
    return yes,no
        
    
    return 
    