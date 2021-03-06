#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:07:30 2018

@author: genevieve
"""
import os.path
import os
import re
import numpy as np
import pandas as pd
from pprint import pprint
import csv

# Gensim

from gensim import corpora 
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.models import ldamodel


# NLTK Stop words
#from nltk.corpus import stopwords
import nltk 
#nltk.download()  
from nltk.corpus import stopwords

    
# spacy for lemmatization
import spacy

# Plotting tools
#import pyLDAvis
#import pyLDAvis.gensim  # don't skip this
#import matplotlib.pyplot as plt
#%matplotlib inline


import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

def train_index(working_dir):
     df = pd.read_csv(working_dir)
     train_id =df["Participant_ID"].tolist() 
     
     
     return train_id

def data_retrieve(working_dir,train_id):
    
    
#    for root, dirs, files in os.walk(working_dir):
#        file_list = []
        participants= []
        transcripts = {}
        participants=train_id
        
#        for filename in files:
#            if filename.endswith('_TRANSCRIPT.csv'):
#               participant_id = int(filename.split('_')[0][0:3]) 
#               
#               participants.append(participant_id )
#               file_list.append(os.path.join(root, filename)) 
                    
#        for file in file_list:
        for participant in participants:
                filename= str(participant)+'_TRANSCRIPT.csv'
                location=os.path.join(working_dir, filename)
                transcripts[participant] = pd.read_csv(location, sep='\t')
#                transcripts[participant]['topic']= np.nan  
                
        return transcripts,participants

    
def convert_df(transcripts,participants):
  
    e_list = []
   
    
    for participant in participants:
      for index, row in transcripts[participant].iterrows():
        
#        if (row.speaker =='Ellie' and (row.value !="hi i'm ellie thanks for coming in today" or
#           row.value !="i was created to talk to people in a safe and secure environment" or
#           row.value !="i'm not a therapist but i'm here to learn about people and would love to learn about you" or
#           row.value !="i'll ask a few questions to get us started" or
#           row.value !="and please feel free to tell me anything your answers are totally confidential " or
#           row.value !="i don't judge i can't i'm a computer" or
#           row.value !='think of me as a friend' or 
#           row.value !="IntroV4Confirmation (hi i'm ellie thanks for coming in today i was created to talk to people in a safe and secure environment i'm not a therapist but i'm here to learn about people and would love to learn about you i'll ask a few questions to get us started and please feel free to tell me anything your answers are totally confidential are you ok with this)" or
#           row.value !="i'm here to learn about people and would love to learn about you i'll ask a few questions to get us started and please feel free to tell me anything your answers are totally confidential" or
#           row.value !="think of me as a friend i don't judge i can't i'm a computer" or
#           row.value !="hi i'm ellie thanks for coming in today i was created to talk to people in a safe and secure environment" or 
#           row.value !='i love listening to people talk ' or 
#           row.value !="i'm not a therapist but i'm here to learn about people and would love to learn about you i'll ask a few questions to get us started" or 
#           row.value !='and please feel free to tell me anything your answers are totally confidential are you okay with this ')):
#            e_list.append(row.value)
          if (row.speaker =='Ellie'):
              e_list.append(row.value)
    e_list=list(set(e_list))
    print(len(e_list))
   
            #del(e_list[row_num])
   
    # Remove IntroV4Confirmation
#    e_list = [i for i in e_list if i[0]!='i love listening to people talk ']
#    e_list = [i for i in e_list if i[0]!='uh_huh (uh huh)']
    
    y=e_list.index("IntroV4Confirmation (hi i'm ellie thanks for coming in today i was created to talk to people in a safe and secure environment i'm not a therapist but i'm here to learn about people and would love to learn about you i'll ask a few questions to get us started and please feel free to tell me anything your answers are totally confidential are you ok with this)")
    del(e_list[y])
    y=e_list.index("i'm not a therapist but i'm here to learn about people and would love to learn about you i'll ask a few questions to get us started")
    del(e_list[y]) 
    y=e_list.index("i'm not a therapist but i'm here to learn about people and would love to learn about you i'll ask a few questions to get us started ")
    del(e_list[y]) 
    y=e_list.index("i love listening to people talk ")
    del(e_list[y])
    y=e_list.index('uh_huh (uh huh)')
    del(e_list[y])
    y=e_list.index('and please feel free to tell me anything your answers are totally confidential ')
    del(e_list[y])
    y=e_list.index('and please feel free to tell me anything you answers are totally confidential')
    del(e_list[y])
    y=e_list.index("i'm here to learn about people and would love to learn about you i'll ask a few questions to get us started and please feel free to tell me anything your answers are totally confidential ")
    del(e_list[y])
    y=e_list.index("hi i'm ellie thanks for coming in today i was created to talk to people in a safe and secure environment think of me as a friend i don't judge i can't i'm a computer  ")
    del(e_list[y])
    y=e_list.index("hi i'm ellie thanks for coming in today i was created to talk to people in a safe and secure environment think of me as a friend i don't judge i can't i'm a computer")
    del(e_list[y])
    y=e_list.index("and please feel free to tell me anything you're answers are totally confidential")
    del(e_list[y])
    y=e_list.index('and please feel free to tell me anything your answers are totally confidential are you okay with this ')
    del(e_list[y])
    y=e_list.index("i''m here to learn about people and would love to learn about you i'll ask a few questions to get us started and please feel free to tell me anything your answers are totally confidential")
    del(e_list[y])
    y=e_list.index("hi i'm ellie thanks for coming in today i was created to talk to people in a safe and secure environment think of me as a friend i don't judge i can't i'm a computer i'm here to learn about people and would love to learn about you i'll ask a few questions to get us started and please feel free to tell me anything your answers are totally confidential ")
    del(e_list[y])
    y=e_list.index("i'm not a therapist but i'm here to learn about people and would love to learn about you i'll ask a few questions to get us started   ")
    del(e_list[y])
    y=e_list.index("i'm not a therapist but i'm here to learn about people and would love to learn about you")
    del(e_list[y]) 
    y=e_list.index("i'm not a therapist but i'm here to learn about people and would love to learn about you ")
    del(e_list[y])
    y=e_list.index("hi i'm ellie thanks for coming in today " )
    del(e_list[y])
    y=e_list.index('thanks it was great chatting with you')
    del(e_list[y])
    y=e_list.index('thanks for sharing your thoughts with me ')
    del(e_list[y])
    y=e_list.index("okay i think i've asked everything i need to thanks for sharing your thoughts with me ")
    del(e_list[y])
    y=e_list.index("i'm great thanks")
    del(e_list[y])
    y=e_list.index("great_thanks (i'm great thanks)")
    del(e_list[y])
    y=e_list.index('i was created to talk to people in a safe and secure environment ')
    del(e_list[y])
    y=e_list.index('i was created to talk to people in a safe and secure environment')
    del(e_list[y])
    y=e_list.index("i'm sorry ")
    del(e_list[y])
    y=e_list.index('appreciate_open (thanks for sharing your thoughts with me)')
    del(e_list[y])
    y=e_list.index('that_sucks (that sucks)')
    del(e_list[y])
    y=e_list.index("i'm sorry to hear that ")
    del(e_list[y])
    y=e_list.index("sorry_hear (i'm sorry to hear that)")
    del(e_list[y])
    y=e_list.index('thanks for sharing your thoughts with me')
    del(e_list[y])
    y=e_list.index("hi i'm ellie thanks for coming in today i was created to talk to people in a safe and secure environment " )
    del(e_list[y])
    y=e_list.index("Ellie17Dec2012_03 (that's good)")
    del(e_list[y])
    y=e_list.index('thank_you (thank you)')
    del(e_list[y])
    y=e_list.index("thats_good (that's good)")
    del(e_list[y])
    y=e_list.index("hi i'm ellie thanks for coming in today")
    del(e_list[y])
    y=e_list.index('oh my gosh')
    del(e_list[y])
    y=e_list.index("that's so good to hear")
    del(e_list[y])
    y=e_list.index('right2 (right)')
    del(e_list[y])
    y=e_list.index("good_hear (that's so good to hear)")
    del(e_list[y])
    y=e_list.index("that's so good to hear ")
    del(e_list[y])
    y=e_list.index("i'd love to hear all about it ")
    del(e_list[y])
    y=e_list.index("i'd love to hear all about it")
    del(e_list[y])
    y=e_list.index("yeah i'm sorry to hear that")
    del(e_list[y])
    y=e_list.index('asked_everything (okay i think i have asked everything i need to)')
    del(e_list[y])
    y=e_list.index("okay i think i've asked everything i need to ")
    del(e_list[y])
    y=e_list.index("okay i think i've asked everything i need to")
    del(e_list[y])
    y=e_list.index('oh_no (oh no)')
    del(e_list[y])
    y=e_list.index('uh_oh (uh oh)')
    del(e_list[y])
    y=e_list.index("i'll ask a few questions to get us started")
    del(e_list[y])
    y=e_list.index('that makes sense')
    del(e_list[y])
    y=e_list.index('that makes sense ')
    del(e_list[y])
    y=e_list.index('makes_sense (that makes sense)')
    del(e_list[y])
    y=e_list.index('whatever comes to your mind ')
    del(e_list[y])
    y=e_list.index('mind (whatever comes to your mind)')
    del(e_list[y])
    y=e_list.index("back_later (let's come back to that later)")
    del(e_list[y])
    y=e_list.index("let's come back to that later")
    del(e_list[y])
    y=e_list.index('isee_downer (i see)')
    del(e_list[y])
    y=e_list.index('and please feel free to tell me anything your answers are totally confidential')
    del(e_list[y])
    y=e_list.index("yeah3 (yeah)")
    del(e_list[y])
    y=e_list.index("i'm sorry")
    del(e_list[y])
    y=e_list.index("oh")
    del(e_list[y])
    y=e_list.index("Ellie17Dec2012_02 (uh huh)")
    del(e_list[y])
    y=e_list.index("im_sorry (i'm sorry)")
    del(e_list[y])
    y=e_list.index('yeah i see what you mean ')
    del(e_list[y])
    y=e_list.index('yeah i see what you mean')
    del(e_list[y])
    y=e_list.index('that sounds really hard')
    del(e_list[y])
    y=e_list.index('wow (wow)')
    del(e_list[y])
    y=e_list.index('yeah that sounds really hard')
    del(e_list[y])
    y=e_list.index('yeah that sounds really hard ')
    del(e_list[y])
    y=e_list.index('yeah how hard is that')
    del(e_list[y])
    y=e_list.index('yeah how hard is that ')
    del(e_list[y])
    y=e_list.index("thats_great (that's great)")
    del(e_list[y])
    y=e_list.index("Ellie17Dec2012_04 (that's great)")
    del(e_list[y])
    y=e_list.index('mm (mm)')
    del(e_list[y])
    y=e_list.index('mhm i see what you mean ')
    del(e_list[y])
    y=e_list.index('yeah i understand')
    del(e_list[y])
    y=e_list.index('yes (yes)')
    del(e_list[y])
    
    
    e_list= [x for x in e_list if str(x) != 'nan'] 
    print(len(e_list))
    
    
    
 
    
    # removing single word or 2 word sentences     
    for sentence in e_list:
      if len(sentence.split())== 1 or len(sentence.split())== 2 :
       e_list.remove(sentence)
#    print("new_list")
    print (len(e_list))
#S    print(e_list)
    
    
    return e_list



def stop_word():
    stop_words = stopwords.words('english')

    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    
    stop_words.extend(['all', 'just', 'being', 'over', 'both', 'through', 'yourselves', 'its', 'before',
                    'herself', 'had', 'should', 'to', 'only', 'under', 'ours', 'has', 'do', 'them', 
                    'his', 'very', 'they', 'not', 'during', 'now', 'him', 'nor', 'did', 'this', 'she',
                    'each', 'further', 'where', 'few', 'because',  'some', 'are', 'our', 'ourselves',
                    'out', 'what', 'for', 'while', 'does', 'above', 'between', 't', 'be', 'we', 'who', 'were', 'here',
                    'hers', 'by', 'on', 'about', 'of', 'against', 's', 'or', 'own', 'into', 'yourself', 'down', 'your', 
                    'from', 'her', 'their', 'there', 'been', 'whom', 'too', 'themselves', 'was', 'until', 'more', 'himself', 
                    'that', 'but', 'don', 'with', 'than', 'those', 'he', 'me', 'myself', 'these', 'up', 'will', 'below',
                    'can', 'theirs', 'my', 'and', 'then', 'is', 'am', 'it', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 
                    'if', 'again', 'no', 'when', 'same', 'how', 'other', 'which', 'you', 'after', 'most', 'such', 'why', 'a', 
                    'off', 'i', 'yours', 'so', 'the', 'having', 'once','thanks', "i'm","get",'oh',
                     'about',"<laughter>","i'll",'us',"okay","uh","um","know","think",'something','like',"that's",'really',
                     'sometime','things',"know","i\'m","that\'s", "how","eh","mm", "thing","(what","thing","well","anything",
                    "that)","example","see","(when","(can","(that\'s","much""could","(how","i\'ve","what\'s","(why","feel","(tell",'huh',
                    "(what\'s","hey","give_example","mhm","(i",'uh',"would",'guess','ellie','xxx','do','okay_confirm','tell','great','uh'])
    #print(stop_words)
    
    
    

    
    return(stop_words)



def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



def Bi_tri_model(data_words):
    # Build the bigram and trigram models
    bigram = Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = Phrases(bigram[data_words], threshold=100)  
    
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)
    
    # See trigram example
    print(trigram_mod[bigram_mod[data_words[0]]])
    return bigram_mod, trigram_mod 

   

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts,stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts,bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts,bigram_mod,trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    
    texts_out = []
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
    



def create_Dic_Corpus(data_lemmatized):

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    
    # Create Corpus
    texts = data_lemmatized
    
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    # View
    print(corpus[:1])
    return corpus,id2word


def topic_model(corpus,id2word):
    # Build LDA model
    lda_model = ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=20, 
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
    #Print the Keyword in the 10 topics
    print(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    #print(type(lda_model))
    return lda_model

def score (corpus,data_lemmatized,id2word,lda_model):
    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
    
    # Compute Coherence Score using cv
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score c_v: ', coherence_lda)
    
    # Compute Coherence Score using UMass
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence="u_mass")
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score u_mass: ', coherence_lda)
    
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    steps : no. of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values_u_mass = []
    coherence_values_c_v = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel_u_mass = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        coherence_values_u_mass.append(coherencemodel_u_mass.get_coherence())
        coherencemodel_c_v = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values_c_v.append(coherencemodel_c_v.get_coherence())   
        
    
    return model_list, coherence_values_u_mass, coherence_values_c_v, model




if __name__ == '__main__':
    
    train_dir='/media/hdd1/genfyp/depression_data/train_split_Depression_AVEC2017.csv'
    train_id=train_index(train_dir)
    
    working_dir = "/media/hdd1/genfyp/raw_data/transcript_data/"
#    with open("e_list.csv", 'rb') as f:
#        reader = csv.reader(f)
#        e_list = list(reader)

    transcripts,participants =data_retrieve(working_dir,train_id)
    
    #list ofEllie dictionary of all the unique sentences Ellie said 
    e_list=convert_df(transcripts,participants)
    #words to be removed
    stop_words=stop_word()
    #remover of stop_words
    data_words = list(sent_to_words(e_list))
    data_words = filter(None, data_words)
    
    print(data_words[:1])
    
    #building model
    bigram_mod, trigram_mod =Bi_tri_model(data_words)
    
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words, stop_words)
    
    
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops,bigram_mod)
    #Remove empty strings from a list of strings
    data_words_bigrams = filter(None, data_words_bigrams)
    
    
    
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    print(data_lemmatized[:1])
    
    #Create the Dictionary and Corpus needed for Topic Modeling
    corpus,id2word=create_Dic_Corpus(data_lemmatized)
    
    #create topic modeling
    lda_model=topic_model(corpus,id2word)
    
    
    score (corpus,data_lemmatized,id2word,lda_model)
    
    model_list,  coherence_values_u_mass, coherence_values_c_v, model = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)

    lda_model.save('lda_model')
    id2word.save("id2word.dict")
    
    # Show graph
    #u_mass
#    import matplotlib.pyplot as plt
#    limit=40; start=2; step=6;
#    x = range(start, limit, step)
#    plt.plot(x, coherence_values_u_mass)
#    plt.xlabel("Num Topics")
#    plt.ylabel("Coherence score")
#    plt.legend(("coherence_values"), loc='best')
#    plt.show()
#    #c_v
#    limit=40; start=2; step=6;
#    x = range(start, limit, step)
#    plt.plot(x, coherence_values_c_v)
#    plt.xlabel("Num Topics")
#    plt.ylabel("Coherence score")
#    plt.legend(("coherence_values"), loc='best')
#    plt.show()
    
  #  save_csv(corpus,data_lemmatized,id2word,lda_model)
    
#    with open("corpus.csv", 'wb') as myfile:
#                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#                    wr.writerow(corpus)
#    with open("id2word.csv", 'wb') as myfile:
#                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#                    wr.writerow(id2word)   
#    with open("lda_model.csv", 'wb') as myfile:
#                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#                    wr.writerow(lda_model)  
    with open("e_list.csv", 'wb') as myfile:
                        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                        wr.writerow(e_list)
