# -*- coding: utf-8 -*-
"""
Created on Sat May 10 17:51:01 2014

@author: davestanley
"""



#Imports
import json
import nltk
import collections
from supporting_funcs import *
from json_clean import *


def extract_features(filename):
    
    #Variables
    train_fract = 0.5
    
    
    #Load JSON data
    json_data=open(filename)
    data = json.load(json_data)
    json_data.close()
    Nemails = len(data)
    Nemails = 2
    
    #Clean data
    for ind in range(Nemails):
        body_html_merge(data[ind])
    
    
    
    Nemails_test = int(floor(Nemails * train_fract))
    
    
    #Get a list of words for each email
    data_words = list()
    for ind in range(Nemails):
        email_tokens = tokenize_email(data[ind]['bodyText'])
        email_english = convert2english(email_tokens)
        email_nouns = extract_onlynouns(email_english)
        email_stems = convert2stems(email_nouns)
        email_lcase = convert2lcase(email_stems)
        data_words.append(email_lcase)
    
    
    cnt = collections.Counter()
    for ind in range(Nemails_test):
        for word in data_words[ind]:
            cnt[word.lower()] += 1
    
    feature_words = list()
    for words in cnt.most_common(100):
        feature_words.append(words[0])
    
    
    
    #Get a feature vector for each email
    data_features = list()
    for ind in range(Nemails):
        email_lcase = data_words[ind]
        curr_featurevector = list()
        for feature in feature_words:
            curr_featurevector.append(email_lcase.count(feature.lower()))
        data_features.append(curr_featurevector)
        
        
    return data_features


