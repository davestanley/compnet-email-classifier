# -*- coding: utf-8 -*-
"""
Created on Sat May 10 17:54:38 2014

@author: davestanley
"""

#Import
import json
from supporting_funcs import *
from json_clean import body_html_merge
import pdb


def extract_words(filename):
    #Variables
    train_fract = 0.5
    
    
    #Load JSON data
    json_data=open(filename,'r')
    data = json.load(json_data)
    json_data.close()
    Nemails = len(data)
#    if Nemails > 30: Nemails = 50
    
    #Clean data
    for ind in range(Nemails):
        body_html_merge(data[ind])
    
    
    print "Extracted " + str(Nemails) + " emails."
    #Get a list of words for each email
    data_words = list()
    for ind in range(Nemails):
        email_tokens = tokenize_email(data[ind]['bodyText'])
        email_english = convert2english(email_tokens)
        email_nostops = extract_no_stopwords(email_english)
        email_stems = convert2stems(email_nostops)
        email_lcase = convert2lcase(email_stems)
        data_words.append(email_lcase)
    
    return data_words