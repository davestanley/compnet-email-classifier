# -*- coding: utf-8 -*-
"""
Created on Fri May  9 22:40:03 2014

@author: davestanley
"""
#Load data
import json
from pprint import pprint
json_data=open('./neurotalk-emails.json.txt')

data = json.load(json_data)
#pprint(data)
json_data.close()


#GET FEATURE VECTOR
import nltk
ind = 0 # one of the 132 emails contained in data
email = nltk.clean_html(data[ind]['bodyText'])  # clean up html tags
words = nltk.word_tokenize(email)  # tokenize it


import collections
cnt = collections.Counter()
for word in words:
     cnt[word.lower()] += 1


common_words = ['brain','neuron','neural','dendrite','axon','cerebral','plasticity']


