# -*- coding: utf-8 -*-
"""
Created on Sat May 10 11:44:36 2014

@author: davestanley
"""
import pdb
import nltk
from nltk.corpus import wordnet
from nltk.tag import pos_tag


# General functions
def get_features(data_words,feature_words,normalize_on):
    data_features = list()
    Nemails = len(data_words)
    for ind in range(Nemails):
        email_lcase = data_words[ind]
        curr_featurevector = list()
        for feature in feature_words:
            curr_featurevector.append(email_lcase.count(feature.lower()))
        if normalize_on:
            curr_featurevector = [1.0 * cfve / len(email_lcase) for cfve in curr_featurevector] # Element-wise division
        data_features.append(curr_featurevector)
    return data_features

# Adds a label onto the end of each feature vector
def add_label(data_features, label):
    for feature in data_features:
        feature.append(label)

# Pops a label onto the end of each feature vector
def pop_labels(data_features):
    labels = list()
    for feature in data_features:
        labels.append(feature.pop(-1))
    return labels

##Email Functions
def tokenize_email(uin):
    if uin is not None:
        uin_clean = nltk.clean_html(uin)
        uin_token = nltk.word_tokenize(uin_clean)
        return uin_token
            #nltk crashes when uin is type None; therefore ignore
    else:
        None

def convert2english(tokens):
    out = list()
    for token in tokens:
        if is_english(token):
            out.append(token)
    return out

def extract_onlynouns(tokens):
    out = list()
    for token in tokens:
        pos = pos_tag(nltk.word_tokenize(token.lower()))[0][1]
        if (pos == "NN") or (pos == "NNP"):
            out.append(token)
    return out

def extract_no_stopwords(tokens):
    from nltk.corpus import stopwords
    out = list()
    out = [w for w in tokens if not w in stopwords.words('english')]
    return out

def convert2stems(tokens):
    out = list()
    port = nltk.stem.PorterStemmer()
    for token in tokens:
        out.append(port.stem(token))
    return out

def convert2lcase(tokens):
    out = list()
    for token in tokens:
        out.append(token.lower())
    return out


##Basic Functions
def is_english(someword):
    if not wordnet.synsets(someword):
        return False
    else:
        return True
        

def random_pipelist(source,dest):   # Takes a random element from one list and appends to another
    from random import randrange
    random_index = randrange(0,len(source))
    dest.append(source[random_index])
    source.pop(random_index)


