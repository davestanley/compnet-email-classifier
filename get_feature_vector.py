# -*- coding: utf-8 -*-
"""
Created on Fri May  9 22:40:03 2014

@author: davestanley
"""

from extract_words import extract_words
import collections
import numpy as np
import json
from supporting_funcs import *


# Parameters
Nfeature_words = 40
fract_training = 0.6
fract_cv = 0.2
fract_test = 1 - fract_training - fract_cv
normalize_on = 0    # =1 to normalize feature vector values


# Extract the set of words in each email
data_words = extract_words('neurotalk-emails.json')
Nemails = len(data_words)
Nemails_training = int(floor(Nemails * fract_training))


# Count the number of occurrences of all words in all emails
cnt = collections.Counter()
for ind in range(Nemails_training):
    for word in data_words[ind]:
        cnt[word.lower()] += 1


# Use this (cnt) to generate a list of the N most common words
#   These will be our feature words
#   Need to update this to sample words from non-neuro datasets as well.
feature_vector = list()
for words in cnt.most_common(Nfeature_words):
    feature_vector.append(words[0])


# Add template words to our list of feature word 
brain_words = [u'brain',u'neuron',u'neural',u'dendrite',u'axon',u'cerebral',u'plasticity',u'artificial',u'viagara','mind']
feature_vector = brain_words + feature_vector     # our final feature vector
feature_vector = convert2stems(feature_vector)

# Save our feature word vector for future usage
json_featurevector=open('./feature_vector.json','wb')
json.dump(feature_vector,json_featurevector)


#Get a feature vector for each email
data_features = get_features(data_words,feature_vector,normalize_on)
add_label(data_features, 1) # add 1 for correct (neuro talk)
    
# Get words from other example emails
nnt_words = extract_words('./data/spam/non-neurotalk-emails.json')
nt_words = extract_words('./data/spam/non-talk-emails.json')

# Get features from other example emails
nnt_features = get_features(nnt_words,feature_vector,normalize_on)
add_label(nnt_features, 0)  # add 0 for incorrect
nt_features = get_features(nt_words,feature_vector,normalize_on)
add_label(nt_features, 0)  # add 0 for incorrect

## Build train, test, and validation sets
#   Pull random emails from each feature set
N_nnt = len(nnt_features)
N_nt = len(nt_features)

# Training list
train_features = list()
for i in range(int(Nemails*fract_training)):
    random_pipelist(data_features,train_features)

for i in range(int(N_nnt*fract_training)):
    random_pipelist(nnt_features,train_features)

for i in range(int(N_nt*fract_training)):
     random_pipelist(nt_features,train_features)

# Cross validation list
cv_features = list()
for i in range(int(Nemails*fract_cv)):
    random_pipelist(data_features,cv_features)

for i in range(int(N_nnt*fract_cv)):
    random_pipelist(nnt_features,cv_features)

for i in range(int(N_nt*fract_cv)):
    random_pipelist(nt_features,cv_features)


# Test list - just take remaining emails
test_features = list()
for i in range(len(data_features)):
    random_pipelist(data_features,test_features)

for i in range(len(nnt_features)):
    random_pipelist(nnt_features,test_features)

for i in range(len(nt_features)):
    random_pipelist(nt_features,test_features)


# Extract labels
ytr = pop_labels(train_features)
ycv = pop_labels(cv_features)
yts = pop_labels(test_features)


# Convert everything to numpy arrays
ytr = np.array(ytr)
ycv = np.array(ycv)
yts = np.array(yts)
train_features = np.array(train_features)
nnt_features = np.array(nnt_features)
nt_features = np.array(nt_features)

# Run SVM
from sklearn import svm, datasets

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(train_features, ytr)

out = svc.predict(train_features)
out = svc.predict(test_features)



# Stats
tps = sum((out == 1).__and__(yts == 1))
fps = sum((out == 1) .__and__ (yts == 0))
fns = sum((out == 0) .__and__ (yts == 1))
tns = sum((out == 0) .__and__ (yts == 0))

sensitivity = tps / (tps + fns)
specificity = tns / (fps + tns)

print "Sensitivity " + repr(sensitivity)
print "Specificity " + repr(specificity)







