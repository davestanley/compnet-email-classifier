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
import os

# IMPORTANT - change to working code directory!
currdir = os.path.join(os.getenv("HOME"),'src','compnet-email-classifier')
os.chdir(currdir)
filename_ham = os.path.join(currdir,'data','ham','neurotalk-emails.json')
filename_spam_nnt = os.path.join(currdir,'data','spam','non-neurotalk-emails.json')
filename_spam_nt = os.path.join(currdir,'data','spam','non-talk-emails.json')


# Parameters
Nfeature_words = 10
fract_training = 0.6    # Training
fract_cv = 0.0  # Cross validation (not used)
fract_test = 1 - fract_training - fract_cv  # Testing
normalize_on = 1    # =1 to normalize feature vector values


# Extract the set of words in each email
data_words = extract_words(filename_ham)
Nemails = len(data_words)
Nemails_training = int(np.floor(Nemails * fract_training))


# Count the number of occurrences of all words in all emails
cnt = collections.Counter()
for ind in range(Nemails_training):
    for word in data_words[ind]:
        cnt[word.lower()] += 1


# Use this (cnt) to generate a list of the N most common words
#   These will be our feature words
feature_vector = list()
for words in cnt.most_common(Nfeature_words):
    feature_vector.append(words[0])


# Add template words to our list of feature word 
brain_words = [u'brain',u'neuron',u'neural',u'dendrite',u'axon',u'cerebral',u'plasticity',u'artificial',u'viagara','mind']
feature_vector = brain_words + feature_vector     # our final feature vector
feature_vector = convert2stems(feature_vector)

# Save our feature word vector for future usage
with open("feature_vector.json", "w") as json_featurevector: 
     json.dump(feature_vector, json_featurevector)
# Note, apparently this code will automatically close the file. See here: https://stackoverflow.com/questions/8011797/open-read-and-close-a-file-in-1-line-of-code


#Get a feature vector for each email
data_features = get_features(data_words,feature_vector,normalize_on)
add_label(data_features, 1) # add 1 for correct (neuro talk)
    
# Get words from other example emails
nnt_words = extract_words(filename_spam_nnt)  # Non neuro talks
nt_words = extract_words(filename_spam_nt)  # Non talks

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
cv_features = np.array(cv_features)
test_features = np.array(test_features)

# Do zscore
train_features2 = myzscore(train_features,1)
#cv_features2 = myzscore(cv_features,1)
test_features2 = myzscore(test_features,1)



# Run SVM
from sklearn import svm, datasets

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(train_features2, ytr)

out = svc.predict(train_features2)
out = svc.predict(test_features2)


# Stats
tps = sum((out == 1).__and__(yts == 1))
fps = sum((out == 1) .__and__ (yts == 0))
fns = sum((out == 0) .__and__ (yts == 1))
tns = sum((out == 0) .__and__ (yts == 0))

sensitivity = 1.0 * tps / (tps + fns)
specificity = 1.0 * tns / (fps + tns)

print ("Sensitivity " + repr(sensitivity))
print ("Specificity " + repr(specificity))


#Expected output - will vary depending on the random training dataset that is chosen
#Extracted 132 emails.
#Extracted 10 emails.
#Extracted 19 emails.
#Sensitivity 0.90566037735849059
#Specificity 0.66666666666666663

