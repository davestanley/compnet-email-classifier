# -*- coding: utf-8 -*-
"""
Created on Sat May 10 11:47:08 2014

@author: davestanley
"""

#Imports
import json
import nltk
import copy
import os

#Functions
def body_html_merge_raw(uin):
    bodyText = uin['bodyText']
    if bodyText is None:
        uin['bodyText'] = uin['bodyHtml']
    elif not uin['bodyHtml'] is None :
        uin['bodyText'] = uin['bodyHtml']
    uin['bodyHtml'] = None
    return uin

def body_html_merge(uin):
    bodyText = uin['bodyText']
    if bodyText is None:
        uin['bodyText'] = nltk.clean_html(uin['bodyHtml'])
    else:
        uin['bodyText'] = nltk.clean_html(uin['bodyHtml'])
    uin['bodyHtml'] = None



# IMPORTANT - change to working code directory!
currdir = '/Users/davestanley/src/compnet-email-classifier'
os.chdir(currdir)
data_open = os.path.join(currdir,'data','ham','neurotalk-emails.json')
data_save = os.path.join(currdir,'data','spam','neurotalk-emails_merged.json')

#Load JSON data
json_data=open(data_open,'r')
data = json.load(json_data)
#pprint(data)
json_data.close()


#Get one big list of all words in all emails
Nemails = len(data)

data_clean = copy.deepcopy(data)
for ind in range(Nemails):
    body_html_merge(data_clean[ind])


json_data_merged=open(data_save,'w')
json.dump(data_clean,json_data_merged)





