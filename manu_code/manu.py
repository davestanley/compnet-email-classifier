# -*- coding: utf-8 -*-
"""
Created on Fri May  9 23:41:29 2014

@author: manu
"""

#Load data
import json
from pprint import pprint
json_data=open('./neurotalk-emails.json.txt')

data = json.load(json_data)
pprint(data)
json_data.close()

import nltk
ind = 0 # one of the 132 emails contained in data
email = nltk.clean_html(data[ind]['bodyText'])  # clean up html tags
words = nltk.word_tokenize(email)  # tokenize it

##Remove html tags
#from lxml import html
#from lxml.html.clean import clean_html
#
#temp2 = temp['bodyText']
#temp2 = '''Seminar Series\r\n\r\n\r\n\r\nFriday, May 9, 2014 at 2:00pm\r\n\r\n\r\n\r\nBoston University\r\n\r\n677 Beacon Street\r\n\r\nAuditorium, Room B02\r\n\r\nRefreshments will be served after the talk in Room B03\r\n\r\n\r\n\r\nFor'''
#tree = html.parse(temp2)
#tree = clean_html(tree)
#
#text = tree.getroot().text_content()



#