# -*- coding: utf-8 -*-
"""
Created on Sat May 10 14:35:17 2014

@author: davestanley
"""

import json

json_data=open('./neurotalk-emails2.json.txt')
data = json.load(json_data)
#pprint(data)
json_data.close()



json_data_merged=open('./neurotalk-emails2.json.txt','wb')
json.dump(data,json_data_merged)


import json
with open('data.json', 'w') as fp:
    json.dump(data_clean, fp)
    
    
with open('data.json', 'r') as fp:
    data = json.load(fp)