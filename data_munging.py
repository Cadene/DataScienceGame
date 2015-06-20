# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import time
import os
import datetime
from dateutil import parser
import csv

from unidecode import unidecode
from nltk import PorterStemmer

####################################################################################################################################
# Functions
####################################################################################################################################

# tools 
def isNaN(num):
    return num != num

# exemple
# duration2sec("P1DT1H15M33S")
def duration2sec(duration):
	if isNaN(duration):
		return -1.
	sec = 0.
	duration_regex = "P(([0-9]+)D)?T(([0-9]+)H)?(([0-9]+)M)?(([0-9]+)S)?"
	match = re.findall(duration_regex, duration)
	if len(match) != 1 or len(match[0]) != 8:
		print "Error: ", duration
		return -1.
	match = match[0]
	if match[1]: # day
		day = int(match[1])
		sec += day * 86400
	if match[3]: # hour
		hour = int(match[3])
		sec += hour * 3600
	if match[5]: # min
		minu = int(match[5])
		sec += minu * 60
	if match[7]: # sec
		seco = int(match[7])
		sec += seco
	return sec

# Convert functions

def convert_duration(df):
	return duration2sec(df['duration'])

def convert_caption(df):
	if df['caption']:
		return 1
	else:
		return 0 # column = false or NaN

def convert_licensedContent(df):
	if not df['licensedContent']:
		return 0
	else:
		return 1 # column = True or NaN

def convert_topicIds(df):
    #create list of topicIds from raw string
    if type(df['topicIds']) == str:
    	return ' '.join(df['topicIds'].split(';'))
    else:
    	return ''

def convert_relevantTopicIds(df):
    #create list of relevantTopicIds from raw strings
    if type(df['relevantTopicIds']) == str:
    	return ' '.join(df['relevantTopicIds'].split(';'))
    else:
    	return ''

def convert_published_at(df):
    #parse the sate into datetime format 
    if len(df['published_at']) == 24:
    	return parser.parse(df['published_at'][:19])
    else:
    	return datetime.datetime.now()

def convertion_description_url(df):
	url_regex = r'(https?:\/\/)?([a-zA-Z0-9]+\.)?([a-zA-Z0-9]+)\.[a-zA-Z0-9]+\/?([\"\'\=a-zA-Z0-9\/\?\_\-]+)?'
	string = str(df['description'])
	return re.sub(url_regex, r'\3', string)

# Add functions

def add_dimension_2d(df):
	if df['dimension'] == '2d' or isNaN(df['dimension']):
		return 1
	else:
		return 0

def add_dimension_3d(df):
	if df['dimension'] == '3d':
		return 1
	else: # if 2d or NaN
		return 0 

def add_definition_hd(df):
	if df['definition'] == 'hd' or isNaN(df['definition']):
		return 1
	else:
		return 0 

def add_definition_sd(df):
	if df['definition'] == 'sd':
		return 1
	else: # if hd or NaN
		return 0



def add_description_is_url(df):
	url_regex = r'(https?:\/\/)?([a-zA-Z0-9]+\.)?([a-zA-Z0-9]+)\.[a-zA-Z0-9]+\/?([\"\'\=a-zA-Z0-9\/\?\_\-]+)?'
	string = str(df['description'])
	match = re.findall(url_regex, string)
	if len(match) != 1 or len(match[0]) != 4:
		return 0
	else:
		return 1



# re.sub(url_regex, string, '\3')

# Empty functions

def empty_df(df, index, default):
	if isNaN(df[index]):
		return default
	else:
		return df[index]


def convertion_title_strip(df):
	return unidecode(unicode(df['title'], 'utf-8'))

def convertion_description_strip(df):
	return unidecode(unicode(df['description'], 'utf-8'))	

def features_transforming(df):
	# convert
	df['duration'] = df.apply(convert_duration, axis=1)
	df['caption'] = df.apply(convert_caption, axis=1)
	df['licensedContent'] = df.apply(convert_licensedContent, axis=1)
	df['topicIds'] = df.apply(convert_topicIds, axis=1)
	df['relevantTopicIds'] = df.apply(convert_relevantTopicIds, axis=1)
	df['published_at'] = df.apply(convert_published_at, axis=1)
	df['description'] = df.apply(convertion_description_url, axis=1)
	df['title'] = df.apply(convertion_title_strip, axis=1)
	df['description'] = df.apply(convertion_description_strip, axis=1)
	df['title'] = df['title'].apply(lambda r: ' '.join([stemmer.stem(word) for word in r.split(" ")]) )
	df['description'] = df['description'].apply(lambda r: ' '.join([stemmer.stem(word) for word in r.split(" ")]) )
	print "features_transforming convert :", (time.time() - t0)
	# add
	df['dimension_2d'] = df.apply(add_dimension_2d, axis=1)
	df['dimension_3d'] = df.apply(add_dimension_3d, axis=1)
	df['definition_hd'] = df.apply(add_definition_hd, axis=1)
	df['definition_sd'] = df.apply(add_definition_sd, axis=1)
	df['description_is_url'] = df.apply(add_description_is_url, axis=1)
	print "features_transforming add :", (time.time() - t0)
	# empty
	viewCount_default = df['viewCount'].median() # in case median() is exec each time
	likeCount_default = df['likeCount'].median()
	dislikeCount_default = df['dislikeCount'].median()
	favoriteCount_default = df['favoriteCount'].median()
	commentCount_default = df['commentCount'].median()
	description_default = ""
	df['viewCount'] = df.apply(empty_df, axis=1, args=('viewCount',viewCount_default))
	df['likeCount'] = df.apply(empty_df, axis=1, args=('likeCount',likeCount_default))
	df['dislikeCount'] = df.apply(empty_df, axis=1, args=('dislikeCount',dislikeCount_default))
	df['favoriteCount'] = df.apply(empty_df, axis=1, args=('favoriteCount',favoriteCount_default))
	df['commentCount'] = df.apply(empty_df, axis=1, args=('commentCount',commentCount_default))
	#df['description'] = df.apply(empty_df, axis=1, args=('description',description_default))
	print "features_transforming empty :", (time.time() - t0)


####################################################################################################################################
# Main
####################################################################################################################################


# pre pre processing

# def prepreprocessing(path2oldfile, path2newfile):
# 	original_file= open(path2oldfile,'r')# r when we only wanna read file
# 	revised_file = open(path2newfile,'w')# w when u wanna write sth on the file
# 	virgule_regex = r'^(,)?'
# 	quote_regex = r',".*(,).*",'
# 	for aline in original_file:
# 		n_aline = re.sub(virgule_regex, '', aline)
# 		nn_aline = re.sub(quote_regex, '', n_aline)
# 	original_file.close()
# 	revised_file.close()

def preprocessing(df):
	df = df.fillna('')
	df.drop(u'favoriteCount',1).columns
	for col in l_digit_col:
		df[col] =df[col].apply(lambda r: int(r) if str(r).isdigit() else -1 )
	for col in l_digit_col:
	    mean = df[col].median()
	    df[col] = df[col].apply(lambda r: mean if r==-1 else r )
	print "prepreprocessing :", (time.time() - t0)
	return df

stemmer = PorterStemmer()

folder = os.getcwd() ; print folder

t0 = time.time()

l_digit_col = [u'viewCount',u'likeCount', u'dislikeCount', u'commentCount']

####################################################################################################################################
# train_sample.csv

train_df = pd.read_csv('./data/train_sample.csv',sep=",", header=0, escapechar='\\',
                       quotechar='"', error_bad_lines=False, encoding=None )
print "Reading train_sample.csv :", (time.time() - t0)

train_df = preprocessing(train_df)

features_transforming(train_df)
print "features_transforming(train_df) :", (time.time() - t0)

train_df.to_csv('./data/train_sample_munged.csv', sep=',', index=None)

####################################################################################################################################
# test_sample.csv

test_df = pd.read_csv('./data/test_sample.csv', sep=",", header=0, escapechar='\\', quotechar='"', error_bad_lines=False)
print "Reading test_sample.csv :", (time.time() - t0)

test_df = preprocessing(test_df)

features_transforming(test_df)
print "features_transforming(test_df) :", (time.time() - t0)

test_df.to_csv('./data/test_sample_munged.csv', sep=',', index=None)




	



