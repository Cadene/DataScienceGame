# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import os
import datetime
from dateutil import parser

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
	match = re.findall(duration_regex, duration)
	if len(match) != 1 or len(match[0]) != 8:
		print "Error: {}", duration
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
    	return df['topicIds'].split(";")
    else:
    	return []

def convert_relevantTopicIds(df):
    #create list of relevantTopicIds from raw strings
    if type(df['relevantTopicIds']) == str:
    	return df['relevantTopicIds'].split(";")
    else:
    	return []

def convert_published_at(df):
    #parse the sate into datetime format 
    if len(df['published_at']) == 24:
    	return parser.parse(df['published_at'][:19])
    else:
    	return datetime.datetime.now()

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

# Empty functions

def empty_df(df, index, default):
	if isNaN(df[index]):
		return default
	else:
		return df[index]

def features_transforming(df):
	# convert
	df['duration'] = df.apply(convert_duration, axis=1)
	df['caption'] = df.apply(convert_caption, axis=1)
	df['licensedContent'] = df.apply(convert_licensedContent, axis=1)
	df['topicIds'] = df.apply(convert_TopicIds, axis=1)
	df['relevantTopicIds'] = df.apply(convert_relevantTopicIds, axis=1)
	df['published_at'] = df.apply(convert_published_at, axis=1)
	# add
	df['dimension_2d'] = df.apply(add_dimension_2d, axis=1)
	df['dimension_3d'] = df.apply(add_dimension_3d, axis=1)
	df['definition_hd'] = df.apply(add_definition_hd, axis=1)
	df['definition_sd'] = df.apply(add_definition_sd, axis=1)
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
	df['description'] = df.apply(empty_df, axis=1, args=('description',description_default))


####################################################################################################################################
# Main
####################################################################################################################################

folder = os.getcwd() ; print folder

train_df = pd.read_csv('./data/train_sample.csv', header=0, escapechar='\\', quotechar='"')
test_df = pd.read_csv('./data/test_sample.csv', header=0, escapechar='\\', quotechar='"')

duration_regex = "P(([0-9]+)D)?T(([0-9]+)H)?(([0-9]+)M)?(([0-9]+)S)?"

features_transforming(train_df)
features_transforming(test_df)

train_df.to_csv('./data_munging/train_sample.csv', sep=',', index=None)
test_df.to_csv('./data_munging/test_sample.csv', sep=',', index=None)



	



