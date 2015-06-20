# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import csv as csv
import re

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


def features_transforming(df):
	df['duration'] = df.apply(convert_duration, axis=1)
	df['caption'] = df.apply(convert_caption, axis=1)
	df['licensedContent'] = df.apply(convert_licensedContent, axis=1)
	df['dimension_2d'] = df.apply(add_dimension_2d, axis=1)
	df['dimension_3d'] = df.apply(add_dimension_3d, axis=1)
	df['definition_hd'] = df.apply(add_definition_hd, axis=1)
	df['definition_sd'] = df.apply(add_definition_sd, axis=1)


####################################################################################################################################
# Main
####################################################################################################################################

train_df = pd.read_csv('./data/train_sample.csv', header=0, escapechar='\\', quotechar='"')
test_df = pd.read_csv('./data/test_sample.csv', header=0, escapechar='\\', quotechar='"')

duration_regex = "P(([0-9]+)D)?T(([0-9]+)H)?(([0-9]+)M)?(([0-9]+)S)?"

features_transforming(train_df)
features_transforming(test_df)

train_df.to_csv('./data_munging/train_sample.csv', sep=',', index=None)
test_df.to_csv('./data_munging/test_sample.csv', sep=',', index=None)



	



