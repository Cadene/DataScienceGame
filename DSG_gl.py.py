import graphlab as gl
import numpy as np
import pandas as pd
import logging
import os
#logging.getLogger("requests").setLevel(logging.WARNING)
logging.disable(logging.INFO)

sf_train = gl.SFrame.read_csv('./data/train_sample_munged.csv', delimiter=',', 
	header=True, error_bad_lines=False, comment_char='', 
	escape_char='\\', double_quote=True, quote_char='"', 
	skip_initial_space=True, column_type_hints=None,
	 na_values=['NA'], line_terminator='\n', usecols=[],
	  nrows=None, verbose=True)
sf_test = gl.SFrame.read_csv('./data/train_sample_munged.csv', delimiter=',', 
	header=True, error_bad_lines=False, comment_char='', 
	escape_char='\\', double_quote=True, quote_char='"', 
	skip_initial_space=True, column_type_hints=None,
	 na_values=['NA'], line_terminator='\n', usecols=[],
	  nrows=None, verbose=True)

del sf_train['dimension']
del sf_train['definition']

del sf_test['dimension']
del sf_test['definition']


#CONV 2 NGRAM
def conv2ngrame(sf,nb_gram,col2parse, newcol):
    sf[newcol] = gl.text_analytics.count_ngrams(sf[col2parse], to_lower=True, n=nb_gram)
    return sf


def stopwords(sf,col_name):
    sf[col_name] = sf[col_name].dict_trim_by_keys(gl.text_analytics.stopwords(lang='en'), True)
    return sf

def tfidf(sf,col_name):
    sf[col_name] = gl.text_analytics.tf_idf(sf[col_name])['docs']
    return sf

print sf_train.column_names()



###categorical features
Features = ['viewCount','likeCount','dislikeCount', 'commentCount', 'duration', 'caption'
			,'licensedContent', 'dimension_2d', 'dimension_3d', 'definition_hd', 'definition_sd']


m = gl.boosted_trees_classifier.create(sf_train, "video_category_id", features=Features, 
	max_iterations=50, validation_set='auto', verbose=True, 
	class_weights=None)




##on text
sf_train['stacked'] = sf_train['title'] + sf_train['description'] + sf_train['topicIds'] + sf_train['relevantTopicIds'] 

newFeatures = ['stacked'] +Features

sf_train = conv2ngrame(sf_train,1,'stacked', 'uni')
sf_train = stopwords(sf_train,'uni')
sf_train = tfidf(sf_train,'uni')

sf_train = conv2ngrame(sf_train,2,'stacked', 'bi')
sf_train = stopwords(sf_train,'bi')
sf_train = tfidf(sf_train,'bi')

m = gl.boosted_trees_classifier.create(sf_train, "video_category_id", features=['uni','bi'], 
	max_iterations=10, validation_set='auto', verbose=True,
	class_weights=None)