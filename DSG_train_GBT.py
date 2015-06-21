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
sf_test = gl.SFrame.read_csv('./data/test_sample_munged.csv', delimiter=',', 
	header=True, error_bad_lines=False, comment_char='', 
	escape_char='\\', double_quote=True, quote_char='"', 
	skip_initial_space=True, column_type_hints=None,
	 na_values=['NA'], line_terminator='\n', usecols=[],
	  nrows=None, verbose=True)

del sf_train['dimension']
del sf_train['definition']

del sf_test['dimension']
del sf_test['definition']

sf_train['likeCount'] = (sf_train['likeCount']/(sf_train['viewCount'] +1) )*100
sf_train['dislikeCount'] = (sf_train['dislikeCount']/(sf_train['viewCount'] +1) )*100
sf_train['dislikeCount'] = (sf_train['dislikeCount']/(sf_train['viewCount'] +1) )*100
sf_train['commentCount'] = (sf_train['commentCount']/(sf_train['viewCount'] +1) )*100
sf_train['title'] = sf_train['title'].apply(lambda x: ' '.join([word.lower() for word in x.split(" ") if len(word)>3 ]) )
sf_train['description'] = sf_train['description'].apply(lambda x: ' '.join([word.lower() for word in x.split(" ") if len(word)>3 ]) )


sf_test['likeCount'] = (sf_test['likeCount']/(sf_test['viewCount'] +1) )*100
sf_test['dislikeCount'] = (sf_test['dislikeCount']/(sf_test['viewCount'] +1) )*100
sf_test['dislikeCount'] = (sf_test['dislikeCount']/(sf_test['viewCount'] +1) )*100
sf_test['commentCount'] = (sf_test['commentCount']/(sf_test['viewCount'] +1) )*100
sf_test['title'] = sf_test['title'].apply(lambda x: ' '.join([word.lower() for word in x.split(" ") if len(word)>3 ]) )
sf_test['description'] = sf_test['description'].apply(lambda x: ' '.join([word.lower() for word in x.split(" ") if len(word)>3 ]) )




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


##on text
sf_train['stacked'] = sf_train['title'] + sf_train['description'] + sf_train['topicIds'] + sf_train['relevantTopicIds'] 
sf_test['stacked'] = sf_test['title'] + sf_test['description'] + sf_test['topicIds'] + sf_test['relevantTopicIds'] 


sf_train = conv2ngrame(sf_train,1,'stacked', 'uni')
sf_train = stopwords(sf_train,'uni')
sf_train = tfidf(sf_train,'uni')
sf_train = conv2ngrame(sf_train,2,'stacked', 'bi')
sf_train = stopwords(sf_train,'bi')
sf_train = tfidf(sf_train,'bi')


sf_test = conv2ngrame(sf_test,1,'stacked', 'uni')
sf_test = stopwords(sf_test,'uni')
sf_test = tfidf(sf_test,'uni')
sf_test = conv2ngrame(sf_test,2,'stacked', 'bi')
sf_test = stopwords(sf_test,'bi')
sf_test = tfidf(sf_test,'bi')

sf_train.save('./data/sf_train')
sf_test.save('./data/sf_test')


#load from binary
sf_train = gl.SFrame('./data/sf_train')
sf_test = gl.SFrame('./data/sf_test')


###categorical features
Features = ['viewCount','likeCount','dislikeCount', 'commentCount', 'duration', 'caption'
			,'licensedContent', 'dimension_2d', 'dimension_3d', 'definition_hd', 'definition_sd']
newFeatures = Features + ['topicIds']

train , val = sf_train.random_split(0.8)
gbt = gl.boosted_trees_classifier.create(train, "video_category_id", features=['uni','bi'], 
	max_iterations=10, validation_set=val, verbose=True, max_depth=None,
	class_weights=None)


y_pred = gbt.predict(sf_test)


submit = pd.DataFrame()
submit['id']=sf_test['id']





submit.save("./results/AM_uni.csv")

y_pred.save('./results/AM_uni_bi.csv')
















#KNN
knn = gl.nearest_neighbor_classifier.create(sf_train, "video_category_id", features=Features, 
												distance="cosine", verbose=True)

knn.predict(sf_train)