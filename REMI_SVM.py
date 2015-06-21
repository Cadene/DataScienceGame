# -*- coding: utf-8 -*-

# REMI_SVM.py

import sklearn.feature_extraction.text as txt
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import hstack

from sklearn import svm
from sklearn import neighbors
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, train_test_split,StratifiedKFold, StratifiedShuffleSplit


from lib_DSG import ColumnSelector, DenseTransformer

import time
t0 = time.time()

df_train = pd.read_csv('./data/train_sample_munged.csv', header=0, escapechar='\\', quotechar='"', low_memory=False)
df_train = df_train.fillna('')

df_train['likeCount'] = (df_train['likeCount']/(df_train['viewCount'] +1) )*100
df_train['dislikeCount'] = (df_train['dislikeCount']/(df_train['viewCount'] +1) )*100
df_train['dislikeCount'] = (df_train['dislikeCount']/(df_train['viewCount'] +1) )*100
df_train['commentCount'] = (df_train['commentCount']/(df_train['viewCount'] +1) )*100
df_train['word'] = df_train['title'] + ' ' + df_train['description']
df_train['topic'] = df_train['topicIds'] + ' ' + df_train['relevantTopicIds']

df_test = pd.read_csv('./data/test_sample_munged.csv', header=0, escapechar='\\', quotechar='"', low_memory=False)
df_test = df_test.fillna('')

df_test['likeCount'] = (df_test['likeCount']/(df_test['viewCount'] +1) )*100
df_test['dislikeCount'] = (df_test['dislikeCount']/(df_test['viewCount'] +1) )*100
df_test['dislikeCount'] = (df_test['dislikeCount']/(df_test['viewCount'] +1) )*100
df_test['commentCount'] = (df_test['commentCount']/(df_test['viewCount'] +1) )*100
df_test['word'] = df_test['title'] + ' ' + df_test['description']
df_test['topic'] = df_test['topicIds'] + ' ' + df_test['relevantTopicIds']


dico_pattern={'match_lowercase_only':'\\b[a-z]+\\b',
              'match_word':'\\w{1,}',
              'match_word1': '(?u)\\b\\w+\\b',
              'match_word_punct': '\w+|[,.?!;]',
              'match_NNP': '\\b[A-Z][a-z]+\\b|\\b[A-Z]+\\b',
              'match_punct': "[,.?!;'-]"
             }

stopwords = [u'http', u'com', u'www', u's', u'subscribe' , u'new', u'like', u'watch', u't', u'2014', u'1'
, u'2013', u'2', u'la', u'en', u'world', u'make', u'v', u'check', u'time', u'10', u'best', u'3', u'5', u'day', 
u'y', 'the', 'to', 'a', 'of', 'in', 'floor','for', 'is', 'you','video' , 'this', 'with', 'at', 'it', 'i',
            'that', 'we', 'your' , 'be', 'are', 'about', 'and', 'on', 'what', 'by', 'was', 'http', 'https', 'if',
            'get', 'can', 'up']


# TF-IDF

tfv_title = TfidfVectorizer(lowercase=True, stop_words='english', token_pattern=dico_pattern["match_word1"], 
                      ngram_range=(1, 2), max_df=1.0, min_df=1, max_features=None, 
                      vocabulary=None, binary=True, norm=u'l2', 
                      use_idf=True, smooth_idf=True, sublinear_tf=True)

tfv_desc = TfidfVectorizer(lowercase=True, stop_words='english', token_pattern=dico_pattern["match_word1"], 
                      ngram_range=(1, 2), max_df=1.0, min_df=1, max_features=None, 
                      vocabulary=None, binary=True, norm=u'l2', 
                      use_idf=True, smooth_idf=True, sublinear_tf=True)

tfv_topicid = TfidfVectorizer(lowercase=True, stop_words=None, token_pattern=dico_pattern["match_word1"], 
                      ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, 
                      vocabulary=None, binary=True, norm=u'l2', 
                      use_idf=True, smooth_idf=True, sublinear_tf=True)

tfv_rel_topic = TfidfVectorizer(lowercase=True, stop_words=None, token_pattern=dico_pattern["match_word1"], 
                      ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, 
                      vocabulary=None, binary=True, norm=u'l2', 
                      use_idf=True, smooth_idf=True, sublinear_tf=True)

title_pipe = make_pipeline(ColumnSelector(key='title'), tfv_title)
desc_pipe = make_pipeline(ColumnSelector(key='description'), tfv_desc)
topicId_pipe = make_pipeline(ColumnSelector(key='topicIds'), tfv_topicid)
reltopicID_pipe = make_pipeline(ColumnSelector(key='relevantTopicIds'), tfv_rel_topic)

pipeline = make_union(title_pipe, desc_pipe, topicId_pipe, reltopicID_pipe)

# TF-IDF 2 colomn

tfv_word = TfidfVectorizer(lowercase=True, stop_words=stopwords, token_pattern=dico_pattern["match_word1"], 
                      ngram_range=(1, 2), max_df=1.0, min_df=2, max_features=None, 
                      vocabulary=None, binary=True, norm=u'l2', 
                      use_idf=True, smooth_idf=True, sublinear_tf=True)

tfv_topic = TfidfVectorizer(lowercase=True, stop_words=None, token_pattern=dico_pattern["match_word1"], 
                      ngram_range=(1, 1), max_df=1.0, min_df=2, max_features=None, 
                      vocabulary=None, binary=True, norm=u'l2', 
                      use_idf=True, smooth_idf=True, sublinear_tf=True)

word_pipe = make_pipeline(ColumnSelector(key='word'), tfv_word)
topic_pipe = make_pipeline(ColumnSelector(key='topic'), tfv_topic)

pipeline = make_union(word_pipe, topic_pipe)

pipeline.transformer_weights=[2, 1]

hstack((pipeline.fit_transform(df_train),df_train[['viewCount', 'likeCount', 'dislikeCount','commentCount', 'duration', 'caption', 'licensedContent', 'dimension_2d', 'dimension_3d', 'definition_hd', 'definition_sd' ]]))
hstack((pipeline.fit_transform(df_test),df_test[['viewCount', 'likeCount', 'dislikeCount','commentCount', 'duration', 'caption', 'licensedContent', 'dimension_2d', 'dimension_3d', 'definition_hd', 'definition_sd'  ]]))

# features categorie

viewCount_pipe = make_pipeline(ColumnSelector(key='viewCount'))
likeCount_pipe = make_pipeline(ColumnSelector(key='likeCount'))
dislikeCount_pipe = make_pipeline(ColumnSelector(key='dislikeCount'))
commentCount_pipe = make_pipeline(ColumnSelector(key='commentCount'))
duration_pipe = make_pipeline(ColumnSelector(key='duration'))
caption_pipe = make_pipeline(ColumnSelector(key='caption'))
licensedContent_pipe = make_pipeline(ColumnSelector(key='licensedContent'))
dimension_2d_pipe = make_pipeline(ColumnSelector(key='dimension_2d'))
dimension_3d_pipe = make_pipeline(ColumnSelector(key='dimension_3d'))
definition_hd_pipe = make_pipeline(ColumnSelector(key='definition_hd'))
definition_sd_pipe = make_pipeline(ColumnSelector(key='definition_sd'))
description_is_url_pipe = make_pipeline(ColumnSelector(key='description_is_url'))

pipeline = make_union(viewCount_pipe, likeCount_pipe, dislikeCount_pipe, commentCount_pipe, duration_pipe, caption_pipe, licensedContent_pipe, dimension_2d_pipe, dimension_3d_pipe, definition_hd_pipe, definition_sd_pipe, description_is_url_pipe)

list_ca = ['viewCount' ]#,'likeCount' ,'dislikeCount' ,'commentCount' ,'duration' ,'dimension_2d' ,'dimension_3d' ,'definition_hd' ,'definition_sd' ,'description_is_url']

#,'caption' ,'licensedContent' 

X = df_train[list_ca]

# pipeline.transformer_weights=[1, 1, 1, 1]



# TRAINING

clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
#clf = neighbors.KNeighborsClassifier(1)

Y = df_train['video_category_id'].values
X = pipeline.fit_transform(df_train)
X_test = pipeline.transform(df_test)
print X.shape

alphas = [0.3]
for i in alphas:
    clf.C = i
    sss = StratifiedShuffleSplit(Y, 10, test_size=0.2, random_state=0)
    t0 = time.time()
    scores_sss = cross_val_score(clf, X, Y,scoring='accuracy',cv=sss, n_jobs=-1)
    print (time.time() - t0)
    print ("SSS: acc: %0.4f, std: %0.4f, alpha: %s" %(scores_sss.mean(), scores_sss.std(), i))
    t0 = time.time()
    clf.fit(X,Y)
    print (time.time() - t0)
    y_pred = clf.predict(X_test)
    submit = pd.DataFrame(index=None)
    submit['id'] = df_test['id']
    submit['Pred'] = y_pred
    submit.to_csv('./results/RC_LinearSVC_'+str(scores_sss.mean())+'.csv',sep=';',index=None)

# 2 features tfidf
# C = 1
# SSS: acc: 0.7672, std: 0.0016
# C = 2
# SSS: acc: 0.7421, std: 0.0013

# 4 features tfidf
# SSS: acc: 0.8030, std: 0.0018, alpha: 0.3
# SSS: acc: 0.8007, std: 0.0017, alpha: 0.5
# SSS: acc: 0.7983, std: 0.0015, alpha: 0.7



