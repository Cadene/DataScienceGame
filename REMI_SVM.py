# -*- coding: utf-8 -*-

# REMI_SVM.py

import sklearn.feature_extraction.text as txt
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import hstack

from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import chi2, SelectKBest
from sklearn import neighbors
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, train_test_split,StratifiedKFold, StratifiedShuffleSplit, KFold
from sklearn.metrics import accuracy_score

from lib_DSG import ColumnSelector, DenseTransformer

import time


def getProbaSVM(dist_mat):
    dist_mat = (dist_mat - dist_mat.min()) / (dist_mat.max() - dist_mat.min())
    return dist_mat


t0 = time.time()

df_train = pd.read_csv('./data/train_sample_munged.csv', header=0, escapechar='\\', quotechar='"', low_memory=False)
df_train = df_train.fillna('')

df_train['likeCount'] = (df_train['likeCount']/(df_train['viewCount'] +1) )*100
df_train['dislikeCount'] = (df_train['dislikeCount']/(df_train['viewCount'] +1) )*100
df_train['dislikeCount'] = (df_train['dislikeCount']/(df_train['viewCount'] +1) )*100
df_train['commentCount'] = (df_train['commentCount']/(df_train['viewCount'] +1) )*100
df_train['word'] = df_train['title'] + df_train['description']
df_train['topic'] = df_train['topicIds'] + df_train['relevantTopicIds']

df_test = pd.read_csv('./data/test_sample_munged.csv', header=0, escapechar='\\', quotechar='"', low_memory=False)
df_test = df_test.fillna('')

df_test['likeCount'] = (df_test['likeCount']/(df_test['viewCount'] +1) )*100
df_test['dislikeCount'] = (df_test['dislikeCount']/(df_test['viewCount'] +1) )*100
df_test['dislikeCount'] = (df_test['dislikeCount']/(df_test['viewCount'] +1) )*100
df_test['commentCount'] = (df_test['commentCount']/(df_test['viewCount'] +1) )*100
df_test['word'] = df_test['title'] + df_test['description']
df_test['topic'] = df_test['topicIds'] + df_test['relevantTopicIds']


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


# TF-IDF 2 colomn

tfv_word = TfidfVectorizer(lowercase=True, stop_words=stopwords, token_pattern=dico_pattern["match_word1"], 
                      ngram_range=(1, 2), max_df=0.5, min_df=2, max_features=None, 
                      vocabulary=None, binary=True, norm=u'l2', 
                      use_idf=True, smooth_idf=True, sublinear_tf=True)

tfv_topic = TfidfVectorizer(lowercase=True, stop_words=None, token_pattern=dico_pattern["match_word1"], 
                      ngram_range=(1, 1), max_df=0.5, min_df=2, max_features=None, 
                      vocabulary=None, binary=True, norm=u'l2', 
                      use_idf=True, smooth_idf=True, sublinear_tf=True)

word_pipe = make_pipeline(ColumnSelector(key='word'), tfv_word)
topic_pipe = make_pipeline(ColumnSelector(key='topic'), tfv_topic)

pipeline = make_union(word_pipe, topic_pipe)

#pipeline.transformer_weights[[2,1]]


#X = hstack((pipeline.fit_transform(df_train),df_train[['viewCount', 'likeCount', 'dislikeCount','commentCount', 'dimension_2d', 'definition_hd']]))
X = hstack((pipeline.fit_transform(df_train),df_train[['viewCount', 'likeCount', 'dislikeCount','commentCount']]))
X = X.tocsr()
#X_test = hstack((pipeline.transform(df_test),df_test[['viewCount', 'likeCount', 'dislikeCount','commentCount', 'dimension_2d', 'definition_hd' ]]))
X_test = hstack((pipeline.transform(df_test),df_test[['viewCount', 'likeCount', 'dislikeCount','commentCount' ]]))
X_test = X_test.tocsr()
Y = df_train['video_category_id'].values

#X = pipeline.fit_transform(df_train)
#X_test = pipeline.transform(df_test)
# print X.shape

# TRAINING

clf_svm = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=0.5, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
clf_nb = MultinomialNB(alpha=0.04, fit_prior=True, class_prior=None)

#clf_svm = neighbors.KNeighborsClassifier(1)

clf_svm.C = 0.3
clf_nb.alpha = 0.04

l_svm_score = []
l_nb_score = []
l_blend_score = []

alpha = 1.
beta = 20.

#sss = StratifiedShuffleSplit(Y, 5, test_size=0.2, random_state=0)
sss = KFold(len(Y), n_folds=5, shuffle=True)
kbest = SelectKBest(chi2, k=300000)
for train_idx, val_idx in sss:
    x_train, y_train, x_val, y_val = X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]
    x_train = kbest.fit_transform(x_train, y_train)
    x_val = kbest.transform(x_val)
    #
    clf_svm.fit(x_train, y_train)
    svm_predict_proba = getProbaSVM(clf_svm.decision_function(x_val))
    score_svm = accuracy_score(clf_svm.predict(x_val), y_val)
    l_svm_score.append(score_svm)
    #
    clf_nb.fit(x_train, y_train)
    nb_predict_proba = clf_nb.predict_proba(x_val)
    score_nb = accuracy_score(clf_nb.predict(x_val), y_val)
    l_nb_score.append(score_nb)
    #
    blend_mat = alpha*nb_predict_proba + beta*svm_predict_proba
    y_pred_blend = clf_svm.classes_[np.argmax(blend_mat,1)]
    score_blend = accuracy_score(y_pred_blend, y_val)
    l_blend_score.append(score_blend)
    #
    print score_svm, score_nb, score_blend
    

print ("SVM: acc: %0.4f, std: %0.4f, alpha: %s" %(np.mean(l_svm_score), np.std(l_svm_score), clf_svm.C))
print ("NB: acc: %0.4f, std: %0.4f, alpha: %s" %(np.mean(l_nb_score), np.std(l_nb_score), clf_nb.alpha))
print ("Blend: acc: %0.4f, std: %0.4f, c: %s, alpha: %s" %(np.mean(l_blend_score), np.std(l_blend_score), clf_svm.C, clf_nb.alpha))

X_final = kbest.fit_transform(X,Y)
X_test_final = kbest.transform(X_test)

clf_svm.fit(X_final,Y)
y_pred_svm = clf_svm.predict(X_test_final)
mat_dist = clf_svm.decision_function(X_test_final)
y_pred_svm_proba = getProbaSVM(mat_dist)
submit_svm = pd.DataFrame(index=None)
submit_svm['id'] = df_test['id']
submit_svm['Pred'] = y_pred_svm
submit_svm.to_csv('./results/final_SVM.csv',sep=';',index=None)

clf_nb.fit(X_final,Y)
y_pred_nb = clf_nb.predict(X_test_final)
y_pred_nb_proba = clf_nb.predict_proba(X_test_final)
submit_nb = pd.DataFrame(index=None)
submit_nb['id'] = df_test['id']
submit_nb['Pred'] = y_pred_nb
submit_nb.to_csv('./results/final_NB.csv',sep=';',index=None)

blend_mat = alpha*y_pred_nb_proba + beta*y_pred_svm_proba
y_pred_blend = clf_svm.classes_[np.argmax(blend_mat,1)]
submit_blend = pd.DataFrame(index=None)
submit_blend['id'] = df_test['id']
submit_blend['Pred'] = y_pred_blend
submit_blend.to_csv('./results/final_blend.csv',sep=';',index=None)



def to_number(s):
  try:
      s1 = float(s)
      return s1
  except ValueError:
      return s







