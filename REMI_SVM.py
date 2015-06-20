# -*- coding: utf-8 -*-

# REMI_SVM.py

import sklearn.feature_extraction.text as txt
import pandas as pd

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, train_test_split,StratifiedKFold, StratifiedShuffleSplit

from lib_DSG import ColumnSelector, DenseTransformer

import time
t0 = time.time()

df_train = pd.read_csv('./data/train_sample_munged.csv', header=0, escapechar='\\', quotechar='"', low_memory=False)
df_train = df_train.fillna('')

df_test = pd.read_csv('./data/test_sample_munged.csv', header=0, escapechar='\\', quotechar='"', low_memory=False)
df_test = df_test.fillna('')

# processing

dico_pattern={'match_lowercase_only':'\\b[a-z]+\\b',
              'match_word':'\\w{1,}',
              'match_word1': '(?u)\\b\\w+\\b',
              'match_word_punct': '\w+|[,.?!;]',
              'match_NNP': '\\b[A-Z][a-z]+\\b|\\b[A-Z]+\\b',
              'match_punct': "[,.?!;'-]"
             }
tfv_title = TfidfVectorizer(lowercase=True, stop_words='english', token_pattern=dico_pattern["match_word1"], 
                      ngram_range=(1, 2), max_df=1.0, min_df=2, max_features=None, 
                      vocabulary=None, binary=True, norm=u'l2', 
                      use_idf=True, smooth_idf=True, sublinear_tf=True)

tfv_desc = TfidfVectorizer(lowercase=True, stop_words='english', token_pattern=dico_pattern["match_word1"], 
                      ngram_range=(1, 2), max_df=1.0, min_df=5, max_features=None, 
                      vocabulary=None, binary=True, norm=u'l2', 
                      use_idf=True, smooth_idf=True, sublinear_tf=True)

tfv_topicid = TfidfVectorizer(lowercase=True, stop_words=None, token_pattern=dico_pattern["match_word1"], 
                      ngram_range=(1, 1), max_df=1.0, min_df=2, max_features=None, 
                      vocabulary=None, binary=True, norm=u'l2', 
                      use_idf=True, smooth_idf=True, sublinear_tf=True)

tfv_rel_topic = TfidfVectorizer(lowercase=True, stop_words=None, token_pattern=dico_pattern["match_word1"], 
                      ngram_range=(1, 1), max_df=1.0, min_df=2, max_features=None, 
                      vocabulary=None, binary=True, norm=u'l2', 
                      use_idf=True, smooth_idf=True, sublinear_tf=True)

# clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
# gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
# shrinking=True, tol=0.001, verbose=False)

clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)

title_pipe = make_pipeline(ColumnSelector(key='title'), tfv_title)
desc_pipe = make_pipeline(ColumnSelector(key='description'), tfv_desc)
topicId_pipe = make_pipeline(ColumnSelector(key='topicIds'), tfv_topicid)
reltopicID_pipe = make_pipeline(ColumnSelector(key='relevantTopicIds'), tfv_rel_topic)
viewCount_pipe = make_pipeline(ColumnSelector(key='viewCount'))

pipeline = make_union(title_pipe, desc_pipe, topicId_pipe, reltopicID_pipe, viewCount_pipe)

# pipeline.transformer_weights=[1, 1, 1, 1]

Y = df_train['video_category_id'].values
X = pipeline.fit_transform(df_train)
print X.shape
X_test = pipeline.transform(df_test)

alphas = [0.3, 0.5]
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





