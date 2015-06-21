
# coding: utf-8

# # DSG: MultinomialNB

# In[273]:

import numpy as np
import pandas as pd
import os

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import cross_val_score, train_test_split,StratifiedKFold, StratifiedShuffleSplit, KFold
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from lib_DSG import ColumnSelector, DenseTransformer

folder = os.getcwd() ; print folder


# # Importing data

# In[274]:

pd_train = pd.read_csv('./data/train_sample_munged.csv', header=0, escapechar='\\', quotechar='"', low_memory=False, error_bad_lines=False)
pd_test = pd.read_csv('./data/test_sample_munged.csv', header=0, escapechar='\\', quotechar='"', low_memory=False, error_bad_lines=False )


# In[275]:

pd_train = pd_train.fillna('')
pd_test = pd_test.fillna('')


# In[276]:

pd_train.columns




pd_train['title'] = pd_train['title'] + pd_train['description']
pd_train['topics'] = pd_train['topicIds'] + pd_train['relevantTopicIds']

pd_test['title'] = pd_test['title'] + pd_test['description']
pd_test['topics'] = pd_test['topicIds'] + pd_test['relevantTopicIds']


# In[336]:

dico_pattern={'match_lowercase_only':'\\b[a-z]+\\b',
              'match_word':'\\w{4,}',
              'match_word1': '(?u)\\b\\w+\\b',
              'match_3char': '(?u)\\b\\w+\\b\\w+\\b',
              'match_word_punct': '\w+|[,.?!;]',
              'match_NNP': '\\b[A-Z][a-z]+\\b|\\b[A-Z]+\\b',
              'match_punct': "[,.?!;'-]"
             }

stopwords = [u'http', u'com', u'www', u's', u'subscribe'
, u'new', u'like', u'watch', u't', u'2014', u'1'
, u'2013', u'2', u'la', u'en'
, u'world', u'make', u'v', u'check', u'time'
, u'10', u'best', u'3', u'5', u'day', u'y']
tfv_title = TfidfVectorizer(lowercase=True, stop_words='english', token_pattern=dico_pattern["match_word1"], 
                      ngram_range=(1, 2), max_df=0.5, min_df=2, max_features=None, 
                      vocabulary=None, binary=True, norm=u'l2', 
                      use_idf=True, smooth_idf=True, sublinear_tf=True)

tfv_rel_topic = TfidfVectorizer(lowercase=True, stop_words=None, token_pattern=dico_pattern["match_word1"], 
                      ngram_range=(1, 1), max_df=0.5, min_df=2, max_features=None, 
                      vocabulary=None, binary=True, norm=u'l2', 
                      use_idf=True, smooth_idf=True, sublinear_tf=True)

clf = MultinomialNB(alpha=0.05, fit_prior=True, class_prior=None)

title_pipe = make_pipeline(ColumnSelector(key='title'), tfv_title)
desc_pipe = make_pipeline(ColumnSelector(key='topics'), tfv_rel_topic)

pipeline = make_union(title_pipe, desc_pipe)
pipeline.transformer_weights=[1, 1]


# In[337]:

Y = pd_train[u'video_category_id'].values
X = pipeline.fit_transform(pd_train) 
X_test = pipeline.transform(pd_test)
print X.shape
print X_test.shape


# In[338]:

alphas = np.arange(0.01, 0.2, 0.03)
#alphas = [1e-3, 1e-2, 1e-1, 1, 5]
print alphas


# In[ ]:




# In[342]:

from sklearn.cross_validation import KFold
clf.alpha=0.05

l=[]
#sss = StratifiedShuffleSplit(Y, 5, test_size=0.2, random_state=0)
sss = KFold(len(Y), n_folds=5, shuffle=True)
kbest = SelectKBest(chi2, k=100000)
for train_idx, val_idx in sss:
    
    x_train, y_train, x_val, y_val = X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]
    
    x_train = kbest.fit_transform(x_train, y_train)
    x_val = kbest.transform(x_val)
    
    score_sss = clf.fit(x_train, y_train).score(x_val, y_val)
    l.append(score_sss)
    print score_sss
    
print ("SSS: acc: %0.4f, std: %0.4f, alpha: %s" %(np.mean(l), np.std(l), clf.alpha))


# # OTHER MODEL

# numFeat = 40
# 
# Features = np.array(tfv.get_feature_names())
# sorted_indices = np.argsort(np.array(X.sum(0))[0])[::-1]
# rankFeatures = Features[sorted_indices][:numFeat]
# print rankFeatures

# # SUBMIT KAGGLE

# In[ ]:




# In[343]:

X_final = kbest.fit_transform(X,Y)
X_test_final = kbest.transform(X_test)


# In[344]:

clf.alpha = 0.04


# In[345]:

clf.fit(X_final,Y)


# In[346]:

y_pred = clf.predict(X_test_final)
nb_pred_proba = clf.predict_proba(X_test_final)


# In[ ]:




# In[ ]:




# In[347]:

submit = pd.DataFrame(index=None)
submit['id']=pd_test['id']
submit['Pred']=y_pred


# In[ ]:




# In[348]:

submit.to_csv(folder+'/results/0.arda1.csv',sep=';',index=None)


# # FINAL DUMP

# In[ ]:

def getProbaSVM(dist_mat):

    #dist_mat = dist_mat - dist_mat.min()
    dist_mat = (dist_mat - dist_mat.min()) / (dist_mat.max() - dist_mat.min())

    return dist_mat


# In[196]:

import pickle
nb_pred_proba = clf.predict_proba(X_test)
svm_dist=pickle.load(file("/home/arda/Desktop/save.p"))
svm_pred_proba = getProbaSVM(svm_dist)
y_pred_svm = clf.classes_[np.argmax(svm_dist,1)]


# In[ ]:




# In[251]:

alpha = 1. ;beta = 20.
final = (alpha*nb_pred_proba + beta*svm_pred_proba)
y_pred_final = clf.classes_[np.argmax(final,1)]


# In[252]:

#alpha = 3. ;beta = 1. ===>71,24956
#alpha = 1. ;beta = 3. ===>72,02076
#alpha = 1. ;beta = 2. ===>71,77550
#alpha = 1. ;beta = 4. ===>72,22285
#alpha = 1. ;beta = 5. ===>72,42234
#alpha = 1. ;beta = 7. ===>72,77210
#alpha = 1. ;beta = 9. ===>72,97764
#alpha = 1. ;beta = 12. ===>73,1244%


# In[253]:


submit = pd.DataFrame(index=None)
submit['id']=pd_test['id']
submit['Pred']=y_pred_final


# In[254]:

submit.to_csv(folder+'/results/0.arda1.csv',sep=';',index=None)


# In[ ]:



