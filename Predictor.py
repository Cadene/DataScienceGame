# -*- coding: utf-8 -*-


import pickle
from sklearn.svm import SVC
from sklearn.cross_validation import KFold,cross_val_score
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import NMF
from scipy.sparse import coo_matrix, hstack
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import DistanceMetric

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].values

class DenseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
    def fit(self, X, y=None, **fit_params):
        return self

class Predictor(object):


    def preprocess(self,any_set,is_train):
        return any_set

    def predict(self,test_set):
        pass

    def train(self,train_set,labels):
        pass

    def save(self,model_name):
        pickle.dump(self,open(model_name,"wb"),protocol=2)

    @staticmethod
    def load(model_name):
        return pickle.load(open(model_name,"rb"))


class TitleDescPredictor(Predictor):

    def __init__(self):
        Predictor.__init__(self)


    def preprocess(self,any_set,is_train):

        if is_train:
            dico_pattern={'match_lowercase_only':'\\b[a-z]+\\b',
              'match_word':'\\w{2,}',
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
                              ngram_range=(1, 2), max_df=1.0, min_df=2, max_features=None,
                              vocabulary=None, binary=True, norm=u'l2',
                              use_idf=True, smooth_idf=True, sublinear_tf=True)

            title_pipe = make_pipeline(ColumnSelector(key='title'), tfv_title)
            desc_pipe = make_pipeline(ColumnSelector(key='description'), tfv_desc)
            self.pipeline = make_union(title_pipe, desc_pipe)

            return self.pipeline.fit_transform(any_set)
        else:
            return self.pipeline.transform(any_set)

    def train(self,train_set,labels):
        train_set = self.preprocess(train_set,True)
        self.clf.fit(train_set,labels)

    def train_test(self,clf,train_set,labels):
        train_set = self.preprocess(train_set,True)
        self.clf = clf
        skf = KFold(n=len(labels), n_folds=10, shuffle=True,random_state=None)
        scores_skf = cross_val_score(self.clf, train_set, labels,scoring='accuracy',cv=skf, n_jobs=-1)
        print("Cross val: {}, mean {}, std {}".format(scores_skf, scores_skf.mean(), scores_skf.std()))
        return scores_skf.mean()

    def predict(self,test_set):
        test_set = self.preprocess(test_set,False)
        return self.clf.predict(test_set)



class CatPredictor(Predictor):

    def __init__(self):
        Predictor.__init__(self)

    def preprocess(self,any_set,is_train):
        return any_set[['viewCount', 'likeCount', 'dislikeCount','commentCount' ]]

    def train(self,train_set,labels):
        train_set = self.preprocess(train_set,True)
        self.clf.fit(train_set,labels)

    def train_test(self,clf,train_set,labels):
        train_set = self.preprocess(train_set,True)
        print("Size test {}".format(train_set.shape))
        self.clf = clf
        skf = KFold(n=len(labels), n_folds=10, shuffle=True,random_state=None)
        scores_skf = cross_val_score(self.clf, train_set, labels,scoring='accuracy',cv=skf, n_jobs=-1)
        print("Cross val: {}, mean {}, std {}".format(scores_skf, scores_skf.mean(), scores_skf.std()))
        return  scores_skf.mean()

    def predict(self,test_set):
        test_set = self.preprocess(test_set,False)
        return self.clf.predict(test_set)




class TopicsPredictor(Predictor):

    def __init__(self):
        Predictor.__init__(self)

    def preprocess(self,any_set,is_train):

        if is_train:
            dico_pattern={'match_lowercase_only':'\\b[a-z]+\\b',
              'match_word':'\\w{1,}',
              'match_word1': '(?u)\\b\\w+\\b',
              'match_word_punct': '\w+|[,.?!;]',
              'match_NNP': '\\b[A-Z][a-z]+\\b|\\b[A-Z]+\\b',
              'match_punct': "[,.?!;'-]"
             }

            tfv_topicid = TfidfVectorizer(lowercase=True, stop_words=None, token_pattern=dico_pattern["match_word1"],
                                  ngram_range=(1, 1), max_df=1.0, min_df=2, max_features=None,
                                  vocabulary=None, binary=True, norm=u'l2',
                                  use_idf=True, smooth_idf=True, sublinear_tf=True)

            # tfv_rel_topic = TfidfVectorizer(lowercase=True, stop_words=None, token_pattern=dico_pattern["match_word1"],
            #                       ngram_range=(1, 1), max_df=1.0, min_df=2, max_features=None,
            #                       vocabulary=None, binary=True, norm=u'l2',
            #                       use_idf=True, smooth_idf=True, sublinear_tf=True)


            topicId_pipe = make_pipeline(ColumnSelector(key=u'topicIds'), tfv_topicid)
            # reltopicID_pipe = make_pipeline(ColumnSelector(key=u'relevantTopicIds'), tfv_rel_topic)

            self.pipeline = topicId_pipe #make_union(topicId_pipe, reltopicID_pipe)


            return self.pipeline.fit_transform(any_set)
        else:
            return  self.pipeline.transform(any_set)



class FullTextBasicPredictor(Predictor):

    def __init__(self):
        Predictor.__init__(self)

    def preprocess(self,any_set,is_train):

        if is_train:
            print("hi ?")
            tfv_text = TfidfVectorizer(ngram_range=(1, 2),lowercase=True, max_features=20000,min_df=25,max_df=0.4)
            tfv_topics = TfidfVectorizer(lowercase=True, max_features=500,max_df=0.5)

            title_pipe = make_pipeline(ColumnSelector(key=u'title'), tfv_text)
            topics_pipe = make_pipeline(ColumnSelector(key=u'topicIds'), tfv_topics)
            rel_topic_pipe = make_pipeline(ColumnSelector(key=u'relevantTopicIds'), tfv_topics)
            text_pipe = make_pipeline(ColumnSelector(key=u'description'), tfv_text)

            self.pipeline = make_union(title_pipe, topics_pipe,rel_topic_pipe,text_pipe)
            return self.pipeline.fit_transform(any_set)
        else:
            return  self.pipeline.transform(any_set)

    def train(self,train_set,labels):
        train_set = self.preprocess(train_set,True)
        self.clf.fit(train_set,labels)

    def train_test(self,clf,train_set,labels):
        train_set = self.preprocess(train_set,True)
        print("Size test {}".format(train_set.shape))
        self.clf = clf
        skf = KFold(n=len(labels), n_folds=10, shuffle=True,random_state=None)
        scores_skf = cross_val_score(self.clf, train_set, labels,scoring='accuracy',cv=skf, n_jobs=-1)
        print("Cross val: {}, mean {}, std {}".format(scores_skf, scores_skf.mean(), scores_skf.std()))
        return  scores_skf.mean()

    def predict(self,test_set):
        test_set = self.preprocess(test_set,False)
        print(test_set.shape)
        return self.clf.predict(test_set)



class FullBasicPredictor(Predictor):

    def __init__(self):
        Predictor.__init__(self)

    def preprocess(self,any_set,is_train):
        normalizeX(any_set)
        if is_train:


            tfv_text = TfidfVectorizer(lowercase=True, max_features=5000,min_df=25,max_df=0.4)
            tfv_topics = TfidfVectorizer(lowercase=True, max_features=500,max_df=0.5)
            tfv_desc = TfidfVectorizer(lowercase=True, max_features=2500,min_df=50,max_df=0.4)
            tfv_rel_topics = TfidfVectorizer(lowercase=True, max_features=50,max_df=0.5)

            title_pipe = make_pipeline(ColumnSelector(key=u'title'), tfv_text)
            topics_pipe = make_pipeline(ColumnSelector(key=u'topicIds'), tfv_topics)
            rel_topic_pipe = make_pipeline(ColumnSelector(key=u'relevantTopicIds'), tfv_desc)
            text_pipe = make_pipeline(ColumnSelector(key=u'description'),tfv_desc)


            self.pipeline = make_union(title_pipe, topics_pipe)#,rel_topic_pipe,text_pipe)

            return hstack((self.pipeline.fit_transform(any_set),any_set[['viewCount', 'likeCount', 'dislikeCount','commentCount'  ]]))

        else:

            return hstack((self.pipeline.transform(any_set),any_set[['viewCount', 'likeCount', 'dislikeCount','commentCount' ]]))


    def train(self,train_set,labels):
        train_set = self.preprocess(train_set,True)
        print(train_set.shape)
        self.clf.fit(train_set,labels)

    def train_test(self,clf,train_set,labels):
        train_set = self.preprocess(train_set,True)
        print("Size test {}".format(train_set.shape))
        self.clf = clf
        skf = KFold(n=len(labels), n_folds=10, shuffle=True,random_state=None)
        scores_skf = cross_val_score(self.clf, train_set, labels,scoring='accuracy',cv=skf, n_jobs=-1)
        print("Cross val: {}, mean {}, std {}".format(scores_skf, scores_skf.mean(), scores_skf.std()))
        return  scores_skf.mean()

    def predict(self,test_set):
        test_set = self.preprocess(test_set,False)
        print(test_set.shape)
        return self.clf.predict(test_set)



def normalizeX(X):
    X['viewCount'] += 1
    X['likeCount'] /= X['viewCount']
    X['dislikeCount'] /= X['viewCount']
    X['commentCount'] /= X['viewCount']
    X.fillna(0)



if __name__ == "__main__":

    ##CODE COMMUN
    pd_train = pd.read_csv('./data/train_sample_munged.csv', header=0, escapechar='\\', quotechar='"', low_memory=False)
    pd_test = pd.read_csv('./data/test_sample_munged.csv', header=0, escapechar='\\', quotechar='"', low_memory=False)

    pd_train = pd_train.fillna('')
    pd_test = pd_test.fillna('')

    X = pd_train



    Y = pd_train[u'video_category_id'].values


    ########################################


    pred = FullBasicPredictor()
    score_ks = pred.train_test(DecisionTreeClassifier(criterion="gini"),X,Y) # Tester les paramètres avec ça
    pred.train(X,Y)  # FINAL TRAIN
    pred.save("models/GINI_Tree_01")  # Model SAVE

    pred = FullBasicPredictor()
    score_ks = pred.train_test(DecisionTreeClassifier(criterion="entropy"),X,Y) # Tester les paramètres avec ça
    pred.train(X,Y)  # FINAL TRAIN
    pred.save("models/Entropy_Tree_01")  # Model SAVE

    #
    # pred = FullBasicPredictor()
    # score_ks = pred.train_test(KNeighborsClassifier(n_neighbors=10),X,Y) # Tester les paramètres avec ça
    # pred.train(X,Y)  # FINAL TRAIN
    # pred.save("models/KNN")  # Model SAVE

    #
    # pred = FullTextBasicPredictor()
    # score_ks = pred.train_test(MultinomialNB(alpha=0.1),X,Y) # Tester les paramètres avec ça
    # pred.train(X,Y)  # FINAL TRAIN
    # pred.save("models/0.76_Title_NB_01")  # Model SAVE

    #
    # pred = FullTextBasicPredictor()
    # score_ks = pred.train_test(MultinomialNB(alpha=0.1),X,Y) # Tester les paramètres avec ça
    # pred.train(X,Y)  # FINAL TRAIN
    # pred.save("models/0.76_Title_NB_01")  # Model SAVE

    #
    # pred = FullTextBasicPredictor()
    # score_ks = pred.train_test(MultinomialNB(alpha=0.1),X,Y) # Tester les paramètres avec ça
    # pred.train(X,Y)  # FINAL TRAIN
    # pred.save("models/0.76_Title_NB_01")  # Model SAVE


    #############################################


    #
    # pred = FullTextBasicPredictor()
    # score_ks = pred.train_test(MultinomialNB(alpha=0.1),X,Y) # Tester les paramètres avec ça
    # pred.train(X,Y)  # FINAL TRAIN
    # pred.save("models/0.76_Title_NB_01")  # Model SAVE

    #
    # pred = TitlePredictor()
    # score_ks = pred.train_test(MultinomialNB(alpha=0.1),X,Y) # Tester les paramètres avec ça
    # pred.train(X,Y)  # FINAL TRAIN
    # pred.save("models/0.76_Title_NB_01")  # Model SAVE
    #
    # pred = DescPredictor()
    # score_ks = pred.train_test(MultinomialNB(alpha=0.1),X,Y) # Tester les paramètres avec ça
    # pred.train(X,Y)  # FINAL TRAIN
    # pred.save("models/0.76_Desc_NB_01")  # Model SAVE
    #
    # pred = TopicsPredictor()
    # score_ks = pred.train_test(MultinomialNB(alpha=0.1),X,Y) # Tester les paramètres avec ça
    # pred.train(X,Y)  # FINAL TRAIN
    # pred.save("models/0.76_Topic_NB_01")  # Model SAVE
    # #
    # pred = FullTextBasicPredictor()
    #
    # core_ks = pred.train_test(MultinomialNB(alpha=0.1),X,Y) # Tester les paramètres avec ça

    # for n in range(1,25):
    #     core_ks = pred.train_test(KNeighborsClassifier(n_neighbors=n),X,Y)
    # pred.train(X,Y)  # FINAL TRAIN
    # pred.save("models/0.20_cat")  # Model SAVE


    # pred = FullTextBasicPredictor()
    # for l in ["hinge"]:
    #     for p in ["l2"]:
    #             score_ks = pred.train_test(SGDClassifier(loss=l, penalty=p,n_iter=5, alpha=0.0001,n_jobs=-1),X,Y)
    #

    #pred.train(X,Y)  # FINAL TRAIN
    #pred.save("models/0.76_Title_NB_01")  # Model SAVE

    # RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
    # bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
