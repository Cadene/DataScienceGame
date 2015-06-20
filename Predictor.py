import pickle
from sklearn.svm import SVC
from sklearn.cross_validation import KFold,cross_val_score
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin

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

    def preprocess(self,any_set):
        return any_set

    def predict(self,test_set):
        test_set = self.preprocess(test_set)
        pass

    def train(self,train_set,labels):
        pass

    def save(self,model_name):
        pickle.dump(self,open(model_name,"wb"),protocol=2)

    @staticmethod
    def load(self,model_name):
        return pickle.load(open(model_name,"rb"))


class TitleDescpredictor(Predictor):

    def preprocess(self,any_set):
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
                              ngram_range=(1, 2), max_df=1.0, min_df=2, max_features=None,
                              vocabulary=None, binary=True, norm=u'l2',
                              use_idf=True, smooth_idf=True, sublinear_tf=True)


        title_pipe = make_pipeline(ColumnSelector(key='title'), tfv_title)
        desc_pipe = make_pipeline(ColumnSelector(key='description'), tfv_desc)
        pipeline = make_union(title_pipe, desc_pipe)

        return pipeline.fit_transform(any_set)

    def train(self,train_set,labels):
        train_set = self.preprocess(train_set)
        self.clf.fit(train_set,labels)

    def train_test(self,clf,train_set,labels):
        train_set = self.preprocess(train_set)
        self.clf = clf
        skf = KFold(n=len(labels), n_folds=10, shuffle=True,random_state=None)
        scores_skf = cross_val_score(self.clf, train_set, labels,scoring='accuracy',cv=skf, n_jobs=-1)
        print("Cross val: {}, mean {}, std {}".format(scores_skf, scores_skf.mean(), scores_skf.std()))

    def predict(self,test_set):
        test_set = self.preprocess(test_set)
        self.clf.predict(test_set)



if __name__ == "__main__":

    ##CODE COMMUN
    pd_train = pd.read_csv('./data/train_sample_munged.csv', header=0, escapechar='\\', quotechar='"', low_memory=False)
    pd_test = pd.read_csv('./data/test_sample_munged.csv', header=0, escapechar='\\', quotechar='"', low_memory=False)

    pd_train = pd_train.fillna('')
    pd_test = pd_test.fillna('')

    X = pd_train
    Y = pd_train[u'video_category_id'].values

    pred = TitleDescpredictor()
    pred.train_test(MultinomialNB(alpha=0.1),X,Y) # Tester les paramètres avec ça


    pred.train(X,Y)  # FINAL TRAIN


    pred.save("0.76_Title_Desc_NB_01")  # Model SAVE
