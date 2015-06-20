from gensim.models.doc2vec import Doc2Vec
import argparse
import numpy as np
from sklearn import metrics
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import ShuffleSplit
from sklearn.ensemble import AdaBoostClassifier

def formatNP(trainfile,model,model2):
    i=0
    X = []
    Y = []
    authors = ["twain", "austen", "wilde", "doyle", "poe", "shakespeare"]
    with open(trainfile, "r") as f:
        for line in f:
            splitted = line.split(";")
            sentence = " ".join(splitted[:-1]).replace("\n"," ").strip()
            author = splitted[-1:][0].strip()
            X.append(np.concatenate((model["kn_{}_{}".format(i,authors.index(author))],model2["kn_{}_{}".format(i,authors.index(author))])))
            Y.append(authors.index(author))
            i+=1

    return np.array(X),np.array(Y)





parser = argparse.ArgumentParser()
parser.add_argument("--model", default="w2pC/w2v.bin", type=str)
parser.add_argument("--model_cb", default="w2pC/w2v_cb.bin", type=str)
parser.add_argument("--output", default="w2v.csv", type=str)
args = parser.parse_args()
model = Doc2Vec.load_word2vec_format(args.model, binary=True)
model_cb = Doc2Vec.load_word2vec_format(args.model, binary=True)

X,Y  = formatNP("train_sample.csv",model,model_cb)
#clf = SVC(C=1.0, kernel='linear',class_weight='auto')

clf = AdaBoostClassifier(n_estimators=100)
cv = ShuffleSplit(len(Y), test_size=0.2)
scores = cross_validation.cross_val_score(clf, X, Y,cv=cv, scoring='accuracy')
print np.mean(scores)
