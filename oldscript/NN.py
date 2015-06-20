from gensim.models.doc2vec import Doc2Vec
import argparse
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from NNPy import *


def formatNP(trainfile,model):
    i=0
    X = []
    Y = []
    authors = ["twain", "austen", "wilde", "doyle", "poe", "shakespeare"]
    with open(trainfile, "r") as f:
        for line in f:
            splitted = line.split(";")
            sentence = " ".join(splitted[:-1]).replace("\n"," ").strip()
            author = splitted[-1:][0].strip()
            X.append(np.concatenate((model["kn_{}_{}".format(i,authors.index(author))],model["kn_{}_{}".format(i,authors.index(author))])))
            a = np.zeros(6)
            a[authors.index(author)] = 1
            Y.append(a)
            i+=1

    return np.array(X),np.array(Y)





parser = argparse.ArgumentParser()
parser.add_argument("--model", default="w2pC/w2v.bin", type=str)
parser.add_argument("--model_cb", default="w2pC/w2v_cb.bin", type=str)
parser.add_argument("--output", default="w2v.csv", type=str)
args = parser.parse_args()
model = Doc2Vec.load_word2vec_format(args.model, binary=True)
model_cb = Doc2Vec.load_word2vec_format(args.model, binary=True)

net = NetworkModule(HorizontalModule(),)