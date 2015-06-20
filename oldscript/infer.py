from gensim.models.doc2vec import Doc2Vec
import argparse


def authorSimMax(model, id ,choosed):
    authors = ["twain", "austen", "wilde", "doyle", "poe", "shakespeare"]
    max_v = -1
    max_i = -1
    choosed = choosed.strip()
    for i, author in enumerate(authors):
        sim = model.similarity(author, "un_" + str(id))
        if sim > max_v:
            max_v = sim
            max_i = i

        if authors[max_i] == choosed:
            return choosed
        if authors[max_i] != choosed:
            if model.similarity(choosed, "un_" + str(id)) < -0.5:
                return authors[max_i]
            else:
                return choosed




def main(args):
    model = Doc2Vec.load_word2vec_format(args.model, binary=True)

    with open(args.output, "w") as out:
        out.write("Id;Pred\n")
        for line in open(args.csv, "r"):

            id_a, choosed = line.split(";")

            out.write("{};{}\n".format(id_a, authorSimMax(model, id_a,choosed)))


parser = argparse.ArgumentParser()

parser.add_argument("--model", default="w2pC/vect.bin", type=str)
parser.add_argument("--output", default="w2v.csv", type=str)
parser.add_argument("--csv",default="best.csv",type=str)
args = parser.parse_args()
main(args)
