import argparse
from random import shuffle

#
# def iterTrainSentences(filename):
#     i=0
#     authors = ["twain", "austen", "wilde", "doyle", "poe", "shakespeare"]
#     with open(filename, "r") as f:
#         for line in f:
#             splitted = line.split(";")
#             sentence = " ".join(splitted[:-1]).replace("\n"," ").strip()
#             author = splitted[-1:][0].strip()
#             yield "kn_{}_{} {}".format(i,authors.index(author), sentence)
#             i+=1

def iterTrainSentences(filename):
    i=0
    authors = ["twain", "austen", "wilde", "doyle", "poe", "shakespeare"]
    with open(filename, "r") as f:
        for line in f:
            splitted = line.split(";")
            sentence = " ".join(splitted[:-1]).replace("\n"," ").strip()
            author = splitted[-1:][0].strip()
            yield "{} {}".format(author, sentence)
            i+=1

def iterTestSentences(filename):

    with open(filename, "r") as f:
        for line in f:
            splitted = line.split(";")
            sentence = " ".join(splitted[1:]).replace("\n"," ").strip()
            id_s = splitted[0].strip()

            yield "un_{} {}".format(id_s, sentence)


def main(args):
    with open(args.output, "w") as out:
        datas = [line for line in iterTrainSentences(args.train)] + [line for line in iterTestSentences(args.test)]
        shuffle(datas)
        for line in datas:
            out.write(line+"\n")


parser = argparse.ArgumentParser()

parser.add_argument("--train", default="train_sample.csv", type=str)
parser.add_argument("--test", default="test_sample.csv", type=str)
parser.add_argument("--output", default="out", type=str)
args = parser.parse_args()
main(args)
