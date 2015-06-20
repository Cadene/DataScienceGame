# coding: utf8
#from gensim.models.doc2vec import Doc2Vec
import argparse

changed = 0 # Compteur de changement.
def maj(choosed, line_test ,words,wc):
    """
    fonction qui compare un auteur prédit avec les noms propres.
    choosed = auteur prédit
    line_test = texte
    words = dictionnaire {nom_propre:auteur}
    wc = dictionnaire {nom_propre: nombre d'apparition dans le train_set}
    """
    global changed
    splitted = line_test.split(";") #on split sur ;
    sentence = " ".join(splitted[1:]).replace("\n"," ").strip() #on récupère la phrase et on enlève le saut de ligne
    choosed = choosed.strip() # on enlève les whitespaces
    new = None # premier auteur nom propre
    dif = False # si changement d'auteur par rapport à celui prédit -> vrai 
    last_word = None
    i=0
    for word in sentence.split():  # tous les mots
    
        if word in words.keys(): # si le mot est un nom propre
            if words[word] != choosed or dif: # et que son auteur est différent de celui prédit
                #print("{}, {}/{}".format(sentence,words[word],choosed))
                if new is None: #première fois
                    dif = True
                    new = words[word]
                    last_word = word
                else:
                    if new == words[word]: # deuxième fois => ON EST SUR QUE C'EST BON = CHANGEMENT
                        print("DOUBLE, CHANGING, {}".format(changed))
                        changed += 1
                        return new

                    else: # deux auteurs différents, on change pas
                        print("DISCREPENCY - NOT CHANGING")
                        return choosed

    if new is not None and wc[last_word] > 100: # un seul nom propre, qui a une apparition élevé => on change
        changed += 1
        print("SINGLE HIGH FREQUENCY CHANGING {}, {}/{} {}".format(changed,new,choosed,sentence))
        return new
    else:
        return choosed



def buildNameModel(train):
    """
    crée le dictionnaire des noms propres avec leurs compte
    """
    Mwords = {}
    Mwordscount = {}
    skip = set()
    with open(train, "r") as f:
        for line in f:
            splitted = line.split(";")
            sentence = " ".join(splitted[:-1]).replace("\n"," ").strip()
            author = splitted[-1:][0].strip()

            for word in sentence.split():
                if word[0].isupper(): #commence par une majuscule
                    if word in skip:
                        continue
                    elif word in Mwords and author != Mwords[word]: # un mot a deux auteur différents, on supprime
                        del Mwords[word]
                        del Mwordscount[word]
                        skip.add(word)
                    else:
                        if word in Mwords:
                            Mwordscount[word] += 1
                        else:
                            Mwords[word] = author
                            Mwordscount[word] = 1

    for word in Mwordscount:
        if Mwordscount[word] < 1:
            del Mwords[word]

    #print Mwords
    return (Mwords,Mwordscount)

def main(args):
    global changed
    words,wc=buildNameModel(args.train)

    with open(args.output, "w") as out:
        out.write("Id;Pred\n")
        for line_pred,line_test in zip(open(args.csv, "r"),open(args.test,"r")):
            id_a, choosed = line_pred.split(";")
            out.write("{};{}\n".format(id_a, maj(choosed,line_test,words,wc)))
    print("Changed {} ".format(changed))



parser = argparse.ArgumentParser()
parser.add_argument("--train",default="train_sample.csv",type=str)
parser.add_argument("--test",default="test_sample.csv",type=str)
parser.add_argument("--csv",default="best.csv",type=str)
parser.add_argument("--output",default="verif_name.csv",type=str)

args = parser.parse_args()
main(args)
