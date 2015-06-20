# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from Predictor import *
import os



def main(args):
    output_f = args.output_folder
    model_f = args.model_folder

    models =  next(os.walk(model_f))[2]

    print(models)


    for model in models:

        pd_test = pd.read_csv(args.test_csv, header=0, escapechar='\\', quotechar='"', low_memory=False)
        pd_test = pd_test.fillna('')
        mod = Predictor.load(model_f+"/"+model)
        predictions = mod.predict(pd_test)
        print(predictions)


        submit = pd.DataFrame(index=None)
        submit['id']=pd_test['id']
        submit['Pred']=predictions
        submit.to_csv(output_f+"/"+model,sep=';',index=None)





parser = argparse.ArgumentParser()
parser.add_argument("output_folder", type=str)
parser.add_argument("model_folder", type=str)
parser.add_argument("test_csv", type=str)
args = parser.parse_args()

main(args)
