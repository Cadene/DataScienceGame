# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

df = pd.read_csv('./submit_prediction/ms_random-forest.csv', sep=';')
df['id'] = range(1,df.shape[0]+1)

dfdict = {}
dfdict = df['id']
dfdict = df['prediction']

final_df = pd.DataFrame(dfdict)

final_df.to_csv('./submit/ms_random-forest.csv', sep=';', index=None, cols=["id","prediction"])