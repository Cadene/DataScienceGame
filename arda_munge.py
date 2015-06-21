# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import time
import os
import datetime
from dateutil import parser

folder = os.getcwd() ; print folder

####################################################################################################################################
# train_sample.csv

train_df = pd.read_csv('./data/train_sample.csv',sep=",", header=0, escapechar='\\', quotechar='"', error_bad_lines=False)


features_transforming(train_df)
print "features_transforming(train_df) :", (time.time() - t0)

train_df.to_csv('./data/train_sample_munged.csv', sep=',', index=None)

####################################################################################################################################
# test_sample.csv

test_df = pd.read_csv('./data/test_sample.csv', sep=",", header=0, escapechar='\\', quotechar='"', error_bad_lines=False)
print "Reading test_sample.csv :", (time.time() - t0)

features_transforming(test_df)
print "features_transforming(test_df) :", (time.time() - t0)

test_df.to_csv('./data/test_sample_munged.csv', sep=',', index=None)




	



