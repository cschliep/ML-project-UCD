

import numpy as np
import os
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from pandas import Series, DataFrame

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import glob
#glob.glob('Resultant DBs/df*.csv')
#    ['Resultant DBs/df0-200k.csv',
#    'Resultant DBs/df200k-300k.csv',
#    'Resultant DBs/df300k-400k.csv',
#    'Resultant DBs/df400k-500k.csv',
#    'Resultant DBs/df600k-700k.csv',
#    'Resultant DBs/df700k-800k.csv',
#    'Resultant DBs/df800k-829262.csv']

#data = pd.read_csv("Resultant DBs//df0-200k.csv")


all_dfs = []
for one_filename in glob.glob('Resultant DBs/df*.csv'):
    print(f'Loading {one_filename}')
    new_df = pd.read_csv(one_filename,)

    all_dfs.append(new_df)

print(all_dfs)
len(all_dfs)

df = pd.concat(all_dfs)
df.shape


def naCol(df):
    for column in df.columns:
        print("column name -- {} ,Missing Values - {} ".format(column,df[column].isna().sum()))
naCol(df)





#test-train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = .75)
