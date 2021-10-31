# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:08:41 2019

@author: magic
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('breast-cancer-data.csv',index_col = 1) #col as index
del df['id']
#df.index=df['diagnosis']
#del df['diagnosis']
#df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})

print(df.head())

from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
pca.fit(df)
existing_2d = pca.transform(df)

existing_df_2d = pd.DataFrame(existing_2d)
existing_df_2d.index = df.index
existing_df_2d.columns = ['PC1','PC2','PC3']
print(existing_df_2d.head())
print(pca.explained_variance_ratio_)
