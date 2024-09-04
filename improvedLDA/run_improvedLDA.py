# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:11:22 2022

@author: 86138
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from Topic import *
from newlda1 import *

def readtxt(datadir="../../getdata/RecSysDatasets-master/conversion_tools/ml-1m"):
    dfdir="../../ml1m-supplement/full_txt.csv"
    if os.path.exists(dfdir):
        df=pd.read_csv(dfdir)
        return df
    
    item_file=datadir+"/movies.dat"
    item_sep ='::'
    origin_data = pd.read_csv(item_file, delimiter=item_sep, header=None, engine='python')
    processed_data = origin_data
    release_year = []
    for i in range(origin_data.shape[0]):
        split_type = origin_data.iloc[i, 2].split('|')
        type_str = ' '.join(split_type)
        processed_data.iloc[i, 2] = type_str
        origin_name = origin_data.iloc[i, 1]
        year_start = origin_name.find('(') + 1
        year_end = origin_name.find(')')
        title_end = year_start - 2
        year = origin_name[year_start:year_end]
        title = origin_name[0: title_end]
        processed_data.iloc[i, 1] = title
        release_year.append(year)
    processed_data.insert(2, 'release_year', pd.Series(release_year))
    
    info_save_path = '../../ml1m-supplement/info.csv'
    supinfo= pd.read_csv(info_save_path, delimiter=",", header=None, engine='python')
    supinfo.columns =["movieid","title","summary"]
    supinfo.drop_duplicates('movieid',inplace = True)
    processed_data.columns =["movieid","title","release_year","genre"]
    df = pd.merge(processed_data, supinfo, how ='inner', on ='movieid')
    df=df.fillna(method='backfill',axis=0)
    df["fulltxt"]=df["title_x"]+" "+df["summary"]
    # df["fulltxt"] = df["title_x"] + " " + df["genre"] + " " + df["summary"]
    # df["fulltxt"]=df["summary"]
    df.to_csv("../../ml1m-supplement/full_txt.csv",index=False)
    return df


if __name__ == '__main__':
    df=readtxt()
    # mylda=LLDA(list(df["summary"]),K=100)
    # mylda.training()
    # mylda.save_top_words(3)
    # mylda.save_top_topics(5)
    mylda=LocalLDA(list(df["fulltxt"]),0.1,0.2,10)
    mylda.run_training(200,3)
    mylda.save_top_words(3)
    mylda.save_top_topics(2)