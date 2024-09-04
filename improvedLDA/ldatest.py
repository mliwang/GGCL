# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 16:07:18 2022
test our lda
@author: 86138
"""
import pandas as pd
import re
from Topic import *
from string import punctuation as punctuation_en
# 
def gettxt(path):
    data = open(path,"r",encoding="utf-8").read()
    stxt=[]
    alltxt=data.split("******")
    n=0
    for t in alltxt:
        # regex=re.compile("[^\u4e00-\u9fa5，。、；！：（）《》“”？_]")#去掉文中汉字标点以外的东西
        t=t.replace("\n", " ")
        t = re.sub("[{}]+".format(punctuation_en), "", t) 
        # n=n+len(t)
        # cs=" ".join(t)
        # t= [w for w in t if w not in stopwords.words('english')]
        stxt.append(t)
    return stxt



txt=gettxt("test/shakespeare.txt")

mylda=LLDA(txt)
mylda.training()
mylda.save_top_words(10)