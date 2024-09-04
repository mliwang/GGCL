# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:27:38 2022
中高层的缓存
@author: 86138
"""

import pandas as pd
import numpy as np
import os
import random
#得到训练集所有的pid
def getTop3(raw,n):
    t=raw.nlargest(n).index.values
    d={}
    for i in range(n):
        d['t'+str(i+1)]=t[i]
    return pd.Series(d)


def PidUnderTopic(raw,n):
    maintittle=['t'+str(i+1) for i in range(n) ]
    main=list(raw[maintittle].values)
#    del raw['t1','t2','t3']
    raw=raw.drop(maintittle)
    raw[list(set(raw.index)-set(main))]=-1
    return raw

def getcontentTopic(x):
    contentname=x.name
    contentname=int(contentname[7:])
    res=[]
    for t in x.values:
        nc=pd.Series({"content":contentname,"topic":t})
        res.append(nc)       
    return res
def spiltTrainAndTest(datadir="content_topic.csv"):
    takedata=pd.read_csv(datadir,encoding="utf-8_sig", engine='python')#拿到各个内容的主题情况
    # takedata=pd.concat(map(pd.read_csv, "content_topic.csv"))
    print(takedata.info())
    #
    contens=[]
    topics=[]
    for index, row in takedata.iteritems():
        contentname=int(index[7:])
        for t in row.values:
            contens.append(contentname)
            topics.append(t)
    return pd.DataFrame({"content":contens,"topic":topics})#记录各个主题下有哪些content
def ldAspiltTrainAndTest(datadir="doc_topic_lda.csv"):
    n=5
    takedata = pd.read_csv(datadir,index_col='movieid')#拿到各个内容的主题分布
    playlistTopic = takedata.apply(lambda x: getTop3(x, n), axis=1)
    # playlistTopic1 = pd.concat([takedata, playlistTopic], axis=1)
    # playlistTopic1.to_csv("DistributeAndMainTopic.csv")  # 获得主题分布及3个主要主题
    contens = []
    topics = []
    for index, row in playlistTopic.iterrows():
        contentname = index
        for t in row.values:
            contens.append(contentname)
            topics.append(t)
    return pd.DataFrame({"content": contens, "topic": topics})  # 记录各个主题下有哪些content

def LDAcache(train,N,concernUser,modelname="LDA",userfiled="user_id",timefiled="timestamp",itemfiled="item_id"):
    '''
    模拟实际缓存的过程
    1.先拿到所有的待测试用户以及用户历史
    2.根据用户历史找到其过去关心的主题
    3.基于主题查找内容推给用户
    train,用户历史
    N,为最终每个用户需要留存的内容数目
    concernUser，测试目标用户

    Returns
    -------
    缓存列表为dict

    '''
    if modelname=="improved lDA":
        s=spiltTrainAndTest()#内容、主题对照表
    else:
        s=ldAspiltTrainAndTest()
    #每个用户最近点击内容的序列
    dd=dict(train.groupby([userfiled]).apply(lambda x:list(x.sort_values(by=timefiled , ascending=True)[itemfiled])))
    rec={}
    
    for u in concernUser:
        u_c=dd[u]#该用户点击历史情况
        #拿到当前用户所有相关主题
        ctopics=s[s["content"].isin(u_c)]
        ctopics=dict(ctopics["topic"].value_counts())
        ctopics=list(ctopics.keys())
        if len(ctopics)<3:
            pass
        else:
            ctopics=ctopics[:3]
        cache=s[s["topic"].isin(ctopics)]["content"]
        if(len(cache)>=N):
            rec[u]=cache[:N]
        else:
            recom=list(train[itemfiled].values)
            random.shuffle(recom)
            for pc in recom:
                if(len(cache)>=N):
                    rec[u]=cache[:N]
                    break
                if pc not in cache:
                    cache.append(pc)
    return rec
