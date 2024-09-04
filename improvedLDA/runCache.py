# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:42:14 2022

@author: 86138
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import cache as Ca
import cacheWithTopic as ldaca
import pickle
#trace=pd.read_csv("trace_detail_8482",header =None,sep=" ")
#trace.columns=["req_time","firstByteTime","LastByteTime","clientIP","ServerIP","clientHeader",
#"ServerHeader","IfModefinedSinceClientHeader","ExpiresServerHeader", "LastModefiedServerHeader","responseHeaderLen",
#"responseDataLan","URLLen", "GET", "URLValue", "HTTP"]

def readdata(path):
    size_sum = 0
    max_size = 0
    supported = 0
    timestamps = []
    urls = set()
    clientsIP = set()
    serversIP = set()
    url_size= defaultdict()
    eventList = []
    f = open(path)
    print ('reading trace file ' + path)
    line = f.readline()
    while line:
        
        
            #fields = [req_time,firstByteTime,LastByteTime,clientIP,ServerIP,clientHeader,
            #ServerHeader,IfModefinedSinceClientHeader,ExpiresServerHeader, LastModefiedServerHeader,responseHeaderLen,
            #responseDataLan,URLLen, GET, URLValue, HTTP/1.0]
        fields = line.rstrip().split(" ")
            
            #if the request is not GET request deny it
        if len(fields)>=15 and fields[13]=='GET':
                #compute response time in microseconds
                #original_responseLen = (int(fields[11]) + int(fields[10]))*8
                #responseLen = self.generate_size(int(fields[10]),int(fields[11]),self.distribution)
            responseLen = (int(fields[10])+int(fields[11]))#数据包真正大小
            if responseLen>0:
                t = fields[0].split(":")
                start_dt = int(t[0])*1000000+int(t[1])
                t = fields[2].split(":")
                end_dt = int(t[0])*1000000+int(t[1])
    #                    t = fields[1].split(":")
    #                    first_byte = int(t[0])*1000000+int(t[1])
                responseTime = (end_dt - start_dt)/1000
                if responseTime==0:
                    responseTime = 1
                cip = fields[3].split(":")[0]
                sip = fields[4].split(":")[0]
                timestamps.append(start_dt/1000)
                clientsIP.add(cip)
                serversIP.add(sip)
                url = fields[14]
                urls.add(url)
                headLen = int(fields[10])
                url_size.setdefault(url,responseLen)
                if responseLen > url_size[url]:
                    url_size[url] = responseLen
                    
                if url.count("gif")>0 or url.count("jpg")>0 or url.count("jpeg")>0 or url.count("mp4")>0 or url.count("mov")>0 or url.count("mp3")>0 or url.count("swf")>0 \
                    or url.count("GIF")>0 or url.count("JPG")>0 or url.count("JPEG")>0 or url.count("MP4")>0 or url.count("MOV")>0 or url.count("MP3")>0 or url.count("SWF")>0 \
                    or url.count("exe")>0 or url.count("PNG")>0 or url.count("zip")>0 or url.count("ZIP")>0 or url.count("tar")>0 or url.count("rar")>0 or url.count("TAR")>0 \
                    or url.count("RAR")>0 or url.count("tar.gz")>0:
                    
                    isSupported = True
                    if max_size<responseLen:
                        max_size = responseLen
                    size_sum = size_sum + responseLen
                    supported = supported + 1
                else:
                    isSupported = False
                eventList.append(pd.Series({'URL':url,'timestamp':start_dt, 'latency':responseTime,\
                        'speed':responseLen/float(responseTime),'clientIP':cip,'serverIP':sip,\
                        'len':responseLen,'headLen':headLen,'isSupported':isSupported}))
           
        
        line = f.readline()
            #print (line )
        #sort eventList on request timestamp
    f.close()
    d=pd.DataFrame(eventList)
    d.to_csv("processed.csv",index=False)
    return eventList
#eventList=readdata("trace_detail_8482")
def findbetterSplit(b,userflied="clientIP",timeflied="timestamp"):
    mint,mean,threeQu,maxt=list(b.describe().loc[["min","mean","75%","max"],[timeflied]].values.flatten())
    maxUser=0
    bettersplit=mean
    for i in list(np.arange(mint,maxt,(maxt-mint)/1000)):
        splitTime=i#划分点
        train=b.loc[b[timeflied]<=splitTime]#从时间的中位数划分
        test=b.loc[b[timeflied]>splitTime]
        train_user=set(train[userflied])
        test_user=set(test[userflied])
        concernUser=train_user & test_user
        if len(concernUser)>maxUser:
            maxUser=len(concernUser)
            bettersplit=splitTime
            
    return bettersplit,maxUser #在这两个时间段都有请求的用户数为201

def getMatrixP(t,userflied="clientIP",timeflied="timestamp",itemfiled="URL"):#获取用户和内容的流行度矩阵
    user=list(set(t[userflied]))
    conttent=list(set(t[itemfiled]))
    P=np.zeros((len(user),len(conttent)))
    for i,r in t.iterrows():
        uid=user.index(r[userflied])
        urlc=conttent.index(r[itemfiled])
        P[uid,urlc]+=1
    P=pd.DataFrame(P,index=user,columns=conttent)
    return P    



def evaluate(predict,reldict,N):
    """评估函数
    predict  模型推荐结果，dict类型，key为 "clientIP"，value为"URL"list
    relldict  各个用户真实点击情况，dict类型，key为 "clientIP"，value为"URL"list
    N为给每个用户推荐的内容数目
    return NetworkLoadRate,CacheRaplaceRate
    NetworkLoadRate float类型， sum（len(x in Ri and not in Pi)）/sum(len(Ri))
    CacheRaplaceRate float类型， Avg((N-count(x in Ri and in Pi))/N)
    """
    n1=0.0
    n2=0.0
    C=[]
    
    for key, value in reldict.items():
        n2=n2+len(value)
        extrat_ask=len(set(value)-set(predict[key]))
        n1=n1+extrat_ask
#        print(len(set(value).intersection(set(predict[key]))))
        C.append((len(set(value).intersection(set(predict[key]))))/N)
    return n1/n2,sum(C)/len(C)

def getCFresult(stepk=[i for i in range(5,30,5)],strategy="user_CF"):
#     alldata=pd.read_csv("processed.csv")
#     b=alldata.sort_values(by="timestamp" , ascending=True) 
# #    splitTime,maxUser=findbetterSplit(b)#划分点 848284875121897.5
# #    print(splitTime)
#     splitTime=848284875121897.5  #两个时间段内都有行为的用户个数为212
#     train=b.loc[b["timestamp"]<=splitTime]#从时间的中位数划分
#     test=b.loc[b["timestamp"]>splitTime]
#     train_user=set(train["clientIP"])
#     test_user=set(test["clientIP"])
#     concernUser=list(train_user & test_user)
#     #找到只含关心的用户的真实请求情况
#     test=test.loc[test["clientIP"].isin(concernUser)]
#     testdict=dict(test.groupby(["clientIP"]).URL.unique().map(lambda x:list(x)))
    
    
    sep = '::'
    alldata=pd.read_csv("../../getdata/RecSysDatasets-master/conversion_tools/ml-1m/ratings.dat",
                            delimiter=sep, header=None, engine='python')
    alldata.columns =["user_id","item_id","rating","timestamp"]
    
    # alldata=alldata[alldata["rating"]>3]
    b=alldata.sort_values(by="timestamp" , ascending=True)
    splitTime,maxUser=findbetterSplit(b, userflied="user_id",timeflied="timestamp")
    print("时间划分点",splitTime)
    train=b.loc[b["timestamp"]<=splitTime]#从时间的中位数划分
    test=b.loc[b["timestamp"]>splitTime]
    train_user=set(train["user_id"])
    test_user=set(test["user_id"])
    concernUser=list(train_user & test_user)
    #找到只含关心的用户的真实请求情况
    test=test.loc[test["user_id"].isin(concernUser)]
    testdict=dict(test.groupby(["user_id"])["item_id"].unique().map(lambda x:list(x)))
    print("需要参与测试的用户数：",len(concernUser),"参与训练的用户数：",train["user_id"].nunique())
    
    
#    print(trainP)
#     if strategy=="svd":
#         cf = CF_svd(k=K, r=3)
#         rec=cf.fit(trainP,concernUser)
# #        train_dataFloat=trainP.values/255.0
        
        
#     elif strategy=="user_CF":
#         cf = CF_user(k=K)
#         rec=cf.fit(trainP,concernUser)
#     elif strategy=="MF":
#         cf = M_F(trainP,K,concernUser)
#         rec=cf.matrix_fac(0.0001,0.0002)     
        
#     elif strategy=="item_CF":
#         cf = CF_knearest(k=K)
#         rec=cf.fit(trainP.T.iloc[:15000].T,concernUser)
#     el
    fresultLarge = open('ResultWith%s.txt'% (strategy),'w')
    for K in stepk:
        if strategy=="LFU":
            indir="trainP.pkl"
            if os.path.exists(indir):
                with open(indir, 'rb') as f:
                    trainP=pickle.load(f)
            else:
                trainP=getMatrixP(train,userflied="user_id",timeflied="timestamp",itemfiled="item_id")#拿到训练集的流行度矩阵，index为用户，columns为url
                pickle.dump(trainP,open(indir,"wb"))
            rec=Ca.LFU(trainP,K,concernUser)
        elif strategy=="LRU":
            rec=Ca.LRU(train,K,concernUser,userfiled="user_id",timefiled="timestamp",itemfiled="item_id")
        elif strategy=="FIFO":
            rec=Ca.FIFO(train,K,concernUser,userfiled="user_id",timefiled="timestamp",itemfiled="item_id")
        elif strategy=="improvedLDA":
            rec=ldaca.LDAcache(train,K,concernUser,modelname="improved lDA",userfiled="user_id",timefiled="timestamp",itemfiled="item_id")
        elif strategy=="LDA":
            rec=ldaca.LDAcache(train,K,concernUser,modelname="lDA",userfiled="user_id",timefiled="timestamp",itemfiled="item_id")
        NetworkLoadRate,CacheRaplaceRate=evaluate(rec,testdict,K)   
        print(strategy," ",K," ",NetworkLoadRate," satisfaction:",CacheRaplaceRate)
        print(strategy," ",K," ",NetworkLoadRate," satisfaction:",CacheRaplaceRate,"\n",file=fresultLarge)
    fresultLarge.close()
        
#    print(NetworkLoadRate,CacheRaplaceRate)
    return 
if __name__ == '__main__':
    stepk=[i for i in range(5,30,5)]
    getCFresult(stepk,"improvedLDA")

# fresultLarge = open('ResultWithLFU.txt','w')
# for k in range(5,30,5):
#     t,rel,NetworkLoadRate,CacheRaplaceRate=getCFresult(k,"LFU")
#     print("LFU"," ",k," ",NetworkLoadRate," ",CacheRaplaceRate)
#     print("LFU"," ",k," ",NetworkLoadRate," ",CacheRaplaceRate,"\n",file=fresultLarge)
# fresultLarge.close()

#import pickle
#def main():
#    polices=["LFU","LRU","FIFO","svd","user_CF","item_CF"]
##    polices=["item_CF"]
#    K=range(100,1100,100)
##    small_K=[1,50,100,150,200,250,300]
#    fresultLarge = open('wellResult.txt','w')  
#    print("The following lines represent Strategy CacheSize  NetworkLoadRate  CacheRaplaceRate\n",file=fresultLarge)
##    fresultsmall = open('item_CF_SmallScale.txt','w')  
##    print("The following lines represent Strategy CacheSize  NetworkLoadRate  CacheRaplaceRate",file=fresultsmall)
#    for p in polices:
#        for k in K:
#            t,rel,NetworkLoadRate,CacheRaplaceRate=getCFresult(k,p) 
#            print(p," ",k," ",NetworkLoadRate," ",CacheRaplaceRate)
#            print(p," ",k," ",NetworkLoadRate," ",CacheRaplaceRate,"\n",file=fresultLarge)
#            pickle.dump(t, open(str(p)+'_'+str(k)+'_.pickle', 'wb'))
##        for k in small_K:
##            t,rel,NetworkLoadRate,CacheRaplaceRate=getCFresult(k,p) 
##            print(p," ",k," ",NetworkLoadRate," ",CacheRaplaceRate)
##            print(p," ",k," ",NetworkLoadRate," ",CacheRaplaceRate,file=fresultsmall)
##            pickle.dump(t, open(p+'_'+str(k)+'_.pickle', 'wb'))
#        print("\n",file=fresultLarge)
#    fresultLarge.close()
##    fresultsmall.close()
#    return
#main()
#print("done！")
#    