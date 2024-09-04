# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:02:42 2022

@author: 86138
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import sys, re
import numpy as np
from glob import glob
import json
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from scipy.stats import dirichlet,entropy
from scipy.sparse import coo_matrix, bmat
from sklearn.preprocessing import normalize
from string import punctuation as punctuation_en
import random
from nltk.stem import PorterStemmer
import nltk
#nltk.download()
class LLDA:
    def __init__(self,croup, K=15, alpha=None, beta=None):
        self.alpha = alpha   #    torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta = beta    #     torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.K=K#主题个数
        self.garmdt=0.01
        self.garmtw=0.01
        #统计文本中有哪些词及各个词出现的次数
        self.ininit_params(croup)
        
    def preprocess(self,txt):
        def clean_text(text):
            """
            Clean text
            :param text: the string of text
            :return: text string after cleaning
            """
            # acronym
            text = re.sub(r"can\'t", "can not", text)
            text = re.sub(r"cannot", "can not ", text)
            text = re.sub(r"what\'s", "what is", text)
            text = re.sub(r"What\'s", "what is", text)
            text = re.sub(r"\'ve ", " have ", text)
            text = re.sub(r"n\'t", " not ", text)
            text = re.sub(r"i\'m", "i am ", text)
            text = re.sub(r"I\'m", "i am ", text)
            text = re.sub(r"\'re", " are ", text)
            text = re.sub(r"\'d", " would ", text)
            text = re.sub(r"\'ll", " will ", text)
            text = re.sub(r" e mail ", " email ", text)
            text = re.sub(r" e \- mail ", " email ", text)
            text = re.sub(r" e\-mail ", " email ", text)
    
            # spelling correction
            text = re.sub(r"ph\.d", "phd", text)
            text = re.sub(r"PhD", "phd", text)
            text = re.sub(r" e g ", " eg ", text)
            text = re.sub(r" fb ", " facebook ", text)
            text = re.sub(r"facebooks", " facebook ", text)
            text = re.sub(r"facebooking", " facebook ", text)
            text = re.sub(r" usa ", " america ", text)
            text = re.sub(r" us ", " america ", text)
            text = re.sub(r" u s ", " america ", text)
            text = re.sub(r" U\.S\. ", " america ", text)
            text = re.sub(r" US ", " america ", text)
            text = re.sub(r" American ", " america ", text)
            text = re.sub(r" America ", " america ", text)
            text = re.sub(r" mbp ", " macbook-pro ", text)
            text = re.sub(r" mac ", " macbook ", text)
            text = re.sub(r"macbook pro", "macbook-pro", text)
            text = re.sub(r"macbook-pros", "macbook-pro", text)
            text = re.sub(r" 1 ", " one ", text)
            text = re.sub(r" 2 ", " two ", text)
            text = re.sub(r" 3 ", " three ", text)
            text = re.sub(r" 4 ", " four ", text)
            text = re.sub(r" 5 ", " five ", text)
            text = re.sub(r" 6 ", " six ", text)
            text = re.sub(r" 7 ", " seven ", text)
            text = re.sub(r" 8 ", " eight ", text)
            text = re.sub(r" 9 ", " nine ", text)
            text = re.sub(r"googling", " google ", text)
            text = re.sub(r"googled", " google ", text)
            text = re.sub(r"googleable", " google ", text)
            text = re.sub(r"googles", " google ", text)
            text = re.sub(r"dollars", " dollar ", text)
    
            # punctuation
            text = re.sub(r"\+", " + ", text)
            text = re.sub(r"'", " ", text)
            text = re.sub(r"-", " - ", text)
            text = re.sub(r"/", " / ", text)
            text = re.sub(r"\\", " \ ", text)
            text = re.sub(r"=", " = ", text)
            text = re.sub(r"\^", " ^ ", text)
            text = re.sub(r":", " : ", text)
            text = re.sub(r"\.", " . ", text)
            text = re.sub(r",", " , ", text)
            text = re.sub(r"\?", " ? ", text)
            text = re.sub(r"!", " ! ", text)
            text = re.sub(r"\"", " \" ", text)
            text = re.sub(r"&", " & ", text)
            text = re.sub(r"\|", " | ", text)
            text = re.sub(r";", " ; ", text)
            text = re.sub(r"\(", " ( ", text)
            text = re.sub(r"\)", " ( ", text)
    
            # symbol replacement
            text = re.sub(r"&", " and ", text)
            text = re.sub(r"\|", " or ", text)
            text = re.sub(r"=", " equal ", text)
            text = re.sub(r"\+", " plus ", text)
            text = re.sub(r"\$", " dollar ", text)
    
            # remove extra space
            text = ' '.join(text.split())
    
            return text
        from nltk.stem import WordNetLemmatizer
        result = []
        for tokens in txt:
            # print(tokens)
            tokens=tokens.strip()
            tokens = re.sub('[^a-z]+', ' ', str(tokens).strip().lower())
            tokens=tokens.strip().split(" ")
            
            
            filtered = [w for w in tokens if w not in [""]+stopwords.words('english')]
            #仅保留名词或特定POS  
            # print(filtered)
            refiltered =nltk.pos_tag(filtered)
            filtered = [w for w, pos in refiltered if (pos.startswith('NN') or pos.startswith('JJ'))]
            #词干化
            ps =  PorterStemmer()#SnowballStemmer("english")
            filtered = [ps.stem(w) for w in filtered]
            
            result.append(filtered)
        return result
    def getwordcount(self,txt):
        '''
        Parameters
        ----------
        txt : list
            DESCRIPTION.

        Returns  
        -------
        worddict : dict
            各个词的词频.

        '''
        from collections import Counter
        counter=Counter(txt)
        worddict=dict(counter)
        
        return worddict 

    def ininit_params(self,corpus):
        '''
        参数初始化

        Parameters
        ----------
        corpus : TYPE
            DESCRIPTION.

        Returns
        -------
        cntTf : TYPE
            DESCRIPTION.

        '''
        # alltxt=self.load_txt()
        # corpus=alltxt.values()
        corpus =self.preprocess(corpus)#[['I', 'students', 'BeiJin'], ['I', 'want', 'work']]
        from tkinter import _flatten
        all_words=_flatten(corpus)
        self.terms=list(set(all_words))#获取所有的词
        
        
        self.vocabulary = {term: index for index, term in enumerate(self.terms)}
        
        
        self.d=len(self.terms)#总共的词数量
        
            # @field W: the corpus, a list of terms list,
            #   W[m] is the document vector, W[m][n] is the id of the term
        self.W = [[self.vocabulary[term] for term in doc_words] for doc_words in corpus]
        self.M = len(self.W)    #文本个数
        self.WN = len(all_words)#不去重所有文本的长度
        
        
        # @field Lambda: a matrix, shape is M * K,
        # Lambda[m][k] is 1 means topic k is a label of document m
        self.DT = np.ones((self.M, self.K), dtype=float)#文本-主题矩阵
        
        self.TW = np.ones((self.K, self.d), dtype=float)#主题-词矩阵
        
        #初始化dirichlet的均值
        if self.alpha is None:
            self.alpha = [random.uniform(0,1) for _ in range(self.K)]
        elif type(self.alpha) is str and self.alpha == "50_div_K":
            self.alpha = [50.0/self.K for _ in range(self.K)]
        elif type(self.alpha) is float or type(self.alpha) is int:
            self.alpha = [self.alpha for _ in range(self.K)]
        else:
            message = "error alpha_vector: %s" % self.alpha
            raise Exception(message)

        if self.beta is None:
            self.beta = [random.uniform(0,0.7) for _ in range(self.d)]
        elif type(self.beta) is float or type(self.beta) is int:
            self.beta = [self.beta for _ in range(self.d)]
        else:
            message = "error beta: %s" % self.beta
            raise Exception(message)    
            
        #将模型参数初始化为dirichlet分布
        self.DT= dirichlet(alpha =self.alpha).rvs(self.M)
        self.TW= dirichlet(alpha = self.beta).rvs(self.K)
      
        pass
       
  
       
   
    def _multinomial_sample(self,p_vector, random_state=None):
        """
        sample a number from multinomial distribution
        :param p_vector: the probabilities
        :return: a int value
        """
        if random_state is not None:
            return random_state.multinomial(1, p_vector).argmax()
        return np.random.multinomial(1, p_vector).argmax()
    def load_txt(self,file_dir='../data/ml-25m-supplements-master'):
        '''
        Parameters
        ----------
        file_dir : 文件位置  
            DESCRIPTION.

        Returns
        -------
        save_dict : 所有movie 的相关文本
            DESCRIPTION.

        '''
        file_list = glob(file_dir + "/*.txt")
        print("文件个数为：", len(file_list))
    
        save_dict = {}
        for one_file_name in file_list:
            with open(one_file_name, "r", encoding="utf8") as one_file:
                for line in one_file.readlines():
                    line = line.replace("\n", " ").split(":::")
                    if len(line) != 2:
                        continue
                    key = line[0]
                    value = line[1]
                    if key not in save_dict:
                        save_dict[key] = ""
                    save_dict[key] += value
        return save_dict
    # def gradt_chechk(self,x,finf=0.000000001):
    #     '''
    #    保证输入的分布在0到1范围内
    #     Parameters
    #     ----------
    #     grad : TYPE
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     None.

    #     '''

    #     return
    def sample_2gram(self,m,lr=0.0001):
        '''
        采样函数，W中第m个文本采相邻的两个词，然后计算共现概率
        Parameters
        ----------
        m : int
            第m个文本.

        Returns
        -------
        co_loss    上下文信息损失.

        '''
        def SGDupdate(z_i,z_j,t):
            grad_dt=-(z_i/z_j +np.log(z_j))*self.TW[:, t-1]#k*1
            
            theta_m=self.DT[m]
            grad_twi=-np.log(z_j)*z_i*theta_m#1*k
            grad_twj=-(z_i/z_j)*theta_m#1*k
            
            
            self.TW[:,t-1]=self.soft_max(self.TW[:,t-1]-\
                lr*grad_twi.T+2 * self.garmtw *self.TW[:,t-1])
                
            self.TW[:,t]=self.soft_max(self.TW[:,t]-\
                lr*grad_twj.T+2 * self.garmtw *self.TW[:,t])
            
            
            self.DT[m]=self.soft_max(self.DT[m]-\
                lr*grad_dt.T+2 * self.garmdt *self.DT[m])
        
            return
        
        co_loss=0.0#共现损失
        if len(self.W[m])>1:
            z=self._savemenory(m)#拿到当前文本中对各个词的估计概率
            for t in range(1,len(self.W[m])):
                
                z_i= z[t-1] #模型估计的词t-1出现概率
                z_j= z[t]#模型估计的词t-1出现概率
                
                #z_i,z_j做归一化
                
                
                co_loss+=-z_i* np.log(z_j)
                
                SGDupdate(z_i,z_j,t)
            
        return co_loss
    def _savemenory(self,m):
        '''
        算m那个文本对应的词分布，输出归一化结果

        Parameters
        ----------
        m : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # topict_distributes=coo_matrix(self.DT[m])
        # topic_words_distributes=coo_matrix(self.TW)
        X=self.DT[m]@self.TW#改成稀疏矩阵相乘
        # X.resize((1,self.d))
        X_normalized = np.abs(X-np.mean(X))/(np.max(X)-np.min(X))
        X_normalized.resize((self.d))
        return X_normalized
    def soft_max(self,X):
        X_normalized = np.abs(X-np.mean(X))/(np.max(X)-np.min(X))

        t = np.exp(X_normalized)
        a = np.exp(X_normalized) / np.sum(t)
        return a
    def _Klloss(self,lr=0.0000001,step=100):
        '''
        计算当前主题的分布差异

        Returns
        -------
        None.

        '''
       
        def SGDupdate(i,j,t0,t1):
            '''
            update topic distributes

            Parameters
            ----------
            i : TYPE
                DESCRIPTION.
            j : TYPE
                DESCRIPTION.

            Returns
            -------
            None.

            '''
            grad_twi=np.log(t0/t1)-1
            grad_twj=-self.TW[i]/self.TW[j]
            
            self.TW[i]=self.soft_max(self.TW[i]-\
                lr*grad_twi+2 * self.garmtw *self.TW[i])
            self.TW[j]=self.soft_max(self.TW[j]-\
                lr*grad_twj+2 * self.garmtw *self.TW[j]    )
            return
        
        import random
        
        kl_loss=0.0
        if step is None:
            step=self.K*2
        for i in range(step):
            rd = list(range(self.K))
            random.shuffle(rd)
            sample_k=random.sample(rd, 2)#选两个主题
            t0= self.soft_max(self.TW[sample_k[0]])
            t1=self.soft_max(self.TW[sample_k[1]])
            
            KL=entropy(t0, t1)
            kl_loss+=KL
            SGDupdate(sample_k[0],sample_k[1],t0,t1)
        return -kl_loss/(step+1.0)
    def gibssample(self,m):
        real_dict = self.getwordcount(self.W[m])  # 真实的词频
        log_likelihood, per_loss = 0, 0
        y = np.zeros((1, self.d), dtype=float)
        z = self._savemenory(m)

        for t in set(self.W[m]):
            likelihood_t = z[t]  # 拿到出现的各个词估计的词频
            # print("likelihood_t",likelihood_t)
            log_likelihood += -np.log(likelihood_t)
            per_loss += (real_dict[t] / len(self.W[m]) - likelihood_t) ** 2  # real_dict[t]为词t的真实词频
            y[0][t] = real_dict[t] / len(self.W[m])

        return
    def _perplexity(self,m,lr=0.001):
        '''
        Parameters
        ----------
        m : TYPE
            DESCRIPTION.
        lr : TYPE, optional
            DESCRIPTION. The default is 0.01.

        Returns
        -------
        None.

        '''
        def SGDupdate(m,y):
            '''
            每个文本对应的词典真实的情况

            Parameters
            ----------
            m : TYPE
                DESCRIPTION.
            y : TYPE   1*d
                DESCRIPTION.

            Returns
            -------
            None.

            '''
            comterm=2*(self._savemenory(m)-y)#1*d
            
            grad_dtm=comterm@self.TW.T# 1*k
            # print("comterm shape",comterm.shape,"DT[m].T.shape",self.DT[m,:].T.shape)
            dt=self.DT[m]
            dt.resize(1,self.K)
            grad_tw= np.multiply(dt.T,comterm)#TODO
            
            self.DT[m]=self.soft_max(self.DT[m]-\
                lr*grad_dtm+2 * self.garmdt *self.DT[m]   ) 
                
            self.TW=self.soft_max(self.TW -\
                lr*grad_tw+2 * self.garmtw *self.TW  )
            
            return
        
        
        real_dict=self.getwordcount(self.W[m])#真实的词频
        log_likelihood,per_loss=0,0
        y=np.zeros((1,self.d), dtype=float)
        z=self._savemenory(m)
       
        for t in set(self.W[m]):
            likelihood_t =z[t]#拿到出现的各个词估计的词频
            # print("likelihood_t",likelihood_t)
            log_likelihood += -np.log(likelihood_t)
            per_loss+=(real_dict[t]/len(self.W[m])-likelihood_t)**2# real_dict[t]为词t的真实词频
            y[0][t]=real_dict[t]/len(self.W[m])
        SGDupdate(m,y)    
        
        
        return per_loss,log_likelihood
    def _sgd_training(self):
        '''
        把所有的文本送进去训练一次困惑度
            topk 表示把各个主题中概率最大的topk几个词作为代表主题，用于求互信息
        Returns
        -------
        log_likelihood   

        '''
        log_likelihood,per_loss,co_loss,kl_loss=0.0,0.0,0.0,0.0
        for m in range(self.M):
            #calculate  _perplexity and update
            p,l=self._perplexity(m,lr=0.04)
            per_loss+=p
            log_likelihood+=l
            #算共现损失
            co_loss+=self.sample_2gram(m,lr=0.00001)
        # print("self.DT",self.DT)

        # log_likelihood=1.0 * log_likelihood / self.WN#文本的困惑度       
        self.perplexity=1.0 *log_likelihood/ self.WN
        #主题的KL
        kl_loss=self._Klloss(lr=0.001)
        loss=co_loss+kl_loss+per_loss#/ self.WN
        
        print("三个loss的具体值，困惑度的损失：",per_loss,"共现损失:",co_loss*0.01,"KL散度损失：",kl_loss,"困惑度：",log_likelihood)
       

        return loss,log_likelihood
    def updateSGD(self,lr=0.01,garm=0.03):
        '''
        使用随机梯度下降法更新参数

        Returns
        -------
        None.

        '''
        
        return
    def training(self, iteration=100, log=True):
        """
        training this model with gibbs sampling
        :param log: print perplexity after every gibbs sampling if True
        :param iteration: the times of iteration
        :return: None
        """
        idealacc=100
        perplexity=0
        print("start to train......")
        for i in range(iteration):
            if i>0 and perplexity<idealacc:
                break 
            loss,perplexity=self._sgd_training()
            self.perplexity=perplexity
            if log:
                print("after iteration: %s, perplexity: %s" % (i, perplexity))
                
            #更新参数
            # self.updateSGD()
            print("End of iteration:%s"% (i),"*"*20)
        pass
    def save_top_words(self, n_top_words,savedir='topic_word.csv'):
        #保存每个主题下权重较高的term
        T_w=pd.DataFrame()
        for topic_idx in range(self.K):
            topic=self.TW[topic_idx]
            T_w["topic"+str(topic_idx)]=[self.terms[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        T_w.to_csv(savedir,index=False,encoding="utf-8_sig")
        print("\n Successful save top words of topic!")
        return
    def save_top_topics(self, n_top_topics,savedir='content_topic.csv'):
        #保存每个主题下权重较高的topics
        T_w=pd.DataFrame()
        for doc_idx in range(self.M):
            topic=self.DT[doc_idx]
            T_w["content"+str(doc_idx)]=[i for i in topic.argsort()[:-n_top_topics - 1:-1]]
        T_w.to_csv(savedir,index=False,encoding="utf-8_sig")
        print("\n Successful save top words of topic!")
        return