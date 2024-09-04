import numpy as np
import pandas as pd
import gensim
import re

from numpy.random import multinomial as multinom_draw
from gensim.parsing.preprocessing import STOPWORDS as stopwords
from nltk.stem import WordNetLemmatizer
from scipy.stats import dirichlet,entropy

class LocalLDA:
    def __init__(self, docs, alpha, beta, K,
                 localLDA=True, lemma=True, stem=False):
        self.a = alpha
        self.b = beta
        self.garmdt = 0.01
        self.garmtw = 0.01

        # if localLDA:
        #     sentences = []
        #     for doc in docs:
        #         s = splitdocs(doc)
        #
        #         sentences.extend(s)
        #     docs = sentences
        # print(docs)

        # Preprocess the documents, create word2id mapping & map words to IDs
        prepped_corp = prep_docs(docs, stem=stem, lemma=lemma)
        # print(prepped_corp)
        self.word2id = gensim.corpora.dictionary.Dictionary(prepped_corp)
        self.doc_tups = [self.word2id.doc2bow(doc) for doc in prepped_corp]
        self.doc_tups = [doc for doc in self.doc_tups if len(doc) > 1]

        # Gather some general LDA parameters
        self.V = len(self.word2id)
        self.K = K
        self.D = len(self.doc_tups)

        self.w_to_v = self.word2id.token2id
        self.v_to_w = self.word2id

        self.z_dn = []
        self.n_zk = np.zeros(self.K, dtype=int)
        # self.n_d_k = np.zeros((self.D, self.K))
        # self.n_k_v = np.zeros((self.K, self.V))
        self.n_d_k = np.zeros((self.D, self.K), dtype=int)
        self.n_k_v = np.zeros((self.K, self.V), dtype=int)

        self.docs = []
        self.freqs = []
        for d, doctup in enumerate(self.doc_tups):
            ids, freqs = zip(*doctup)
            self.docs.append(list(ids))
            self.freqs.append(list(freqs))

            zets = np.random.choice(self.K, self.K)
            self.z_dn.append(zets)
            for v, z, freq in zip(ids, zets, freqs):
                self.n_zk[z] += freq
                self.n_d_k[d, z] += freq
                self.n_k_v[z, v] += freq

        self.th_hat = None   # will be filled during training
        self.ph_hat = None   # will be filled during training

    def sample_2gram(self, m, lr=0.0001):
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

        def SGDupdate(cur_ph,cur_th,z_i, z_j, t):
            '''

            Args:
                cur_ph: 主题-词分布
                cur_th: 文本-主题分布
                z_i:样本对中具体的值
                z_j:
                t:

            Returns:

            '''
            grad_dt = -(z_i / z_j + np.log(np.abs(z_j))) * cur_ph[:, t - 1]  # k*1

            theta_m = cur_th[m]
            grad_twi = -np.log(np.abs(z_j)) * z_i * theta_m  # 1*k
            grad_twj = -(z_i / z_j) * theta_m  # 1*k

            cur_ph[:, t - 1] = cur_ph[:, t - 1] - \
                                              lr * grad_twi.T + 2 * self.garmtw * cur_ph[:, t - 1]

            cur_ph[:, t] =cur_ph[:, t] - \
                                          lr * grad_twj.T + 2 * self.garmtw * cur_ph[:, t]

            cur_th[m] =cur_th[m] - \
                                       lr * grad_dt.T + 2 * self.garmdt * cur_th[m]

            return cur_ph,cur_th

        cur_ph = self.get_phi()
        cur_th = self.get_theta()
        co_loss = 0.0  # 共现损失
        if len(self.docs[m]) > 1:
            z = cur_th[m] @ cur_ph  # 拿到当前文本中对各个词的估计概率
            s=(m + 1) / 3
            for t in range(1, len(self.docs[m]),10):
                z_i = z[t - 1]  # 模型估计的词t-1出现概率
                z_j = z[t]  # 模型估计的词t-1出现概率

                # z_i,z_j做归一化

                co_loss += -z_i * np.log(np.abs(z_j))

                cur_ph,cur_th=SGDupdate(cur_ph,cur_th,z_i, z_j, t)
                if s == 1:
                    self.ph_hat = cur_ph
                    self.th_hat = cur_th
                elif s > 1:
                    factor = (s - 1) / s
                    self.ph_hat = factor * self.ph_hat + (1 / s * cur_ph)
                    self.th_hat = factor * self.th_hat + (1 / s * cur_th)

        return co_loss

    def _Klloss(self, lr=0.0000001, step=10):
        '''
        计算当前主题的分布差异

        Returns
        -------
        None.

        '''

        def SGDupdate(cur_ph,i, j, t0, t1):
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
            grad_twi = np.log(t0 / t1) - 1
            grad_twj = -cur_ph[i] / cur_ph[j]

            cur_ph[i] = cur_ph[i] - \
                                       lr * grad_twi + 2 * self.garmtw * cur_ph[i]
            cur_ph[j] = cur_ph[j] - \
                                       lr * grad_twj + 2 * self.garmtw * cur_ph[j]
            return cur_ph

        import random

        kl_loss = 0.0
        if step is None:
            step = self.K * 2
        cur_ph = self.get_phi()

        for i in range(step):
            rd = list(range(self.K))
            random.shuffle(rd)
            sample_k = random.sample(rd, 2)  # 选两个主题
            t0 = cur_ph[sample_k[0]]
            t1 = cur_ph[sample_k[1]]

            KL = entropy(t0, t1)
            kl_loss += KL
            cur_ph=SGDupdate(cur_ph,sample_k[0], sample_k[1], t0, t1)
            s = (i + 1) / 3
            if s == 1:
                self.ph_hat = cur_ph

            elif s > 1:
                factor = (s - 1) / s
                self.ph_hat = factor * self.ph_hat + (1 / s * cur_ph)

        return -kl_loss / (step + 1.0)

    def training_iteration(self):
        docs = self.docs
        freqs = self.freqs

        zdn = self.z_dn
        colosss=0.0
        for d, (doc, freq, zet) in enumerate(zip(docs, freqs, zdn)):
            colosss+=self.sample_2gram(d, 0.0001)
            # print('Running documents # %d ,the coloss is ' % (d + 1),coloss)
            if np.any([np.isnan(x) for x in self.n_k_v]):
                raise ValueError('A nan has creeped into n_k_v')
            if np.any([np.isnan(x) for x in self.n_d_k]):
                raise ValueError('A nan has creeped into n_d_k')

            doc_n_d_k = self.n_d_k[d]
            for n, (v, f, z) in enumerate(zip(doc, freq, zet)):
                self.n_k_v[z, v] -= f
                doc_n_d_k[z] -= f
                self.n_zk[z] -= f

                a = doc_n_d_k + self.a
                num_b = self.n_k_v[:, v] + self.b
                den_b = self.n_zk + self.V * self.b

                prob = a * (num_b / den_b)
                prob /= np.sum(prob)
                if np.any([np.isnan(x) for x in prob]):
                    raise ValueError('A nan has creeped into prob')
                z_new = multinom_draw(1, prob).argmax()

                self.z_dn[d][n] = z_new

                self.n_k_v[z_new, v] += f
                doc_n_d_k[z_new] += f
                self.n_zk[z_new] += f
        return colosss

    def run_training(self, iters, thinning,threod=1000):
        for n in range(iters):
            colosss=self.training_iteration()
            perplexity=self.perplexity()
            if perplexity<threod:
                return
            klloss=self._Klloss()

            print('Running iteration # %d ,the perplexity is %f, coloss is %f,the topic loss is %f' % (n + 1,perplexity,colosss,klloss))
            if (n + 1) % thinning == 0:
                cur_ph = self.get_phi()
                cur_th = self.get_theta()

                s = (n + 1) / thinning
                if s == 1:
                    self.ph_hat = cur_ph
                    self.th_hat = cur_th
                elif s > 1:
                    factor = (s - 1) / s
                    self.ph_hat = factor * self.ph_hat + (1 / s * cur_ph)
                    self.th_hat = factor * self.th_hat + (1 / s * cur_th)
                if np.any(self.ph_hat < 0):
                    raise ValueError('A negative value occurred in self.ph_hat'
                                     'while saving iteration %d ' % n)
                if np.any([np.isnan(x) for x in self.ph_hat]):
                    raise ValueError('A nan has creeped into ph_hat')
                wordload = self.ph_hat.sum(axis=0)
                if np.any([x == 0 for x in wordload]):
                    raise ValueError('A word in dictionary has no z-value')

    def get_phi(self):
        num = self.n_k_v + self.b
        den = self.n_zk[:, np.newaxis] + self.V * self.b
        return num / den

    def get_theta(self):
        num = self.n_d_k + self.a
        den = num.sum(axis=1)[:, np.newaxis]
        return num / den

    def perplexity(self):
        phis = self.get_phi()
        thetas = self.get_theta()

        log_per = l = 0
        for doc, th in zip(self.docs, thetas):
            for w in doc:
                log_per -= np.log(np.abs(np.inner(phis[:, w], th)))
            l += len(doc)
        return np.exp(log_per / l)

    def print_topwords(self, n=10):
        ph = self.get_phi()
        topiclist = []
        for k in range(self.K):
            v_ind = np.argsort(-ph[k, :])[:n]
            top_n = [self.v_to_w[x] for x in v_ind]
            top_n.insert(0, str(k))
            topiclist += [top_n]
        print(topiclist)
        pass
    def save_top_words(self, n_top_words,savedir='topic_wordnewlda.csv'):
        #保存每个主题下权重较高的term
        T_w=pd.DataFrame()
        for topic_idx in range(self.K):
            topic= self.get_phi()[topic_idx]
            T_w["topic"+str(topic_idx)]=[self.v_to_w[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        T_w.to_csv(savedir,index=False,encoding="utf-8_sig")
        print("\n Successful save top words of topic!")
        return

    def save_top_topics(self, n_top_topics, savedir='content_topic.csv'):
        # 保存每个主题下权重较高的topics
        T_w = pd.DataFrame()
        for doc_idx in range(self.D):
            topic = self.get_theta()[doc_idx]
            T_w["content" + str(doc_idx)] = [i for i in topic.argsort()[:-n_top_topics - 1:-1]]
        T_w.to_csv(savedir, index=False, encoding="utf-8_sig")
        print("\n Successful save top topics of topic!")
        return


def prep_docs(docs, stem=False, lemma=True):
    return [prep_doc(doc, stem=stem, lemma=lemma) for doc in docs]


def prep_doc(doc, stem=False, lemma=True):
    doc = doc.lower()
    doc = re.sub('[^\w\s]', '', doc)
    doc = doc.split()
    # remove stopwords and short words
    doc = [word for word in doc if word not in stopwords and len(word) > 1]

    if stem:
        p = gensim.parsing.PorterStemmer()
        return [p.stem(word) for word in doc]
    elif lemma:
        lm = WordNetLemmatizer()
        return [lm.lemmatize(word, pos='v') for word in doc]
    else:
        return doc
