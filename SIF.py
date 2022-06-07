import numpy as np
from gensim.models import Word2Vec
from gensim import models
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# 计算主成分，npc为需要计算的主成分的个数
def compute_pc(X, npc):
    svd = TruncatedSVD(n_components=npc, n_iter=5, random_state=0)
    svd.fit(X)
    return svd.components_


# 去除主成分
def remove_pc(X, npc):
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


class CalSentSim(object):

    def __init__(self, a, dim):
        self.a = a  # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
        self.dim = dim  # 词向量的维数
        self.model_path = 'zhwiki.word2vec.vectors'  # 词向量模型的路径 model/sougou_hotel_cut.model
        self.weight_file = 'SIFData/sougou_hotel_cut_fre.txt'  # 词频的路径，每一行由词语及其频率组成  hotel_comments_cut_fre
        self.sent_file = 'SIFData/2000_pos_cut.txt'  # 句子的路径
        self.sent = []  # 句子的列表
        self.word2weight = {}  # 保存词以及频率的字典

    # 加载Word2Vec模型
    def load_Model(self):
        self.model = models.KeyedVectors.load_word2vec_format(self.model_path)
        print("模型加载完成")
        #print(self.model.wv.__getitem__('富有'))
        return self.model

    # 读取句子组，将之保存到一个列表中
    def read_sentence(self):
        with open(self.sent_file, encoding='utf-8') as sf:
            lines = sf.readlines()
            for line in lines:
                if line:
                    line = line.strip()
                    self.sent.append(line.split())
        M = len(self.sent)
        print('文本数目：%d 个' % M)
        print(self.sent)
        print('***************' * 3)
        return self.sent

    # 读取词频文档，并保存到字典，并且更新字典每一个词的SIF权重
    def save_dict(self):
        # 读取词频文档，并保存到字典
        with open(self.weight_file, encoding='utf-8') as wf:
            lines = wf.readlines()
            N = 0  # N为所有词的词频和
            for line in lines:
                line = line.strip()
                if len(line) > 0:
                    line = line.split()
                    if len(line) == 2:
                        self.word2weight[line[0]] = float(line[1])  # line[0]为词，line[1]为词频
                        N += float(line[1])
                    else:
                        print(line)
        # 更新字典每一个词的SIF权重
        with open('SIFData/sougou_hotel_cut_fre.txt', 'wb') as f:
            for key, value in self.word2weight.items():
                self.word2weight[key] = self.a / (self.a + value / N)
                f.write("{0} {1}".format(key, str(self.word2weight.get(key))).encode("utf-8"))
                f.write('\n'.encode("utf-8"))
        # print(self.word2weight)
        # print('***************' * 3)
        print("权重更新完成")
        return self.word2weight

    # 句子词向量的简单平均
    def ave_no_rem(self):
        self.ave = np.zeros((len(self.sent), self.dim))  # 构建一个零矩阵，每一行用来保存一个句子的向量
        for count, sentence in enumerate(self.sent):
            if sentence:
                for word in sentence:
                    try:
                        w = self.model.wv.__getitem__(word)  # 找到w2c模型中对应单词的词向量
                    except:
                        w = np.zeros(self.dim)  # 如果没有找到该单词的向量，则将它的向量全归零
                        #print("出错了！问题在第%d个句子,单词为%s" % (count, word))
                        #print('************************************************'*10)
                    self.ave[count] = self.ave[count] + w  # 对每一个词的权重向量加总
                self.ave[count] = self.ave[count] / len(sentence)  # 进行平均
        return self.ave

    # 句子词向量的简单平均后去除主成分
    def ave_with_rem(self):
        self.ave_rem = np.zeros((len(self.sent), self.dim))  # 构建一个零矩阵，每一行用来保存一个句子的向量
        for count, sentence in enumerate(self.sent):
            if sentence:
                for word in sentence:
                    try:
                        w = self.model.wv.__getitem__(word)  # 找到w2c模型中对应单词的词向量
                    except:
                        w = np.zeros(self.dim)  # 如果没有找到该单词的向量，则将它的向量全归零
                        #print("出错了！问题在第%d个句子,单词为%s" % (count, word))
                        #print('************************************************'*10)
                    self.ave_rem[count] = self.ave_rem[count] + w  # 对每一个词的权重向量加总
                self.ave_rem[count] = self.ave_rem[count] / len(sentence)  # 进行平均
        # 去除主成分
        npc = 1  # number of principal component
        self.ave_rem = remove_pc(self.ave_rem, npc)
        # print(self.ave_rem)
        # print('***************' * 3)
        return self.ave_rem

    # 未去除主成分前的句子SIF向量
    def sif_no_rem(self):  # 传入的sent是一个句子组，其中每一个句子已经做了分词处理
        self.em = np.zeros((len(self.sent), self.dim))  # 构建一个零矩阵，每一行用来保存一个句子的向量
        for count, sentence in enumerate(self.sent):
            if sentence:
                for word in sentence:
                    try:
                        w = self.model.wv.__getitem__(word)  # 找到w2c模型中对应单词的词向量
                    except:
                        w = np.zeros(self.dim)  # 如果没有找到该单词的向量，则将它的向量全归零
                        #print("出错了！问题在第%d个句子,单词为%s" % (count, word))
                    if self.word2weight.get(word, None):
                        self.em[count] = self.em[count] + np.dot(self.word2weight[word], w)  # 对每一个词的权重向量加总
                self.em[count] = self.em[count] / len(sentence)  # 进行平均
            # print(self.em)
            # print('***************' * 3)
        return self.em

    # 去除主成分后的句子SIF向量
    def sif_with_rem(self):
        self.em_remove = np.zeros((len(self.sent), self.dim))
        for count, sentence in enumerate(self.sent):
            if sentence:
                for word in sentence:
                    try:
                        w = self.model.wv.__getitem__(word)  # 找到它的词向量
                    except:
                        w = np.zeros(self.dim)  # 如果没有找到该单词的向量，则将它的向量全归零
                        #print("出错了！问题在第%d个句子,单词为%s" % (count, word))
                    if self.word2weight.get(word, None):  # 找到它的词的权重
                        self.em_remove[count] = self.em_remove[count] + np.dot(self.word2weight[word], w)  # 对每一个词的权重向量加总
                self.em_remove[count] = self.em_remove[count] / len(sentence)
        # 去除主成分
        npc = 1  # number of principal component
        self.em_remove = remove_pc(self.em_remove, npc)
        # print(self.em_remove)
        # print('***************' * 3)
        return self.em_remove


if __name__ == '__main__':
    a = 0.001  # usually in the range [3e-5, 3e-3]
    dim = 100  # 词向量的维数
    css = CalSentSim(a, dim)
    css.load_Model()
    css.read_sentence()
    css.save_dict()

    print("句子词向量平均相似度：")
    ave = css.ave_no_rem()
    sim_ave = cosine_similarity(ave[1].reshape(1, -1), ave[2].reshape(1, -1))
    print(sim_ave)
    print('***************' * 3)

    print("句子词向量平均去主成分相似度：")
    ave_rem = css.ave_with_rem()
    sim_ave_rem = cosine_similarity(ave_rem[1].reshape(1, -1), ave_rem[2].reshape(1, -1))
    print(sim_ave_rem)
    print('***************' * 3)

    print("SIF加权平均相似度：")
    em = css.sif_no_rem()
    sim_no_rem = cosine_similarity(em[1].reshape(1, -1), em[2].reshape(1, -1))
    print(sim_no_rem)
    print('***************' * 3)

    print("SIF加权平均去主成分相似度：")
    em_remove = css.sif_with_rem()
    sim_with_rem = cosine_similarity(em_remove[1].reshape(1, -1), em_remove[2].reshape(1, -1))
    print(sim_with_rem)
    print('***************' * 3)

