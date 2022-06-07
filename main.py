
import pandas as pd
import jieba
import re
import os
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
import torch.nn.functional as F
from torch.autograd import Variable

path="/test/online_shopping_10_cats.csv"
df=pd.read_csv(path)
df.head()

df=df[["review","label"]]
df.head()

print(df.shape)
df.drop_duplicates()

info=re.compile("[0-9a-zA-Z]|作者|当当网|京东|洗发水|蒙牛|衣服|酒店|房间")
df["review"]=df["review"].apply(lambda x:info.sub("",str(x)))   #re.sub用于替换字符串中的匹配项
df["review"].head() #head( )函数读取前五行数据

df["words"]=df["review"].apply(jieba.lcut)
df.head()

words = []
for sentence in df["words"].values:
    for word in sentence:
        words.append(word)
len(words)

words = list(set(words))
words = sorted(words)
len(words)
#embedding lookup要求输入的网络数据是整数。最简单的方法就是创建数据字典：{单词：整数}。然后将评论全部一一对应转换成整数，传入网络。

word2idx = {w:i+1 for i,w in enumerate(words)}
idx2word = {i+1:w for i,w in enumerate(words)}
word2idx['<unk>'] = 0
idx2word[0] = '<unk>'
data = []
label = []

for sentence in df['words']:
    words_to_idx = []
    for word in sentence:
        index = word2idx[word]
        words_to_idx.append(index)
    data.append(words_to_idx)
    #data.append(torch.tensor(words_to_idx))
#label = torch.from_numpy(df['label'].values)
label = df['label'].values
print(np.max([len(x) for x in df["review"]]))
print(np.mean([len(x) for x in df["review"]]))
#数据变长处理
lenlist=[len(i) for i in data]
maxlen=max(lenlist)
maxlen

data_np=np.zeros((62774,1778))
for i in range(len(data)):
    for j in range(len(data[i])):
        data_np[i][j]=data[i][j]
data_np.shape
x_train,x_val,y_train,y_val=train_test_split(data_np,label,test_size=0.2)


class mDataSet(Dataset):
    def __init__(self,x,y):
        self.len = len(x)
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


    def __len__(self):
        return self.len

b=64
#设置用于验证的DataSet
trainDataset=mDataSet(x_train,y_train)
#设置用于训练的DataLoader
train_loader = DataLoader(dataset=trainDataset,   # 传递数据集
                          batch_size=b,     # 小批量的数据大小，每次加载一batch数据
                          shuffle=True,      # 打乱数据之间的顺序
                          )
#设置用于验证的DataSet
validateDataset=mDataSet(x_val,y_val)
#设置用于验证的DataLoader
validate_loader=DataLoader(dataset=validateDataset,   # 传递数据集
                          batch_size=b,     # 小批量的数据大小，每次加载一batch数据
                          shuffle=False,      # 打乱数据之间的顺序
                          )


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(words)+1, 50) #batch*maxlen*50
        self.num_layers=3
        self.hidden_size=100
        self.lstm = nn.LSTM(input_size=50, hidden_size=100, num_layers=3,
                                   batch_first=True)
        self.dropout=nn.Dropout(p=0.5, inplace=False)
        self.fc1 = nn.Linear(in_features=100, out_features=256, bias=True)
        self.fc2 = nn.Linear(in_features=256, out_features=32, bias=True)
        self.fc3 = nn.Linear(in_features=32, out_features=2, bias=True)
        self.sigmoid=nn.Sigmoid()
    def forward(self, input):
        x = self.embedding(input)  # [batch_size, max_len, 100]
        #x = pack_padded_sequence(x,maxlen, batch_first=True)
        h0 = Variable(torch.zeros(self.num_layers , x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers , x.size(0), self.hidden_size)).cuda()
        out, dd = self.lstm(x, (h0, c0))
        #x, (h_n, c_n) = self.lstm(x)
        #output_fw = h_n[-2, :, :]  # 正向最后一次的输出
        #output_bw = h_n[-1, :, :]  # 反向最后一次的输出
        #output = torch.cat([output_fw, output_bw], dim=-1)
       # print(out.shape)
        out = out[:,-1,:].squeeze()
        #out=out.flatten(1)
        out = self.dropout(torch.tanh(self.fc1(out)))
        out = torch.tanh(self.fc2(out))
        #print(out.shape)
        out = self.sigmoid(self.fc3(out))
        #print(out)
        # 可以考虑再添加一个全连接层作为输出层，激活函数处理。
        return out
model=Model().cuda()

# 实例化模型
#model=Model().cuda()
# 定义优化器
optimizer=torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
# 学习率调整（可选）

# 定义损失函数

lossfunc=nn.CrossEntropyLoss().cuda()

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    # 获取前K的索引
    _, pred = output.topk(maxk, 1, True, True) #使用topk来获得前k个的索引
    pred = pred.t() # 进行转置
    # eq按照对应元素进行比较 view(1,-1) 自动转换到行为1,的形状， expand_as(pred) 扩展到pred的shape
    # expand_as 执行按行复制来扩展，要保证列相等
    correct = pred.eq(label.view(1, -1).expand_as(pred)) # 与正确标签序列形成的矩阵相比，生成True/False矩阵
#     print(correct)

    rtn = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0) # 前k行的数据 然后平整到1维度，来计算true的总个数
        rtn.append(correct_k.mul_(100.0 / batch_size)) # mul_() ternsor 的乘法  正确的数目/总的数目 乘以100 变成百分比
    return rtn

epochs=1
def train(train_loader, model, optimizer,lossfunc):
    model.train()
    #data_loader = get_dataloader(True)
    for idx, (data,target) in enumerate(train_loader):
        data=data.long()
        data = Variable(torch.LongTensor(data)).cuda()
        target=target.long()
        target = Variable(torch.tensor(target)).cuda()

        # 梯度清零
        optimizer.zero_grad()


        output = model(data)
        #print(output.shape)
        #print(target.shape)
        loss = lossfunc(output,target)

        # 反向传播
        loss.backward()
        # 梯度更新
        optimizer.step()


        print(idx,loss.item())

if __name__ == '__main__':
    train(train_loader, model, optimizer,lossfunc)
    #print(epoch,loss.item())


































