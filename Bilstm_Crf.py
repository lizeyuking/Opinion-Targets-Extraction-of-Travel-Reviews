import json
import  re
import kashgari
from kashgari.embeddings import BERTEmbedding         #Kashgari 内置了bert预训练语言模型处理模块
from tensorflow.python import keras
from kashgari.callbacks import EvalCallBack
def deal(path):
    def seg_char(sent):
        """
        把句子按字分开，不破坏英文结构
        """
        # 首先分割 英文 以及英文和标点
        pattern_char_1 = re.compile(r'([\W])')
        parts = pattern_char_1.split(sent)
        parts = [p for p in parts if len(p.strip())>0]
        # 分割中文
        pattern = re.compile(r'([\u4e00-\u9fa5])')
        chars = pattern.split(sent)
        chars = [w for w in chars if len(w.strip())>0]
        return chars
    def place(zi,mu):
        """查询子字符串在大字符串中的所有位置"""
        len1 = len(zi)
        pl = []
        for each in range(len(mu)-len1):
            if mu[each:each+len1] == zi:   #找出与子字符串首字符相同的字符位置
                pl.append(each)
        return pl

    with open(path, 'rb') as load_f:
        data=json.load(load_f)
    x=[]
    y=[]
    for d in data:
        x.append(d["s"])
        y.append(d["ot"])
    X=[]
    Y=[]
    # 生成
    for i in range(0,len(x)):
        x1=['O' for i in range(len(x[i]))]
        t = place(y[i],x[i])
        for j in t:
            x1[j]='B'
            for k in range(1,len(y[i])):
                x1[j+k]='I'
        Y.append(x1)

    for i in x:
        X.append(list(seg_char(i)))


    return X,Y
train_x, train_y = deal("data/train_json.json")  #训练用数据集
X,Y=deal("data/test_json.json")
#将初始test数据集切片，使Train:valid:test ≈ 9:1:2
valid_x,valid_y = X[:5893] , Y[:5893]
test_x,test_y = X[5893:] , Y[5893:]


# bert_embed = BERTEmbedding('Chinese-Word-Vectors-master/',
#                            task=kashgari.LABELING,
#                            sequence_length=100)

from kashgari.tasks.labeling import BiLSTM_CRF_Model

# 还可以选择 `CNN_LSTM_Model`, `BiLSTM_Model`, `BiGRU_Model` 或 `BiGRU_CRF_Model`

# model = BiLSTM_CRF_Model(bert_embed)
model = BiLSTM_CRF_Model()
tf_board_callback = keras.callbacks.TensorBoard(log_dir='log', update_freq=1000)
# 这是 Kashgari 内置回调函数，会在训练过程计算精确度，召回率和 F1
eval_callback = EvalCallBack(kash_model=model,
                             valid_x=valid_x,
                             valid_y=valid_y,
                             step=1)
model.fit(train_x,
          train_y,
          x_validate=valid_x,
          y_validate=valid_y,
          epochs=50,
          batch_size=512,
          callbacks=[eval_callback, tf_board_callback])
#评估模型
model.evaluate(test_x, test_y)

model.save('BilstmCrf_model')
