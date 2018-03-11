import pandas as pd
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

datafile = 'data/model.xls'

data = pd.read_excel(datafile)

data = data.as_matrix()

shuffle(data)

#print(data)

p = 0.8

train = data[:int(len(data)*p),:]
test = data[int(len(data)*p):,:]

train_x = Variable(torch.FloatTensor(train[:,:3]))
train_y = Variable(torch.FloatTensor(train[:,3].reshape(232,1)))

print(train_y.shape)

test_x = train[:,:3]
test_y = test[:,3]





class MyNet(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(MyNet,self).__init__()
        self.hidden = nn.Linear(n_feature,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        x = F.sigmoid(x)
        return x







if __name__ == '__main__':
    print('init net')
    net = MyNet(n_feature = 3, n_hidden = 10, n_output = 1)

    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.5)  # 传入 net 的所有参数, 学习率
    loss_func = torch.nn.BCELoss()


    for t in range(100):
        prediction = net(train_x)     # 喂给 net 训练数据 x, 输出预测值
        loss = loss_func(prediction, train_y)     # 计算两者的误差
    
        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        loss.backward()         # 误差反向传播, 计算参数更新值
        optimizer.step()

    #torch.save(net, 'net.pkl')

    #net2 = torch.load('net.pkl')
    #prediction = net2(test_x)


