import torch
import torch.nn as nn
import torch.optim as optim
import re
import jieba
from torch.autograd import Variable

dtype = torch.FloatTensor

class NNLM(nn.Module):
    def __init__(self,m,n_class,n_step,n_hidden):
        super(NNLM, self).__init__()
        self.m=m
        self.n_class=n_class
        self.n_step=n_step
        self.n_hidden=n_hidden
        self.C = nn.Embedding(n_class,m)
        self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
        self.W = nn.Parameter(torch.randn(n_step * m, n_class).type(dtype))
        self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype))
        self.b = nn.Parameter(torch.randn(n_class).type(dtype))

    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, self.n_step * self.m) # [batch_size, n_step * n_class]
        tanh = torch.tanh(self.d + torch.mm(X, self.H)) # [batch_size, n_hidden]
        output = self.b + torch.mm(X, self.W) + torch.mm(tanh, self.U) # [batch_size, n_class]
        return output


class  Processing():
    def __init__(self,doc):
        self.word_dict=dict()
        self.numer_dict=dict()
        self.input_batch=[]
        self.target_batch=[]
        self.size=2
        self.doc=doc
    def doc_pro(self):
        idx=0
        for temp in self.doc:
            doc = re.sub(r'<[^>]+>', ' ', temp)
            doc = re.sub(r'\d', '', doc)
            doc = re.sub('\r', '', doc)
            doc = re.sub('\n', '', doc)
            doc = re.sub('\s*', '', doc)
            doc = re.sub('[()。，:：_,/"`、《》“”—]', '', doc)
            self.doc = jieba.cut(doc, cut_all=False)
            self.doc=list(self.doc)
            doc_len=len(self.doc)
            for word in self.doc:
                self.word_dict.setdefault(word, len(self.word_dict))
                if idx>=self.size and idx<=doc_len-1:
                    n_word=self.doc[idx-2:idx]
                    self.input_batch.append([self.word_dict[i] for i in n_word ])
                    self.target_batch.append(self.word_dict[self.doc[idx]])
                idx+=1
        self.number_dict = {i: w for w, i in (self.word_dict.items())}
    def run(self):
        self.doc_pro()
        self.model = NNLM(5,len(self.word_dict),self.size,5)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        input_batch = Variable(torch.LongTensor(self.input_batch))
        target_batch = Variable(torch.LongTensor(self.target_batch))

        # Training
        for epoch in range(10000):

            optimizer.zero_grad()
            output = self.model(input_batch)

            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            if (epoch + 1) % 1000 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

            loss.backward()
            optimizer.step()

    def Predict(self,input_batch):
        predict=(self.model(input_batch))
        return predict

# Model


    # Predict
#     print(model(input_batch))
#     predict = model(input_batch).data.max(1, keepdim=True)[1]
# #
#
# # Test
    # print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])