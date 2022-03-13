from calendar import c
from distutils.log import debug
from http import client
import torch
import torch.nn as nn
import copy

argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)

def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))

def evaluate_accuracy(net, data_iter):
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    
class FL_server():
    def __init__(self, net, weight_l=[]):   # weight_l 为list类型
        self.list_p = []
        self.sum_w = []
        if weight_l:
            self.list_p = weight_l  
            for li in weight_l:
                for da in li:   
                    self.sum_w.append(da)
        else:
            temp = []
            for i in net.parameters():
                temp.append(i.data)              
            self.list_p.append(temp)            
            self.sum_w = temp
            
        self.num = len(self.list_p)        
        
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_normal_(m.weight)
            elif type(m) == nn.BatchNorm2d:
                m.weight.data.normal_(1.0,0.02) 
                m.bias.data.fill_(0)
        self.nets = copy.deepcopy(net)     
        self.nets.apply(init_weights)
               
    def __str__(self):
        return ("net: {}; num: {}; list: {}; adder: {}".format(self.nets, self.num, self.list_p, self.sum_w))
        
    def __getitem__(self, index):
        assert (index < self.num)
        return self.list_p[index]
    
    def __len__(self):
        return self.num

    def convent_list(self, net):
        temp = []
        for datas in net.parameters():
            temp.append(datas.data)
        return temp
               
    def get_sum(self, clients):     # clients 为list类型
        temp = []
        for a in clients :
            if temp:
                for i in range(len(a)):
                    temp[i] += a[i]
            else:
                temp = a
        return temp
    
    def add_to(self, client):
        if(type(client)==list):
            if self.sum_w:
                assert len(self.sum_w) == len(client)
                for a, b in zip(self.sum_w, client):
                    a += b
            else:
                self.sum_w = client
        else:
            temp = self.convent_list(client)
            if self.sum_w:
                assert len(self.sum_w) == len(temp)
                for con, val in enumerate(self.sum_w):
                    self.sum_w[con] = temp[con] + val
            else:
                    self.sum_w = temp
        self.num += 1          
                                                  
    
class FL_clients():
    def __init__(self, nets, train_data, test_data, loss_func, updata) -> None:
            self.net = copy.deepcopy(nets)
            self.data = train_data
            self.test = test_data
            self.loss = loss_func
            self.updater = updata
            self.pre_weight = []
            for i in nets.parameters():
                self.pre_weight.append(i.data)
                       
    def train_batch (self):
        self.net.train()
        metric = Accumulator(3)
        
        for X, y in self.data:
            # 计算梯度并更新参数
            y_hat = self.net(X)
            l = self.loss(y_hat, y)
            if isinstance(self.updater, torch.optim.Optimizer): 
                # 使用PyTorch内置的优化器和损失函数
                self.updater.zero_grad()
                l.backward()
                self.updater.step()
                metric.add(float(l) * len(y), accuracy(y_hat, y),
                            y.size().numel())
            else:
                # 使用定制的优化器和损失函数
                l.sum().backward()
                self.updater(X.shape[0])
                metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        # 返回训练损失和训练准确率
        return metric[0] / metric[2], metric[1] / metric[2]
    
    def train_epoch (self, num):
        for epoch in range(num):
            train_metrics = self.train_batch()
            test_acc = evaluate_accuracy(self.net, self.test)
            train_loss, train_acc = train_metrics
            print("in epoch:%d, train_loss = %.3f, train_acc = %.3f, test_acc = %.3f" %(epoch, train_loss, train_acc, test_acc))
            
    def show_weight (self):
        temp = []
        for t in self.net.parameters():
            temp.append(t.data)
        return temp
    
    def show_grad (self):
        temp = []
        for t, x in zip(self.net.parameters(), self.pre_weight):
            temp.append(t.data - x)
        return temp
    
    def update_in_weight (self, weight):
        self.pre_weight.clear()
        for x in self.net.parameters():
            self.pre_weight.append(x.data)
            
        for x, y in zip(weight, self.net.parameters()):
            y.data = x
    
    def update_in_grad (self, grad):
        for x, y in zip(grad, self.net.parameters()):
            y.data = x + y.data
            
        self.pre_weight.clear()
        for x in self.net.parameters():
            self.pre_weight.append(x.data)

    def apply_net (self, nets):
        self.net = copy.deepcopy(nets)


    
class FL_system():
    def __init__(self, train_data, test_data, net, loss, updater, num_clients, num_epoch) -> None:
        self.train = train_data
        self.test = test_data
        self.epoch = num_epoch
        self.net = net
        self.num = num_clients
        # self.loss = loss
        # self.updater = updater
        self.client = []
        self.server = FL_server(net)
        for i in range(num_clients):
            self.client.append(FL_clients(net, train_data, test_data, loss, updater))
            self.client[i].update_in_weight(self.server.sum_w)
            
    def round_w (self) -> list:
        temp_weight = []
        for i in range(self.num):
            self.client[i].train_epoch(1)
            temp_weight.append(self.client[i].show_weight())
        return temp_weight
        
    def round_g (self) -> list:
        temp_grad = []
        for i in range(self.num):
            self.client[i].train_epoch(1)
            temp_grad.append(self.client[i].show_grad())
        return temp_grad
    
    def update (self):
        # define own update function
        pass
    
    def fl_avg_w(self):
        sum = []
        tmp = self.round_w()
        for i in tmp:
            if sum:
                for a, b in zip(sum, i):
                    a += b
            else:
                sum = i
                
        for i, para in enumerate(self.server.nets.parameters()) :
            sum[i] /= len(tmp)
            para.data = sum[i]
        
        for i in self.num:
            client[i].apply_net(self.server.nets)