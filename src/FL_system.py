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
    def __init__(self, net, weight_l=[], client_list = []):   # weight_l 为list类型
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
        self.clients = client_list
        
        
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
       
    def get_grad(self, client):
        temp = []
        for a in client.parameters():
            temp.append(a.data)
        return temp
    
    def add_to(self, client):
        if self.sum_w:
            for con, val in enumerate(self.sum_w):
                self.sum_w[con] = self.get_grad(client)[con] + val
        else:
            for i in client.parameters():
                self.sum_w.append(i.data)
        self.num += 1          
                    
                    
    def fl_avg(self):
        assert len(self.sum_w) > 0 
        for a, b in zip(self.nets.parameters(), self.sum_w):
            a.data = b / self.num
        self.sum_w.clear()
        return self.nets
    
    
    def trim_weight (self):
        max = self.weight_l[0]
        for x in self.weight_l:
            if max < x:
                max = x
    
    
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
            # self.pre_weight = copy.deepcopy(nets)
            
            
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

    
    def cal_grad (self):
        # temp = copy.deepcopy(self.net)
        # for t, x in zip(temp.parameters(), self.pre_weight.parameters()):
        #     t.data = t.data - x.data
        temp = []
        for t, x in zip(self.net.parameters(), self.pre_weight):
            temp.append(t.data - x)
        return temp
    
    def update_in_weight (self, weight):
        # for x, y in zip(self.pre_weight.parameters(), self.net.parameters()):
        #     x.data = y.data
        
        self.pre_weight.clear()
        for x in self.net.parameters():
            self.pre_weight.append(x.data)
            
        for x, y in zip(weight, self.net.parameters()):
            y.data = x.data
    
    
    
