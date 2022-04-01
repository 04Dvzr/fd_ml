import torch, torchvision
import torch.nn as nn
from torch.utils import data as _data
import copy, os, sys, time, re
import torch.utils.tensorboard as tb
from PIL import Image

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

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]

def creat_data (train_dir : str, batch_size : int = 16, shuffles : bool = False, default : bool = False):
    if shuffles:        
        # rans = torchvision.transforms.Compose([torchvision.transforms.Grayscale(), torchvision.transforms.Resize((28, 28),), torchvision.transforms.ToTensor()])
        rans = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224),), torchvision.transforms.AutoAugment(), torchvision.transforms.ToTensor()])
    else:
        rans = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224),), torchvision.transforms.ToTensor()])
        # rans = torchvision.transforms.Compose([torchvision.transforms.Grayscale(), torchvision.transforms.Resize((28, 28),), torchvision.transforms.ToTensor()])
    
    if default:
        tr = torchvision.datasets.ImageFolder(os.path.abspath(train_dir), transform=rans)
    else:
        tr = classifer_dataset_folder(os.path.abspath(train_dir), trans = rans)
        
    train = _data.DataLoader(tr, batch_size, shuffle=shuffles, num_workers=4)
    
    return train

class classifer_dataset(_data.Dataset):
    def __init__(self, path : str, trans = None, t_trans = None) -> None:
        super().__init__()
        self.path = os.path.abspath(path)
        self.sample_list = os.listdir(self.path) 
        self.trans = trans
        self.t_trans = t_trans


    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path, self.sample_list[index])).convert(mode = "RGB")
        if self.trans is not None:
            img = self.trans(img)
            
        if re.search('\\\\', self.path):
            key_s = '\\'
        else:
            key_s = '/'
            
        if self.t_trans is not None:
            label = self.t_trans(self.path.split(key_s)[-1])
        else:
            label = self.path.split(key_s)[-1][1:]
        return img, label

    def __len__(self):
        return len(self.sample_list)
 
 
class classifer_dataset_folder(_data.Dataset):
    def __init__(self, path : str, trans = None, t_trans = None) -> None:
        super().__init__()
        self.path = os.path.abspath(path)
        self.kind_list = os.listdir(self.path)
        assert len(self.kind_list) > 0
        self.trans = trans
        self.t_trans = t_trans
        self.sample_list = []
        self.sample_num_index = []
        for i, dir in enumerate(self.kind_list) :
            self.sample_list.append(os.listdir(os.path.join(self.path, dir)))
            self.sample_num_index.append(len(self.sample_list[i]))
        
    def __getitem__(self, index):
        tem = index
        for i, num in enumerate(self.sample_num_index):
            tem -= num
            if(tem < 0):
                count = i
                break
            else:
                index = tem
                
        img = Image.open(os.path.join(self.path, self.kind_list[count], self.sample_list[count][index])).convert(mode = "RGB")
        # img = img2.convert(mode='RGB')
        if self.trans is not None:
            img = self.trans(img)
                        
        if self.t_trans is not None:
            label = self.t_trans(self.kind_list[count])
        else:
            label = torch.tensor(int(self.kind_list[count][1:]))
        return img, label
        
        
        
    def __len__(self):
        return sum(self.sample_num_index)
 
    
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
    def __init__(self, net, test_data):   # weight_l 为list类型
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_normal_(m.weight)
            elif type(m) == nn.BatchNorm2d:
                m.weight.data.normal_(1.0,0.02) 
                m.bias.data.fill_(0)
                
        self.net = copy.deepcopy(net)     
        self.net.apply(init_weights)
        self.data = test_data      
               
    def eval_test(self) -> str:
        test_acc = evaluate_accuracy(self.net, self.data)                                                                      
        return "server test_acc = %.3f" %(test_acc)
    
    def apply_dict(self, dict : dict):
        self.net.load_state_dict(copy.deepcopy(dict))
            
    def show_weight (self) -> list:
        temp = []
        for t in self.net.parameters():
            temp.append(t.data.clone())
        return temp

    
class FL_clients():
    def __init__(self, nets, train_data : str, test_data : str, loss_func, updata : str) -> None:
            self.net = copy.deepcopy(nets)
            self.data = train_data
            self.test = test_data
            self.loss = loss_func
            self.updater = updata #in str
            self.pre_dict = copy.deepcopy(nets.state_dict())
                       
    def train_batch (self):
        metric = Accumulator(3)
        updater = eval(self.updater)
        self.net.train()
        for X, y in self.data:
            # 计算梯度并更新参数
            y_hat = self.net(X)
            l = self.loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer): 
                # 使用PyTorch内置的优化器和损失函数
                updater.zero_grad()
                l.backward()
                updater.step()
                metric.add(float(l) * len(y), accuracy(y_hat, y),
                            y.size().numel())
            else:
                # 使用定制的优化器和损失函数
                l.sum().backward()
                updater(X.shape[0])
                metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        # 返回训练损失和训练准确率
        return metric[0] / metric[2], metric[1] / metric[2]
 
    def train_batch_gpu(self, device):
        
        updater = eval(self.updater)
        metric = Accumulator(3)
        self.net.train()
        for X, y in self.data:
            updater.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = self.net(X)
            l = self.loss(y_hat, y)
            l.backward()
            updater.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        self.net.to('cpu')
        return train_l, train_acc    
    
    def train_epoch (self, device=None) -> str:
        if device == None or device == 'cpu':
            train_loss, train_acc = self.train_batch()            
            test_acc = evaluate_accuracy(self.net, self.test)
        else:
            self.net.to(device)
            train_loss, train_acc = self.train_batch_gpu(device)
            test_acc = evaluate_accuracy_gpu(self.net, self.test)
            self.net.to('cpu')
            torch.cuda.empty_cache()
        # print(", train_loss = %.3f, train_acc = %.3f, test_acc = %.3f" % (train_loss, train_acc, test_acc))
        return ", train_loss = %.3f, train_acc = %.3f, test_acc = %.3f\n" % (train_loss, train_acc, test_acc)
    
    def train_epoch_mul (self, num, device=None) -> str:
        train_loss = 0.0
        train_acc = 0.0
        test_acc = 0.0
        if device == None or device == 'cpu':
            for _ in range(num):
                loss, acc = self.train_batch()
                train_loss += loss  
                train_acc  += acc
                test_acc += evaluate_accuracy(self.net, self.test)
        else:
            for _ in range(num):
                self.net.to(device)
                loss, acc = self.train_batch_gpu(device) 
                train_loss += loss  
                train_acc  += acc
                test_acc += evaluate_accuracy_gpu(self.net, self.test)
                self.net.to('cpu')
                torch.cuda.empty_cache()
        # print(", train_loss = %.3f, train_acc = %.3f, test_acc = %.3f" %(train_loss/num, train_acc/num, test_acc/num))
        return ", train_loss = %.3f, train_acc = %.3f, test_acc = %.3f\n" % (train_loss/num, train_acc/num, test_acc/num)
            
    def show_weight (self) -> list:
        temp = []
        for t in self.net.parameters():
            temp.append(t.data.clone())
        return temp
    
    def show_grad (self) -> dict :
        tmp = copy.deepcopy(self.net.state_dict())
        for a, b in zip(self.pre_dict, tmp):
            if re.search('.*running.*', a) or re.search('.*batch.*', a):
                continue
            tmp[b] -= self.pre_dict[a]   
        return tmp
    
    def apply_net (self, nets):
        self.net = copy.deepcopy(nets)
        
    def apply_dict(self, dict : dict):
        self.net.load_state_dict(copy.deepcopy(dict))
               
    def eval_test(self):
        test_acc = evaluate_accuracy(self.net, self.test)                                                                      
        return "server test_acc = %.3f" %(test_acc)
        
   
class FL_system():
    def __init__(self, train_data_dir : str, test_data_dir : str, net, loss, updater : str, num_clients : int, num_sub_epoch : int = 1 , log_on : bool = False) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train = []
        tr_list = os.listdir(os.path.abspath(train_data_dir))
        assert len(tr_list) >= num_clients
        for dir in tr_list:
            self.train.append(creat_data(os.path.join(os.path.abspath(train_data_dir), dir) , 32, shuffles=True))       
        self.test = creat_data(test_data_dir, 32)
        self.sub_epoch = num_sub_epoch
        self.num = num_clients
        # self.loss = loss
        self.updater = copy.deepcopy(updater)  #in str format
        self.client = []
        self.server = FL_server(net, self.test)
        temp = copy.deepcopy(self.server.net.state_dict())
            
        for i in range(num_clients):
            self.client.append(FL_clients(net, self.train[i], self.test, loss, updater))
            self.client[i].apply_dict(temp)
            
        self.f = None
        if log_on:
            if not os.path.exists('./log'):
                os.mkdir('./log')
            self.f = open(os.path.join(os.path.abspath('./log'), time.strftime("%Y-%m-%d_%H-%M-%S")+'_train.log'), 'w')
        print('training on', self.device)
        if self.f :
            self.f.write('training on ' + str(self.device) + '\n')
            
    def round_w (self) -> list:
        temp_weight = []
        if self.sub_epoch > 1:
            for i in range(self.num):
                st = self.client[i].train_epoch(self.device)
                print("in client %d" %i + st, end='')
                if self.f :
                    self.f.write("in client %d" %i + st)                
                temp_weight.append(copy.deepcopy(self.client[i].net.state_dict()))
        else:
            for i in range(self.num):
                st = self.client[i].train_epoch_mul(self.sub_epoch, self.device)
                print("in client %d" %i + st, end='')
                if self.f :
                    self.f.write("in client %d" %i + st)                        
                temp_weight.append(copy.deepcopy(self.client[i].net.state_dict()))
        return temp_weight
        
    def round_g (self) -> list:# need to fix
        temp_grad = []
        for i in range(self.num):
            self.client[i].train_epoch()
            temp_grad.append(self.client[i].show_grad())
        return temp_grad
    
    def update (self):
        # define own update function
        pass
    
    def fl_avg_w(self):
        sum = dict()
        tmp = self.round_w()
        if self.device == 'cuda:0':
            torch.cuda.empty_cache()
        for i in tmp:
            if sum:
                for a, b in zip(i, sum):
                    assert a == b
                    sum[b] += i[a]
            else:
                sum = copy.deepcopy(i)           
                            
        for ae in sum:
            sum[ae] = torch.div(sum[ae], len(tmp))
            
        self.server.apply_dict(sum)
        val = self.server.eval_test()
        print(val)
        if self.f :            
            self.f.write(val + '\n')                        
        
        for i in range(self.num):
            self.client[i].apply_dict(sum)
