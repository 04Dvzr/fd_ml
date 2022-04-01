import torch, torchvision
import os, struct, random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from functools import reduce


def read_file(dir_img : str, dir_lb : str, dst : str, num_kinds : int):
    fb = open(os.path.abspath(dir_lb), 'rb')
    dst_path = os.path.abspath(dst)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    file_img = []
    for i in range(num_kinds):
        paths = os.path.join(dst_path, str(i))
        if not os.path.exists(paths):
            os.mkdir(paths)
        n = os.path.join(paths, 'image')
        file_img.append(open(n, 'wb+'))
    fb.seek(4)    
    label_num, = struct.unpack('>i', fb.read(4))
    with open(os.path.abspath(dir_img), 'rb') as fm:
        fm.seek(4)
        img_num, = struct.unpack('>i', fm.read(4))
        assert label_num == img_num
        fm.seek(16)
        for _ in range(label_num):
            lb, = struct.unpack('>b', fb.read(1))
            file_img[lb].write(fm.read(784))
               
    for i in range(num_kinds):
        file_img[i].close()
    fb.close()    
        
#生成指定和为maxValue的num个随机数(适用于多0，种类数目大于抽样个数的)
def random_fixed (maxValue, num, seed, sigma = 2.0):
    a = [0]
    for x in range(num):
        sum_v = reduce(lambda x,y:x+y,a)
        random.seed(seed + x)
        a.append(random.randint(0, round((maxValue - sum_v) / random.uniform(1, sigma))))
    if reduce(lambda x,y:x+y,a) < maxValue:
        random.seed(seed)
        a[random.randint(1, num)] += 1
    return a  
    
 #生成指定和为maxValue的num个随机数(适用于平均化，种类数目小于抽样个数的)   
def random_num_with_fix_total(maxValue, num, seed):
    '''生成总和固定的整数序列
    maxvalue: 序列总和
    num：要生成的整数个数'''
    random.seed(seed)
    a = random.sample(range(0,maxValue), k=num-1) # 在0~maxValue之间，采集k个数据
    a.append(0)   # 加上数据开头
    a.append(maxValue)
    a = sorted(a)
    b = [ a[i]-a[i-1] for i in range(1, len(a)) ] # 列表推导式，计算列表中每两个数之间的间隔
    return b

def sample(dir_img : str, dst : str, nums_client : int, num_file : int, iid : bool = True, iid_rate : int = None, seed : int = 123456789):
    img_path = os.path.abspath(dir_img)
    dst_path = os.path.abspath(dst)
    label_list = os.listdir(img_path)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    file_list = []
    sum_list = []
    for i in label_list:
        file_list.append(open(os.path.join(img_path, i, 'image'),'rb'))
        sum_list.append(os.path.getsize(os.path.join(img_path, i, 'image')) // 784)
        
    for i in range(nums_client):
        f = open(os.path.join(dst_path, 'data_{s}'.format(s = i)), 'wb')
        f_l = open(os.path.join(dst_path, 'data_{s}_label'.format(s = i)), 'w')
        index_list = [0] * len(label_list)
        if iid: 
            index_list = [num_file // len(label_list) - 1] * len(label_list)
            for a in range(num_file % len(label_list) + len(label_list)):
                random.seed(seed + i * a + a)
                index_list[random.randint(0, len(label_list) - 1)] += 1               
        else:
            assert iid_rate != None or iid_rate <= 100 or iid_rate > 0
            random.seed(seed + i)
            main_l = random.randint(0, len(label_list) - 1)
            if (num_file - iid_rate * num_file // 100) <= (len(label_list) - 1) * 1.2:
                index_list = random_fixed(num_file - iid_rate * num_file // 100, len(label_list) - 1, seed + i)
            else :
                index_list = random_num_with_fix_total(num_file - iid_rate * num_file // 100, len(label_list) - 1, seed + i)

            index_list.insert(main_l, iid_rate * num_file // 100)
        
        for da in index_list:
            f_l.write(str(da) + '\n')
        f_l.close()
        
        for id, num in enumerate(index_list):
            a = random.sample(range(0, sum_list[id]), k=num)
            for ind in a:
                file_list[id].seek(ind * 784)
                f.write(file_list[id].read(784))
            print(a, end='\n\n')
        
        f.close()
        
    for f_s in file_list:
        f_s.close()

class MNIST_like(Dataset):
    """Some Information about MyDataset"""
    def __init__(self, dir, lb_dir, trans):
        super(MNIST_like, self).__init__()
        self.path = os.path.abspath(dir)
        self.label_list = []
        with open(os.path.abspath(lb_dir), 'r') as fb:
            for i in fb:
                self.label_list.append(int(i))       
        self.trans = trans
                             
    def __getitem__(self, index):
        img = np.empty([784,], dtype = np.uint8) 
        with open(self.path, 'rb', buffering=0) as f:
            f.seek(index * 784)
            for i in range(784):
                data, = struct.unpack('>b', f.read(1))
                img[i] = data
                
        img = img.reshape([28,28],)
        count = index
        for id, num in enumerate(self.label_list):
            count -= num
            if count < 0:
                break        
        return self.trans(img) if self.trans is not None else data, id

    def __len__(self):
        return os.path.getsize(self.path) // 784
        
def creat_data_MNIST (dir : str, lb_dir : str, transform = None, batch_size : int = 16, shuffles : bool = False):
    if not transform:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    data = MNIST_like(dir, lb_dir, transform)
    return DataLoader(data, batch_size, shuffle=shuffles, num_workers=4)    
    
def data_folder(dir : str, transform = None):
    file_list = os.listdir(os.path.abspath(dir))
    
    datas = []
    for fs in [file_list[i:i + 2] for i in range(0, len(file_list), 2)]:
        datas.append(MNIST_like(os.path.join(dir, fs[0]), os.path.join(dir, fs[1]), transform))
    return datas    
    