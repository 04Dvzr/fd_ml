import os, re
from numpy import append
import torch.utils.tensorboard as tb


class log_pro():
    def __init__(self, log_dir : str) -> None:
        self.path = os.path.abspath(log_dir)
        
        self.log_list = []
        for  dir in os.listdir(self.path):
            if os.path.isfile(os.path.join(self.path, dir)):
                self.log_list.append(dir)
        assert len(self.log_list) > 0
        self.file = open(os.path.join(self.path, self.log_list[-1]), 'r')
        self.num = self.find_num(self.file)
        self.sec_path = os.path.join(self.path, 'cleared_log_{st}'.format(st = self.log_list[-1][:-4]))
        if not os.path.exists(self.sec_path):
            os.mkdir(self.sec_path)
        pp = os.path.join(self.sec_path, 'tensorboard')
        if not os.path.exists(pp):
            os.mkdir(pp)
        self.writer = tb.SummaryWriter(pp)
        
    def find_num(self, file):
        state = 0
        counts = 0
        for line in file:
            if re.search('.*epoch', line):
                state += 1
                continue
            if state == 1:
                counts += 1
            elif state > 1:
                break
        return counts - 1
    
    def show_list(self, out = None):
        if out:
            return self.log_list
        else:
            print(self.log_list)
            self.show_current_file()
            
    def show_current_file(self):        
            if re.search('\\\\', self.sec_path):
                key_s = '\\'
            else:
                key_s = '/'
            print('current file is: {st}'.format(st = self.sec_path.split(key_s)[-1][12:]))
            
    def change_file(self, num : int):
        self.file.close()
        self.writer.close()
        
        self.file = open(os.path.join(self.path, self.log_list[num - 1]), 'r')
        self.sec_path = os.path.join(self.path, 'cleared_log_{st}'.format(st = self.log_list[num - 1][:-4]))
        if not os.path.exists(self.sec_path):
            os.mkdir(self.sec_path)
        pp = os.path.join(self.sec_path, 'tensorboard')
        if not os.path.exists(pp):
            os.mkdir(pp)
        self.writer = tb.SummaryWriter(pp)
        self.num = self.find_num(self.file)
        self.show_current_file()
        
    def clean(self):
        client_list = []
        ser_f = open(os.path.join(self.sec_path, 'sever.log'), 'w')
        for i in range(self.num):
            client_list.append(open(os.path.join(self.sec_path, 'client_{n}.log'.format(n=i)), 'w'))
        epoch = 0
        self.file.seek(0)
        for strs in self.file:
            if re.search('.*epoch', strs):
                epoch = int(re.split(' ', strs)[2].strip('\n').strip())
                for i in client_list:
                    i.write(strs)
                ser_f.write(strs)
                
            elif re.search('.*client', strs):
                t = 11
                a = strs[10]
                while strs[t] != ',':
                    a += strs[t]
                    t += 1    
                client_list[int(a)].write(strs[t + 2:])
                self.writer.add_scalar('client_{n} train_loss'.format(n = int(a)), float(strs[t + 15:t + 20]), epoch)
                self.writer.add_scalar('client_{n} train_acc'.format(n = int(a)), float(strs[t + 34:t + 39]), epoch)
                self.writer.add_scalar('client_{n} test_acc'.format(n = int(a)), float(strs[t + 52:t + 57]), epoch)
                
            elif re.search('server.*', strs):
                ser_f.write(strs)
                self.writer.add_scalar('server test_acc', float(strs[18:23]), epoch)
                
        for i in client_list:
            i.close()
        ser_f.close()
        self.file.close()
        self.writer.close()