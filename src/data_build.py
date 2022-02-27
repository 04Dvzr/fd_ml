import os
import random
import shutil
import sys
import math
from functools import reduce

#从source_dir复制num个jpg图片到data_dir
def data_copy(source_dir, data_dir, num, seed = 123456789):
    files = os.listdir(os.path.abspath(source_dir))
    if num >= len(files) / 2:
        num_files = math.floor(len(files) / 2)
        num_loop = math.floor(num / num_files)
        remain_files = num - num_loop * num_files
    else : 
        num_loop = 1
        num_files = num
        remain_files = 0
    counter = 0
    for t in range(num_loop):
        random.seed(seed + t)
        a = random.sample(range(len(files)), num_files)
        for x in range(num_files) :
            counter += 1
            shutil.copyfile(os.path.join(os.path.abspath(source_dir), files[a[x]]), os.path.join(os.path.abspath(data_dir), (str(counter) + ".jpg")))
    random.seed(seed + num_loop + 1)
    a = random.sample(range(len(files)), remain_files)
    for x in range(remain_files):
        counter += 1 
        shutil.copyfile(os.path.join(os.path.abspath(source_dir), files[a[x]]), os.path.join(os.path.abspath(data_dir), (str(counter) + ".jpg")))
    # print("copy done")
    
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

    
def data_create_niid(image_folder_dir, dst_folder_dir, num_of_batch, batch_size, IID_rate, seed = 123456789):     
    img_kinds_list = os.listdir(os.path.abspath(image_folder_dir))
    if IID_rate < 50:
        print("make rate higher than 50")
        return -1
    for x in range(num_of_batch):
        data_dir = os.path.join(os.path.abspath(dst_folder_dir), (r'data_non_iid_' + str(x)))
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f'创建文件夹{data_dir}成功！')
        
        random.seed(seed + x)
        main_lebel = random.randint(0, len(img_kinds_list) - 1)
        main_data_dir = os.path.join(data_dir, img_kinds_list[main_lebel])
        main_label_num = math.floor(batch_size * IID_rate / 100)
        
        if (batch_size - main_label_num) <= (len(img_kinds_list) - 1) * 1.2:
            other_list = random_fixed(batch_size - main_label_num, len(img_kinds_list) - 1, seed + x)
        else :
            other_list = random_num_with_fix_total(batch_size - main_label_num, len(img_kinds_list) - 1, seed + x)

        if not os.path.exists(main_data_dir):
            os.makedirs(main_data_dir)
            print(f'创建文件夹{main_data_dir}成功！')
        data_copy(os.path.join(os.path.abspath(image_folder_dir), img_kinds_list[main_lebel]), main_data_dir, main_label_num, seed + x)
        num_sub_files = 0
        for y in img_kinds_list:   
            if y != img_kinds_list[main_lebel]:
                if other_list[num_sub_files] == 0:
                    continue
                sub_data_dir = os.path.join(data_dir, y)     
                if not os.path.exists(sub_data_dir):
                    os.makedirs(sub_data_dir)
                    print(f'创建文件夹{sub_data_dir}成功！')
                data_copy(os.path.join(os.path.abspath(image_folder_dir), y), sub_data_dir, other_list[num_sub_files], seed + x)
            else:
                continue
            num_sub_files += 1
        print("batch %d done" %(x + 1))
    print("data non iid create done")
    
    
def data_create_iid(image_folder_dir, dst_folder_dir, num_of_batch, batch_size, seed = 123456789):  
    img_kinds_list = os.listdir(os.path.abspath(image_folder_dir))
    for x in range(num_of_batch):
        data_dir = os.path.join(os.path.abspath(dst_folder_dir), (r'data_iid_' + str(x)))
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        print(f'创建文件夹{data_dir}成功！')
                
        if batch_size <= len(img_kinds_list) * 1.2:
            other_list = random_fixed(batch_size, len(img_kinds_list), seed + x)
        elif batch_size <= len(img_kinds_list) * 3:
            other_list = random_num_with_fix_total(batch_size, len(img_kinds_list), seed + x)
        else:
            val = batch_size // len(img_kinds_list) 
            other_list = [val - 1] * len(img_kinds_list)
            remians = batch_size % len(img_kinds_list) 
            for i in range(remians + len(img_kinds_list)):
                random.seed(seed + i)
                other_list[random.randint(0, len(img_kinds_list) - 1)] += 1               
                
        num_sub_files = 0
        for y in img_kinds_list:              
            if other_list[num_sub_files] == 0:
                continue
            sub_data_dir = os.path.join(data_dir, y)    
            if not os.path.exists(sub_data_dir):
                os.makedirs(sub_data_dir)
                print(f'创建文件夹{sub_data_dir}成功！')
            data_copy(os.path.join(os.path.abspath(image_folder_dir), y), sub_data_dir, other_list[num_sub_files], seed + x)
            num_sub_files += 1
        print("batch %d done" %(x + 1))            
    print("data iid create done")
    
    
    
    
if __name__ == '__main__':
    if(len(sys.argv) <= 5):
        sys.exit(-1)
    data_create_niid(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))        
    data_create_iid(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[6]))

