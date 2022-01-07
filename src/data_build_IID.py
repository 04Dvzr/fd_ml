import os
import random
import shutil
import sys
import math
from functools import reduce
        
def data_copy(source_dir, data_dir, num, seed = 123456789):
    files = os.listdir(source_dir)
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
            shutil.copyfile(source_dir + files[a[x]], data_dir + str(counter) + ".jpg")
    random.seed(seed + num_loop + 1)
    a = random.sample(range(len(files)), remain_files)
    for x in range(remain_files):
        counter += 1 
        shutil.copyfile(source_dir + files[a[x]], data_dir + str(counter) + ".jpg")
    # print("copy done")
  
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
    
def data_create(image_folder_dir, num_of_batch, batch_size, IID_rate, seed = 123456789):     
    img_kinds_list = os.listdir(image_folder_dir)
    if IID_rate < 50:
        print("make rate higher than 50")
        return -1
    for x in range(num_of_batch):
        data_dir = r'data_' + str(x) + '/'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        print(f'创建文件夹{data_dir}成功！')
        random.seed(seed + x)
        # print(seed + x)
        main_lebel = random.randint(0, len(img_kinds_list) - 1)
        # print(main_lebel)
        # print(img_kinds_list[main_lebel])
        main_data_dir = data_dir + img_kinds_list[main_lebel] + "/"   
        main_label_num = math.floor(batch_size * IID_rate / 100)
        other_list = random_fixed(batch_size - main_label_num, len(img_kinds_list) - 1, seed + x)
        # print("main %d" %main_label_num)
        # print(other_list)       
        if not os.path.exists(main_data_dir):
            os.makedirs(main_data_dir)
            print(f'创建文件夹{main_data_dir}成功！')
        data_copy(image_folder_dir + img_kinds_list[main_lebel] + "/", main_data_dir, main_label_num, seed + x)
        num_sub_files = 0
        for y in img_kinds_list:   
            num_sub_files += 1   
            if y != img_kinds_list[main_lebel]:
                if other_list[num_sub_files] == 0:
                    continue
                sub_data_dir = data_dir + y + "/"       
                if not os.path.exists(sub_data_dir):
                    os.makedirs(sub_data_dir)
                    print(f'创建文件夹{sub_data_dir}成功！')
                data_copy(image_folder_dir + y + "/", sub_data_dir, other_list[num_sub_files], seed + x)
            else:
                num_sub_files -= 1
        print("batch %d done" %x)
    print("data create done")
                
if __name__ == '__main__':
    if(len(sys.argv) <= 5):
        sys.exit(-1)
    data_create(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))        


