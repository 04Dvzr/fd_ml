import os
import random
import re
import shutil
import sys

if(len(sys.argv) <= 1):
    exit(-1)

# 1 读取文件列表
image_dir = sys.argv[1] # 一开始存放有图片的文件夹，你需要修改成你的文件夹名字
img_name_list = os.listdir(image_dir)
type_list = []
for x in img_name_list:
    if(x.partition(" ")[0] not in type_list):
        type_list.append(x.partition(" ")[0])

for x in type_list:
    if(x[0] != "g"):
        type_list.remove(x)

# 2 创建一个目标文件夹
result_dir = r'result/' # 目标文件夹（最终存放乱序后的文件夹），会自动创建
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    print(f'创建文件夹{result_dir}成功！')
for x in type_list:
    path_dir = result_dir + x
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        print(f'创建文件夹{path_dir}成功！')

for img in img_name_list:
    if img.partition(" ")[0] in type_list :
        shutil.copyfile(image_dir + img, result_dir + img.partition(" ")[0] + '/' + img)

print("clean done")