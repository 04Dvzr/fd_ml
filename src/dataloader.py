import torchvision
import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image

class IMG_Dataset(Dataset):
    """Some Information about MyDataset"""
    def __init__(self, src_path, dst_path, transform=None):
        super(IMG_Dataset, self).__init__()
        self.path_src = os.path.abspath(src_path)
        self.path_dst = os.path.abspath(dst_path)
        
        assert(os.path.exists(self.path_src))
        assert(os.path.exists(self.path_dst))
        if transform == None:
            self.trans = transforms.Compose([transforms.ToTensor()])
        else:
            self.trans = transform
        
        
    def __getitem__(self, index):
        src = os.listdir(self.path_src)
        dst = os.listdir(self.path_dst)
        
        image_src = self.trans(Image.open(os.path.join(self.path_src, src[index])))
        image_dst = self.trans(Image.open(os.path.join(self.path_dst, dst[index])))
        return image_src, image_dst

    def __len__(self):
        return len(os.listdir(self.path_src))    
    
    
def Label_Dataset(path, trans=None, label_trans=None, loader=None):
    if(trans==None):
        tran = transforms.Compose([transforms.ToTensor()])
    else:
        tran = trans
    return torchvision.datasets.ImageFolder(path, tran, label_trans, loader)
    
    
    
    
if __name__ == '__main__':
    import sys
    if(len(sys.argv) <= 2):
        sys.exit(-1)
    data = IMG_Dataset(sys.argv[1], sys.argv[2], transform=transforms.ToTensor())
    print(data[1])