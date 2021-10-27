import os
from PIL import Image
from os.path import join as pjoin
from dataloader.base_datasets import Dataset
from torch.utils import data
from utils.distributed import *

# 读取
class lips(Dataset):
    # 类别
    NUM_CLASS = 3
    # 参数预定义
    def __init__(self,
                 root='E:\dataset\data',split='train',
                 mode=None,transform=None,**kwargs):
        super(lips,self).__init__(root,split,mode,transform,**kwargs)
        txt = pjoin(root,
                    #'txt',
                    split + '.txt')
        fh = open(txt, 'r')
        self.imgs = []
        self.lbls = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            line = line.split()
            self.imgs.append(line[0])
            self.lbls.append(line[1])
        # 判定图片及标签数量是否一致
        assert (len(self.imgs) == len(self.lbls))
        # 输出获取结果
        print('Found {} images in the folder {}'.format(len(self.imgs), root))

    # 获取样本总量
    def __len__(self):
        return len(self.imgs)

    # 读取图片
    def __getitem__(self,item):
        img = Image.open(pjoin(self.root,self.imgs[item])).convert('RGB')
        mask = Image.open(pjoin(self.root,self.lbls[item]))
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            raise RuntimeError('unknown mode for dataloader: {}'.format(self.mode))
        # 归一化、去均值等
        if self.transform is not None:
            img = self.transform(img)
        return img, mask


# if __name__ == '__main__':
#     dataset = lips(split='val')
#     train_dataset = data.DataLoader(dataset, 1, shuffle=True, num_workers=4)
#     for iteration, (img,lbl) in enumerate(train_dataset):
#         print(img.shape)
#         print(lbl.shape)
#         if iteration == 10:
#             break

