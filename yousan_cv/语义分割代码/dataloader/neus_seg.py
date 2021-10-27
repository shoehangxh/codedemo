import os
from PIL import Image
from os.path import join as pjoin
from dataloader.base_datasets import Dataset
from utils.distributed import *

# 读取
class neu_seg(Dataset):
    # 类别
    NUM_CLASS = 21
    # 参数预定义
    def __init__(self,
                 root='/home/aries/Downloads/NEU_defect/',split='train',
                 mode=None,transform=None,**kwargs):
        super(neu_seg,self).__init__(root,split,mode,transform,**kwargs)
        # 图片文件夹
        image_dir = pjoin(root,'images')
        # 标签文件夹
        mask_dir = pjoin(root,'annotations')
        # 根据阶段选定所用的txt文件
        # train
        if split == 'train':
            split_txt = pjoin(root,'train.txt')
        # val
        elif split == 'val':
            split_txt = pjoin(root,'val.txt')
        # 无效类型
        else:
            raise RuntimeError('Unknown dataset split:{}'.format(split))
        # 构建图片列表
        self.images = []
        # 构建标签列表
        self.masks = []
        # 根据txt文件获取图片及标签路径
        with open(pjoin(split_txt), "r") as lines:
            for line in lines:
                # 图片路径
                image = pjoin(image_dir, split, line.rstrip('\n') + ".jpg") # 去掉换行符号
                # 判定是否为文件
                assert os.path.isfile(image)# 断言函数
                # 写入图片列表
                self.images.append(image)
                # 标签路径
                mask = pjoin(mask_dir, split,line.rstrip('\n') + ".png")
                # 判定是否为文件
                assert os.path.isfile(mask)
                # 写入标签列表
                self.masks.append(mask)
        # 判定图片及标签数量是否一致
        assert (len(self.images) == len(self.masks))
        # 输出获取结果
        print('Found {} images in the folder {}'.format(len(self.images), root))

    # 获取样本总量
    def __len__(self):
        return len(self.images)

    # 读取图片
    def __getitem__(self,item):
        img = Image.open(self.images[item]).convert('RGB') #循环，读取都转换成rgb格式
        mask = Image.open(self.masks[item])
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            raise RuntimeError('unknown mode for dataloader: {}'.format(self.mode))
        # 归一化、去均值等
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[item])#推按路径的名字


if __name__ == '__main__':
    from torchvision import transforms
    import torch.utils.data as data
    import torch
    import cv2 as cv


    # Create Dataset
    trainset =neu_seg(split='train')
    # Create Training Loader
    train_dataset = data.DataLoader(trainset, 1, shuffle=True,num_workers=4)
    print(train_dataset)
    for iteration, (img,lbl,dir) in enumerate(train_dataset):
        print(img.shape)
        print(lbl.shape)
        img = img.to('cuda')
        img = torch.squeeze(img,0)
        img = img.cpu().numpy()
        lbl = lbl.to('cuda')
        lbl = torch.squeeze(lbl,0)
        lbl = lbl.cpu().numpy()
        cv.imwrite('img.jpg',img)
        cv.imwrite('lbl.png',lbl)
        break



