import os
from PIL import Image
from os.path import join as pjoin
from dataloader.base_datasets import Dataset
from utils.distributed import *

# 读取
class voc_Loader(Dataset):
    # 类别
    NUM_CLASS = 21
    # 参数预定义
    def __init__(self,
                 root='/home/aries/Downloads/segmentation_datasets/',split='train',
                 mode=None,transform=None,**kwargs):
        super(voc_Loader,self).__init__(root,split,mode,transform,**kwargs)
        # VOC文件夹
        voc_root = pjoin(root,'VOC2012')
        # 图片文件夹
        image_dir = pjoin(voc_root,'JPEGImages')
        # 标签文件夹
        mask_dir = pjoin(voc_root,'SegmentationClassAug')
        # 根据阶段选定所用的txt文件
        # train
        if split == 'train':
            split_txt = pjoin(voc_root,'train.txt')
        # val
        elif split == 'val':
            split_txt = pjoin(voc_root,'val.txt')
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
                image = pjoin(image_dir, line.rstrip('\n') + ".jpg")
                # 判定是否为文件
                assert os.path.isfile(image)
                # 写入图片列表
                self.images.append(image)
                # 标签路径
                mask = pjoin(mask_dir, line.rstrip('\n') + ".png")
                # 判定是否为文件
                assert os.path.isfile(mask)
                # 写入标签列表
                self.masks.append(mask)
        # 判定图片及标签数量是否一致
        assert (len(self.images) == len(self.masks))
        # 输出获取结果
        print('Found {} images in the folder {}'.format(len(self.images), voc_root))

    # 获取样本总量
    def __len__(self):
        return len(self.images)

    # 读取图片
    def __getitem__(self,item):
        img = Image.open(self.images[item]).convert('RGB')
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
        return img, mask, os.path.basename(self.images[item])

    @property
    # 类别名称
    def classes(self):
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'dining-table', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv')

# if __name__ == '__main__':
#     from torchvision import transforms
#     import torch.utils.data as data
#     import torch
#     import cv2 as cv
#
#
#     # Create Dataset
#     trainset =voc_Loader(split='train')
#     # Create Training Loader
#     train_dataset = data.DataLoader(trainset, 1, shuffle=True,num_workers=4)
#     print(train_dataset)
#     for iteration, (img,lbl,dir) in enumerate(train_dataset):
#         print(img.shape)
#         print(lbl.shape)
#         img = img.to('cuda')
#         img = torch.squeeze(img,0)
#         img = img.cpu().numpy()
#         lbl = lbl.to('cuda')
#         lbl = torch.squeeze(lbl,0)
#         lbl = lbl.cpu().numpy()
#         cv.imwrite('img.jpg',img)
#         cv.imwrite('lbl.png',lbl)
#         break



