#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import random
import torchvision.transforms.functional as TF


def my_segmentation_trasforms(image, segmentation):
    if random.random() > 0.5:
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle)
        segmentation = TF.rotate(segmentation, angle)
    return image, segmentation

class SegDataset(Dataset): #读取图像路径+掩膜路径（已经按行存储）
    def __init__(self, traintxt, imagesize, cropsize, transform=None, extra=False):
        self.images = []
        self.labels = []
        self.extra = extra

        lines = open(traintxt,'r').readlines() #读取txt文件用open格式（train_small.txt）
        for line in lines:
            imagepath,labelpath = line.strip().split(' ')
            self.images.append(imagepath)
            self.labels.append(labelpath)

        self.imagesize = imagesize
        self.cropsize = cropsize

        assert len(self.images) == len(self.labels)  #断言函数，条件为false时才会往下执行，否则报错
        self.transform  = transform
        self.samples = []
        for i in range(len(self.images)):
            self.samples.append((self.images[i],self.labels[i]))


    def __getitem__(self, item):
        img_path, label_path = self.samples[item]
        img = cv2.imread(img_path)
        # resize成指定大小和缩放手段
        img = cv2.resize(img, (self.imagesize,self.imagesize),interpolation=cv2.INTER_NEAREST)
        label = cv2.imread(label_path, 0)
        label = cv2.resize(label, (self.imagesize, self.imagesize), interpolation=cv2.INTER_NEAREST)
        # mask = (img/127).astype(np.uint8) 分割标签转换为【0，1，2】
        ## 随机裁剪数据增强，可添加更多操作（裁剪操作）
        #不进行randomresize的操作：保证标签和图像的剪裁是一致的
        randoffsetx = np.random.randint(self.imagesize - self.cropsize)
        randoffsety = np.random.randint(self.imagesize - self.cropsize)
        img = img[randoffsety:randoffsety + self.cropsize, randoffsetx:randoffsetx + self.cropsize]
        label = label[randoffsety:randoffsety + self.cropsize, randoffsetx:randoffsetx + self.cropsize]
        if self.extra is True:
            img, label = my_segmentation_trasforms(img, label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

