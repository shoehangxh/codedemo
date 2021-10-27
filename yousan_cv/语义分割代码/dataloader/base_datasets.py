import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter


# 基本数据
class Dataset(object): #自己读取数据，或者torch.utils.data import Dataset
    # 参数预定义
    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        # 可继承类
        super(Dataset, self).__init__()
        # 路径
        self.root = root
        # 增强
        self.transform = transform
        # 阶段
        self.split = split
        # 模式
        self.mode = mode if mode is not None else split
        # 基础图像尺寸
        self.base_size = base_size
        # 裁剪尺寸
        self.crop_size = crop_size

    # 验证集转换
    def _val_sync_transform(self, img, mask):
        # 裁剪尺寸
        outsize = self.crop_size
        # 判断最短边
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        # 使图像长宽一致
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # 中心裁剪
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # 最终的转换
        img = np.array(img)
        mask = np.array(mask).astype('int32')
        return img, mask

    # 训练时的转换
    def _sync_transform(self, img, mask):
        # 随机镜像反转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        # 获取裁剪后的尺寸
        crop_size = self.crop_size
        # 在基本尺寸上进行随机缩放
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        # 获取图像原始尺寸
        w, h = img.size
        # 获取较短尺寸
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        # 使图像长宽一致
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # 若最小尺寸小于裁剪尺寸，则进行0填充
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # 随机裁剪
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # 加入Gaussian扰动
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # 最终的转换
        img = np.array(img)
        mask = np.array(mask).astype('int32')
        return img, mask

    @property
    # 数据类别
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0
