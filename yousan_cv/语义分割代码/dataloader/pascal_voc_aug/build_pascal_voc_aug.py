import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import scipy.misc as misc
from scipy import io as scipy_io
from skimage import io as skimage_io

def decode_segmap(label_mask, plot=False):
    """
    功能：
        将灰度标签转化为RGB标签
    参数:
        label_mask : (M,N)维度的灰度图
        plot : 是否绘制图例.

    结果:
        rgb: 解码后的（M，N，3）维度的RGB图.
    """
    label_colours = get_pascal_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, 21):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def get_pascal_labels():
    """
    Pascal VOC各类别对应的色彩标签

    结果:
        (21, 3)矩阵，含各类别对应的三通道数值信息
    """
    return np.asarray(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],

        ]
    )

def encode_segmap(mask):
    """
    功能：
        将RBG标签转换为对应的灰度标签
    参数:
        mask:（M，N，3）的RGB标签.
    返回值:
        label_mask:（M，N）的灰度标签.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii #对应的时候，将色彩转换为ii
    label_mask = label_mask.astype(int)
    return label_mask

# SBD路径(含有.mat文件)
sbd_path = os.path.join(r"E:/dataset/Pascal VOC/benchmark_RELEASE","dataset","cls")
# VOC原始分割标签路径
pascal_path = os.path.join(r"E:/dataset/Pascal VOC/VOCdevkit/VOC2012","SegmentationClass")
# 输出路径
aug_path = os.path.join(r"E:/dataset/Pascal VOC/VOCdevkit/VOC2012","SegmentationClassAug")
# RGB图路径
rgb_path = r"E:/dataset/Pascal VOC/VOCdevkit/VOC2012/SegmentationClassAugRGB/"
# 创建输出路径
if os.path.exists(aug_path) is not True:
    os.mkdir(aug_path)
if os.path.exists(rgb_path) is not True:
    os.mkdir(rgb_path)

# 原始标签转灰度图并移动至输出路径
for ii in os.listdir(pascal_path):
    lbl_path = os.path.join(pascal_path,ii)
    lbl = encode_segmap(misc.imread(lbl_path))
    lbl = misc.toimage(lbl, high=lbl.max(), low=lbl.min())
    misc.imsave(os.path.join(aug_path, ii), lbl)
print('Done!')

# 读取SBD路径下的mat文件,并将其保存至输出路径
sbd_filenames = glob.glob(os.path.join(sbd_path,"*.mat")) #后缀限制
for filename_index,single_sbd_filename in enumerate(sbd_filenames):
    # note：print(f"")与print("%d".%(something))的区别
    if filename_index % 500 == 0: print(f"processing sbd dataset - total:{len(sbd_filenames)},now:{filename_index},{filename_index/len(sbd_filenames)*100:.3}%")
    single_id = os.path.split(single_sbd_filename)[1][:-4]
    sbd_data = scipy_io.loadmat(single_sbd_filename)
    skimage_io.imsave(os.path.join(aug_path,f"{single_id}.png"),sbd_data["GTcls"]["Segmentation"][0][0])

# 遍历灰度图并解码为RGB图
for img in os.listdir(aug_path):
    img_path = os.path.join(aug_path,img)
    img1 = misc.imread(img_path)
    decoded = decode_segmap(img1)
    misc.imsave(os.path.join(rgb_path,img), decoded)
print("Done!")