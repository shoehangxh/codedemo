# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import seaborn as sns
import torch
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
import cv2 as cv
from PIL import Image
from torch.utils.data import Dataset
from model.mynet import mynet_re
from model.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

transfer = True
frozen = False
mpl.rcParams['font.serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False

class MyDataset(Dataset):
    def __init__(self, datapath, transform=None):
        imgs = []
        label = []
        self.datapath = datapath
        txtpath = self.datapath + '\\' + 'train.txt'
        datainfo = open(txtpath, 'r')
        for line in datainfo:
            line = line.strip('\n')
            if line == 'train.bat' or line == 'train.txt':
                continue
            words = line.split('_')
            imgs.append(line)
            if words[0] == '0':
                label.append(49.3)
            elif words[0] == '1':
                label.append(51.2)
            elif words[0] == '2':
                label.append(172.3)
            elif words[0] == '3':
                label.append(149.4)
            elif words[0] == '4':
                label.append(23.3)
        self.imgs = imgs
        self.label = label
        assert len(self.imgs) == len(self.label)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        pic = self.imgs[index]
        label = self.label[index]
        pic = Image.open(self.datapath + '\\' + pic)
        if self.transform is not None:
            pic = self.transform(pic)
        label = np.array(label).astype(int)
        label = torch.from_numpy(label)
        return pic, label


def adj_r_squared(num, y_test, y_predict):
    SS_R = sum((y_test - y_predict) ** 2)
    SS_T = sum((y_test - np.mean(y_test)) ** 2)
    r_squared = 1 - (float(SS_R)) / SS_T
    adj_r_squared = 1 - (1 - r_squared) * (len(y_test) - 1) / (len(y_test) - num - 1)
    return adj_r_squared


def regression_method(y_test, y_test_pre, y_train, y_train_pre,train_num, validate_num):

    result = y_test_pre
    result_ = y_train_pre
    MSE = mean_squared_error(result, y_test)
    MSE_ = mean_squared_error(result_, y_train)
    MAE = mean_absolute_error(result, y_test)
    MAE_ = mean_absolute_error(result_, y_train)
    R2 = r2_score(y_test, result)
    ARS = adj_r_squared(validate_num, y_test, result)
    ARS_ = adj_r_squared(train_num, y_train, result_)
    PR = pearsonr(y_test, result)
    PR_ = pearsonr(y_train, result_)

    print(f'MSE={MSE}')
    print(f'MSE_train={MSE_}')
    print(f'MAE={MAE}')
    print(f'MAE_train={MAE_}')
    print(f'ARS={ARS}')
    print(f'ARS_train={ARS_}')
    print(f'PR={PR}')
    print(f'PR_train={PR_}')


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = transforms.Compose([transforms.Resize(101),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_path = img_path = r"../data/4DL/my_data"
    batch_size = 1

    train_dataset = MyDataset(datapath=os.path.join(image_path, "train"),
                                            transform=data_transform)
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)


    validate_dataset = MyDataset(datapath=os.path.join(image_path, "test"),
                                            transform=data_transform)
    validate_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)


    net = mynet_re(3)

    # load pretrain weights
    model_weight_path = r"../data/4DL/resnet18-bs16.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)
    net.eval()
    y_test_pre = []
    y_test = []
    y_train_pre = []
    y_train = []
    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            y_test.append(val_labels.item())
            outputs = net(val_images.to(device))
            y_test_pre.append(outputs.cpu().item())
        for train_data in tqdm(train_loader):
            train_images, train_labels = train_data
            y_train.append(train_labels.item())
            outputs = net(train_images.to(device))
            y_train_pre.append(outputs.cpu().item())
    print('1', y_test_pre)
    print('2', y_test)
    print('3', y_train_pre)
    print('4', y_train)
    #quit()
    regression_method(np.array(y_test), np.array(y_test_pre), np.array(y_train), np.array(y_train_pre), train_num, validate_num)

    y_test = np.array(y_test)

    _a = []
    _b = []
    _c = []
    _d = []
    _e = []

    for i in range(len(y_test)):
        if y_test[i] == 23:
            _a.append(y_test_pre[i])
        elif y_test[i] == 49:
            _b.append(y_test_pre[i])
        elif y_test[i] == 51:
            _c.append(y_test_pre[i])
        elif y_test[i] == 149:
            _d.append(y_test_pre[i])
        elif y_test[i] == 172:
            _e.append(y_test_pre[i])
    my_data = [_a[:8], _b[:8], _c[:8], _d[:8], _e[:8]]
    _a = np.array(_a)
    _b = np.array(_b)
    _c = np.array(_c)
    _d = np.array(_d)
    _e = np.array(_e)
    all__data = [_a, _b, _c, _d, _e]
    print(my_data)

    sns.set(color_codes=True)
    # sns.set_style("dark")
    fig, axes = plt.subplots(figsize=(6, 5), dpi=100)
    parts = axes.violinplot(
        my_data, showmeans=False, showmedians=False,
        showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    quartile1, medians, quartile3 = np.percentile(my_data, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(my_data, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
    inds = np.arange(1, len(medians) + 1)
    axes.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    axes.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    axes.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

    axes.set_title('CNN', fontsize=20, weight='bold')

    axes.yaxis.grid(True)
    axes.set_xticks([y + 1 for y in range(len(my_data))], )
    axes.set_xlabel('type of polymer', fontsize=18
                    , weight='bold'
                    )
    axes.set_ylabel('width of Tg(℃)', fontsize=17
                    , weight='bold'
                    )

    plt.setp(axes, xticks=[y + 1 for y in range(len(my_data))],
             xticklabels=['RAN', 'DI', 'TRI', 'LG', 'VG'],
             # lablesize=18
             # ,weight='bold'
             )

    plt.savefig("cnn.png", bbox_inches='tight', pad_inches=0.0)
    plt.show()