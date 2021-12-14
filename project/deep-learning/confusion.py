# -*- coding:utf-8 -*-
import os
import json
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from PIL import Image
from model.mynet import mynet
from torch.utils.data import Dataset
from model.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


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
            label.append(words[0])
        self.imgs = imgs
        self.label = label
        print(len(self.imgs))
        print(len(self.label))
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


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = transforms.Compose([transforms.Resize(101),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_path = r"../data/4DL/my_data"
    test_dataset = MyDataset(datapath=os.path.join(image_path, "test"), transform=data_transform)

    batch_size = 1
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)
    net = mynet(3)
    # load pretrain weights
    model_weight_path = r"../data/4DL/mynet-bs16.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    # read class_indict
    json_label_path = 'class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=5, labels=labels)
    net.eval()
    with torch.no_grad():
        for test_data in tqdm(test_loader):
            test_images, test_labels = test_data
            outputs = net(test_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), test_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()

