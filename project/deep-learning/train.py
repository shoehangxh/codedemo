# -*- coding:utf-8 -*-
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
import cv2 as cv
from PIL import Image
from model.mynet import mynet
from model.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


transfer = False
frozen = False

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(101),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(101),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    img_path = r"../data/4Vi/data2"
    train_dataset = datasets.ImageFolder(root=os.path.join(img_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    if (not transfer):
        validate_dataset = datasets.ImageFolder(root=os.path.join(img_path, "val"),
                                                transform=data_transform["val"])
        val_num = len(validate_dataset)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

        print("using {} images for training, {} images for validation.".format(train_num,
                                                                            val_num))
    else:
        print("using {} images for training".format(train_num))

    #net = mynet(3)
    net = resnet18()
    #net = resnet34()
    #net = resnet50()
    #net = resnet101()
    #net = resnet152()

    if transfer:
        pretrain_model = torch.load(r"../data/4DL/origin.pth")
        model2_dict = net.state_dict()
        state_dict = {k: v for k, v in pretrain_model.items() if k in model2_dict.keys()}
        model2_dict.update(state_dict)
        net.load_state_dict(model2_dict)
        if frozen:
            for parm in net.net.parameters():
                parm.requires_grad = False
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001)
    epochs = 200
    best_acc = 0.0
    save_path = r'../data/4DL/resnet18-bs16.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device)).float()
            loss = loss_function(logits, labels.long().to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # validate
        if (not transfer):
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(validate_loader)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                (epoch + 1, running_loss / train_steps, val_accurate))
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    main()

