#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020-11-18 21:46
# @Author  : NingAnMe <ninganme@qq.com>
from __future__ import print_function, division
import os
import numpy as np
from PIL import Image
import glob
# import SimpleITK as sitk  # 处理CT数据和X光数据
from torch import optim
import torch.utils.data
import torch
import torch.nn.functional as F

import torch.nn
import torchvision
import matplotlib.pyplot as plt
import natsort  # 自然数排序
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import torchsummary
from torch.utils.tensorboard import SummaryWriter

import shutil
import random
from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net
import time

from ranger.ranger2020 import Ranger

#######################################################
# Checking if GPU is used
#######################################################
device_auto = True

if device_auto:
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available. Training on CPU')
    else:
        print('CUDA is available. Training on GPU')

    device_name = "cuda:0" if train_on_gpu else "cpu"
else:
    # device_name = "cuda:0"
    device_name = "cpu"

device = torch.device(device_name)

#######################################################
# Setting the basic paramters of the model
#######################################################
continute_train = True
print('continute_train = {}'.format(continute_train))

model_name = 'U_Net'
print('model_name = {}'.format(model_name))

sample_num = 512  # 加载的样本数量，全部设置为-1
print('sample_num = {}'.format(sample_num))

valid_size = 0.25
print('valid_size = {}'.format(valid_size))

batch_size = 8
print('batch_size = {}'.format(batch_size))

epoch_max = 20
print('epoch_max = {}'.format(epoch_max))

random_seed = 43
print('random_seed = {}'.format(random_seed))

i_h = 256
i_w = 256
class_num = 7
print('data size = {}, {} , {}'.format(class_num, i_h, i_w))

shuffle = True
valid_loss_min = np.Inf
# 如果RuntimeError: DataLoader worker (pid 30141) exited unexpectedly with exit code 1.
# https://github.com/pytorch/pytorch/issues/5301#issuecomment-453249640
num_workers = 0  # 如果多线程报错，改为0
lossT = list()
lossL = list()
lossL.append(np.inf)
lossT.append(np.inf)
epoch_valid = epoch_max - 2
i_valid = 0

# 是否使用锁页内存
if 'cuda' in device_name:
    pin_memory = True
else:
    pin_memory = False


#######################################################
# Creating a Folder for every data of this train
#######################################################
# drive_proj_path = '.'

result_path = os.path.join(drive_proj_path, 'result')

train_result_folder = os.path.join(result_path, '{}_epoch_{}_batch_{}'.format(model_name, epoch_max, batch_size))

#######################################################
# Setting the folder of saving the predictions
#######################################################

read_pred_folder = os.path.join(train_result_folder, 'pred')

#######################################################
# checking if the model exists and if true then delete
#######################################################

read_model_folder = os.path.join(train_result_folder, 'model')

#######################################################
# create folders
#######################################################

if not continute_train:
    for folder in [train_result_folder, read_pred_folder, read_model_folder]:
        if os.path.exists(folder) and os.path.isdir(folder):
            shutil.rmtree(folder)

        try:
            os.makedirs(folder)
        except OSError:
            print("Creation of the main directory '%s' failed " % folder)
        else:
            print("Successfully created the main directory '%s' " % folder)

#######################################################
# Passing the Dataset of Images and Labels
#######################################################

t_data = img_train_dir
l_data = lab_train_dir
test_image = os.path.join(img_train_dir, 'T000005.jpg')
test_label = os.path.join(lab_train_dir, 'T000005.png')
test_folderP = img_train_dir
test_folderL = lab_train_dir
print(test_image)
print(test_label)


class Images_Dataset_folder(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images_dir, labels_dir, transformI=None, transformM=None, sample_num=-1):
        print("总数：", len(os.listdir(images_dir)))
        self.images = sorted(os.listdir(images_dir))[:sample_num]
        self.labels = sorted(os.listdir(labels_dir))[:sample_num]
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transformI = transformI
        self.transformM = transformM

        if self.transformI:
            self.tx = self.transformI
        else:
            self.tx = torchvision.transforms.Compose([
                #  torchvision.transforms.Resize((128,128)),
                # torchvision.transforms.CenterCrop(96),
                # torchvision.transforms.RandomRotation((-10, 10)),
                # torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        if self.transformM:
            self.lx = self.transformM
        else:
            # self.lx = torchvision.transforms.Compose([
            #     #  torchvision.transforms.Resize((128,128)),
            #     # torchvision.transforms.CenterCrop(96),
            #     # torchvision.transforms.RandomRotation((-10, 10)),
            #     # torchvision.transforms.Grayscale(),
            #     torchvision.transforms.ToTensor(),
            #     # torchvision.transforms.Lambda(lambda x: torch.cat([x, 1 - x], dim=0))
            # ])
            self.lx = torchvision.transforms.Compose([
                np.array,
                torch.from_numpy,
            ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, i):
        i1 = Image.open(self.images_dir + self.images[i])
        l1 = Image.open(self.labels_dir + self.labels[i])

        seed = np.random.randint(0, 100)  # make a seed with numpy generator

        # apply this seed to img tranfsorms
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.tx(i1)

        # apply this seed to target/label tranfsorms
        random.seed(seed)
        torch.manual_seed(seed)
        label = self.lx(l1)
        label = label.long()
        label[label == 255] = 6

        return img, label


Training_Data = Images_Dataset_folder(t_data, l_data, sample_num=sample_num)

#######################################################
# Giving a transformation for input data
#######################################################

data_transform_p = torchvision.transforms.Compose([
    #  torchvision.transforms.Resize((i_h,i_w)),
    #   torchvision.transforms.CenterCrop(96),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_transform_l = torchvision.transforms.Compose([
    #  torchvision.transforms.Resize((i_h,i_w)),
    #   torchvision.transforms.CenterCrop(96),
    np.array,
    torch.from_numpy,
])

#######################################################
# Trainging Validation Split
#######################################################

num_train = len(Training_Data)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory, )

valid_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory, )


#######################################################
# loss function
#######################################################

def calc_loss(prediction, target):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """

    criterion = torch.nn.CrossEntropyLoss()
    ce = criterion(prediction, target)
    loss = ce
    return loss


#######################################################
# Writing the params to tensorboard
#######################################################

writer1 = SummaryWriter(read_model_folder)

#######################################################
# Setting up the model
#######################################################

model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]


def model_unet(model_input, in_channel=3, out_channel=1):
    model = model_input(in_channel, out_channel)
    return model


model_test = model_unet(model_Inputs[0], 3, class_num)

model_test.to(device)

#######################################################
# Using Adam as Optimizer
#######################################################

initial_lr = 0.001
opt = Ranger(model_test.parameters())  # try Ranger
# opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr)  # try Adam
# opt = optim.SGD(model_test.parameters(), lr = initial_lr, momentum=0.99)  # try SGD

#######################################################
# continue train
#######################################################
if continute_train:
    save_path = os.path.join(read_model_folder, 'save.pth')
    print('load model save ======== {}'.format(save_path))
    model_save = torch.load(save_path)
    model_test.load_state_dict(model_save['model'])

    opt.load_state_dict(model_save['optimizer'])
    MAX_STEP = int(1e10)
    # 继续训练的时候，需要基于恢复的optimizer重定义lr_scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-5)
    lr_scheduler.load_state_dict(model_save['lr_scheduler'])

    epoch_start = model_save['epoch_start']
    valid_loss_min = model_save['valid_loss_min']
    print('epoch_start = {}'.format(epoch_start))
    print('valid_loss_min = {}'.format(valid_loss_min))
else:
    epoch_start = 0
    MAX_STEP = int(1e10)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-5)  # 学习率调整


#######################################################
# Getting the Summary of Model
#######################################################
torchsummary.summary(model_test, input_size=(3, i_h, i_w), device=device_name[:4])


#######################################################
# Training loop
#######################################################
for i in range(epoch_start, epoch_max):
    print('epoch == {}'.format(i))
    train_loss = 0.0
    valid_loss = 0.0
    since = time.time()
    lr_scheduler.step(i)
    lr = lr_scheduler.get_lr()

    #######################################################
    # Training Data
    #######################################################

    model_test.train()
    k = 1

    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)

        # If want to get the input images with their Augmentation - To check the data flowing in net
        # input_images(x, y, i, n_iter, k)

        # grid_img = torchvision.utils.make_grid(x)
        # writer1.add_image('images_img', grid_img, 0)

        # grid_lab = torchvision.utils.make_grid(y)
        # writer1.add_image('images_lab', grid_lab, 0)

        opt.zero_grad()

        y_pred = model_test(x)
        lossT = calc_loss(y_pred, y)  # Dice_loss Used

        train_loss += lossT.item() * x.size(0)
        lossT.backward()
        #  plot_grad_flow(model_test.named_parameters(), n_iter)
        opt.step()
        x_size = lossT.item() * x.size(0)
        k = 2
        for name, param in model_test.named_parameters():
            name = name.replace('.', '/')
            writer1.add_histogram(name, param.data.cpu().numpy(), i + 1)
            writer1.add_histogram(name + '/grad', param.grad.data.cpu().numpy(), i + 1)

    #######################################################
    # Validation Step
    #######################################################

    model_test.eval()  # 对Dropout和BN等生效
    torch.no_grad()  # to increase the validation process uses less memory

    for x1, y1 in valid_loader:
        x1, y1 = x1.to(device), y1.to(device)

        y_pred1 = model_test(x1)
        lossL = calc_loss(y_pred1, y1)  # Dice_loss Used

        valid_loss += lossL.item() * x1.size(0)
        x_size1 = lossL.item() * x1.size(0)
    #######################################################
    # Saving the predictions
    #######################################################

    im_tb = Image.open(test_image)
    im_label = Image.open(test_label)
    s_tb = data_transform_p(im_tb)
    s_label = data_transform_l(im_label)

    pred_tb = model_test(s_tb.unsqueeze(0).to(device))
    pred_tb = F.softmax(pred_tb, dim=1)  # 多分类用
    pred_tb = torch.argmax(pred_tb, dim=1)  # 多分类用
    # pred_tb = F.sigmoid(pred_tb)  # 二分类用
    # pred_tb = pred_tb.detach().numpy()
    # pred_tb = threshold_predictions_v(pred_tb)
    pred_img = pred_tb
    pred_img = pred_img.cpu().detach().numpy()
    print(np.min(pred_img), np.max(pred_img), np.mean(pred_img), pred_img.shape)
    img_path = os.path.join(read_pred_folder, 'img_epoch_{}.png'.format(i))
    img = Image.fromarray(pred_img[0].astype(np.uint8))

    # 增加调色板
    img = img.convert("P")
    palette = Image.open(test_label).getpalette()
    img.putpalette(palette)

    img.save(img_path)

    #######################################################
    # To write in Tensorboard
    #######################################################

    train_loss = train_loss / len(train_idx)
    valid_loss = valid_loss / len(valid_idx)

    # if (i + 1) % 1 == 0:
    print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(i, epoch_max, train_loss, valid_loss))
    writer1.add_scalar('Train Loss', train_loss)
    writer1.add_scalar('Validation Loss', valid_loss)
    # writer1.add_image('Pred', pred_tb[0])  # try to get output of shape 3

    #######################################################
    # Early Stopping
    #######################################################

    if valid_loss <= valid_loss_min:  # and i_valid <= 2:
        model_out = os.path.join(read_model_folder, 'save.pth')
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model {}'.format(valid_loss_min, valid_loss, model_out))
        save_dict = {
            'model': model_test.state_dict(),
            'optimizer': opt.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch_start': i,
            'valid_loss_min': valid_loss_min,
        }
        torch.save(save_dict, model_out)
        # print(accuracy)
        if round(valid_loss, 4) == round(valid_loss_min, 4):
            print(i_valid)
            i_valid = i_valid + 1
        valid_loss_min = valid_loss
        # if i_valid ==3:
        #   break