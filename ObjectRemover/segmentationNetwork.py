import torch
from torch import nn, optim, tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

import numpy as np
import cv2
import matplotlib.pyplot as plt

import fastai
from fastai.vision import *
from fastai.vision import learner
from fastai.vision.all import *

def conv_trans(ni, nf, ks = 4, stride = 2, padding = 1, ps=0.50):
    return nn.Sequential(
        nn.ConvTranspose2d(ni, nf, kernel_size=ks, bias=False, stride=stride, padding = padding),
        nn.ReLU(inplace = True),
        nn.BatchNorm2d(nf),
        nn.Dropout(p=ps))


class MyModel(nn.Module):
    def __init__(self, base_model, ps=0.35):
        super(MyModel, self).__init__()
        self.base_model = learner.create_body(base_model, pretrained=True)
        self.seg_head = nn.Sequential(
            conv_trans(ni=512, nf=256, ks=4, stride=2, padding=1, ps=0.8),
            conv_trans(256, 128),
            conv_trans(128, 64),
            conv_trans(64, 32),
            nn.ConvTranspose2d(32, 1, kernel_size=4, bias=False, stride=2, padding=1)
        )

    def forward(self, x):
        # Attach head to base model
        x = self.base_model(x)

        # m3 = torch.nn.Softmax()
        m3 = nn.Sigmoid()
        seg_model = self.seg_head(m3(x))

        return [seg_model]

    # Create custom loss function

def MyLoss(yhat, seg_tgts):
    # seg_loss = CrossEntropyLossFlat(axis=1)(yhat[0], seg_tgts.long()) #For multi-class problems

    seg_loss = torch.nn.BCELoss()(yhat, seg_tgts.float())  # BCEWithLogitLoss
    return 1.0 * seg_loss


def pixel_accuracy(yhat, seg_tgts):  # segmentation accuracy bbox_tgts,
    # pred_mask[pred_mask>0.5] = 1
    # pred_mask[pred_mask<0.5] = 0
    # print(y.shape, seg_tgts.shape)
    y_ = seg_tgts.squeeze(dim=1)
    yhat_ = yhat.squeeze(dim=1)

    yhat_[yhat_ > 0.5] = 1
    yhat_[yhat_ < 0.5] = 0

    return (y_ == yhat_).sum().float() / seg_tgts.numel()