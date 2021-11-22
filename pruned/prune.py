#-*- coding:utf-8 _*-
"""
@author:fxw
@file: prune.py
@time: 2020/07/17
"""
"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: prune.py
@time: 2020/6/27 10:23

"""
import sys

sys.path.append('/home/aistudio/external-libraries')
sys.path.append('./')
import yaml
from models.DBNet import DBNet
import torch
import torch.nn as  nn
import numpy as np
import collections
import torchvision.transforms as transforms
import cv2
import os
import argparse
import math
from PIL import Image
from torch.autograd import Variable


def prune(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config['pruned']['gpu_id']

    model = DBNet(config).cuda()
    model_dict = torch.load(config['pruned']['checkpoints'])['state_dict']
    state = model.state_dict()
    for key in state.keys():
        if key in model_dict.keys():
            state[key] = model_dict[key]
    model.load_state_dict(state)


    bn_weights = []
    for m in model.modules():
        if (isinstance(m, nn.BatchNorm2d)):
            bn_weights.append(m.weight.data.abs().clone())
    bn_weights = torch.cat(bn_weights, 0)

    sort_result, sort_index = torch.sort(bn_weights)

    thresh_index = int(config['pruned']['cut_percent'] * bn_weights.shape[0])

    if (thresh_index == bn_weights.shape[0]):
        thresh_index = bn_weights.shape[0] - 1

    prued = 0
    prued_mask = []
    bn_index = []
    conv_index = []
    remain_channel_nums = []
    for k, m in enumerate(model.modules()):
        if (k > 69):
            break
        if (isinstance(m, nn.BatchNorm2d)):
            bn_weight = m.weight.data.clone()
            mask = bn_weight.abs().gt(sort_result[thresh_index])
            remain_channel = mask.sum()

            if (remain_channel == 0):
                remain_channel = 1
                mask[int(torch.argmax(bn_weight))] = 1

            v = 0
            n = 1
            if (remain_channel % config['pruned']['base_num'] != 0):
                if (remain_channel > config['pruned']['base_num']):
                    while (v < remain_channel):
                        n += 1
                        v = config['pruned']['base_num'] * n
                    if (remain_channel - (v - config['pruned']['base_num']) < v - remain_channel):
                        remain_channel = v - config['pruned']['base_num']
                    else:
                        remain_channel = v
                    if (remain_channel > bn_weight.size()[0]):
                        remain_channel = bn_weight.size()[0]
                    remain_channel = torch.tensor(remain_channel)
                    result, index = torch.sort(bn_weight)
                    mask = bn_weight.abs().ge(result[-remain_channel])

            remain_channel_nums.append(int(mask.sum()))
            prued_mask.append(mask)
            bn_index.append(k)
            prued += mask.shape[0] - mask.sum()
        elif (isinstance(m, nn.Conv2d)):
            conv_index.append(k)
    print('remain_channel_nums', remain_channel_nums)
    print('total_prune_ratio:', float(prued) / bn_weights.shape[0])
    print('bn_index', bn_index)

    new_model = DBNet(config).cuda()

    merge1_index = [3, 12, 18]
    merge2_index = [25, 28, 34]
    merge3_index = [41, 44, 50]
    merge4_index = [57, 60, 66]

    index_0 = []
    for item in merge1_index:
        index_0.append(bn_index.index(item))
    mask1 = prued_mask[index_0[0]] | prued_mask[index_0[1]] | prued_mask[index_0[2]]

    index_1 = []
    for item in merge2_index:
        index_1.append(bn_index.index(item))
    mask2 = prued_mask[index_1[0]] | prued_mask[index_1[1]] | prued_mask[index_1[2]]

    index_2 = []
    for item in merge3_index:
        index_2.append(bn_index.index(item))
    mask3 = prued_mask[index_2[0]] | prued_mask[index_2[1]] | prued_mask[index_2[2]]

    index_3 = []
    for item in merge4_index:
        index_3.append(bn_index.index(item))
    mask4 = prued_mask[index_3[0]] | prued_mask[index_3[1]] | prued_mask[index_3[2]]

    for index in index_0:
        prued_mask[index] = mask1

    for index in index_1:
        prued_mask[index] = mask2

    for index in index_2:
        prued_mask[index] = mask3

    for index in index_3:
        prued_mask[index] = mask4

    print(model)

    ##############################################################
    index_bn = 0
    index_conv = 0

    bn_mask = []
    conv_in_mask = []
    conv_out_mask = []
    tag = 0
    for m in new_model.modules():
        if (tag > 69):
            break
        if (isinstance(m, nn.BatchNorm2d)):
            m.num_features = prued_mask[index_bn].sum()
            bn_mask.append(prued_mask[index_bn])
            index_bn += 1
        elif (isinstance(m, nn.Conv2d)):
            if (index_conv == 0):
                m.in_channels = 3
                conv_in_mask.append(torch.ones(3))
            else:
                m.in_channels = prued_mask[index_conv - 1].sum()
                conv_in_mask.append(prued_mask[index_conv - 1])
            m.out_channels = prued_mask[index_conv].sum()
            conv_out_mask.append(prued_mask[index_conv])
            index_conv += 1
        tag += 1

    conv_change_index = [27, 43, 59]  #
    change_conv_bn_index = [18, 34, 50]  #
    tag = 0
    for m in new_model.modules():
        if (tag > 69):
            break
        if (isinstance(m, nn.Conv2d)):
            if (tag in conv_change_index):
                index = conv_change_index.index(tag)
                index = change_conv_bn_index[index]
                index = bn_index.index(index)
                mask = prued_mask[index]
                conv_in_mask[index + 3] = mask
                m.in_channels = mask.sum()
        tag += 1

    #############################################################
    bn_i = 0
    conv_i = 0
    scale_i = 0
    scale_mask = [mask4, mask3, mask2, mask1]
    #     scale = [70,86,90,94] # FPN
    scale = config['pruned']['scale']  # DB
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if (scale_i > 69):
            if isinstance(m0, nn.Conv2d):
                if (scale_i in scale):
                    index = scale.index(scale_i)
                    m1.in_channels = scale_mask[index].sum()
                    idx0 = np.squeeze(np.argwhere(np.asarray(scale_mask[index].cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(torch.ones(256).cpu().numpy())))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w = m0.weight.data[:, idx0, :, :].clone()
                    m1.weight.data = w[idx1, :, :, :].clone()
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.data[idx1].clone()

                else:
                    m1.weight.data = m0.weight.data.clone()
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.data.clone()

            elif isinstance(m0, nn.BatchNorm2d):
                m1.weight.data = m0.weight.data.clone()
                if m1.bias is not None:
                    m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

        else:
            if isinstance(m0, nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(bn_mask[bn_i].cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                m1.weight.data = m0.weight.data[idx1].clone()
                if m1.bias is not None:
                    m1.bias.data = m0.bias.data[idx1].clone()
                m1.running_mean = m0.running_mean[idx1].clone()
                m1.running_var = m0.running_var[idx1].clone()
                bn_i += 1
            elif isinstance(m0, nn.Conv2d):
                if (isinstance(conv_in_mask[conv_i], list)):
                    idx0 = np.squeeze(np.argwhere(np.asarray(torch.cat(conv_in_mask[conv_i], 0).cpu().numpy())))
                else:
                    idx0 = np.squeeze(np.argwhere(np.asarray(conv_in_mask[conv_i].cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(conv_out_mask[conv_i].cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w = m0.weight.data[:, idx0, :, :].clone()
                m1.weight.data = w[idx1, :, :, :].clone()
                if m1.bias is not None:
                    m1.bias.data = m0.bias.data[idx1].clone()
                conv_i += 1

        scale_i += 1

    print(new_model)

    save_obj = {'prued_mask': prued_mask, 'bn_index': bn_index}
    torch.save(save_obj, os.path.join(config['pruned']['save_checkpoints'], 'pruned_dict.dict'))
    torch.save(new_model.state_dict(), os.path.join(config['pruned']['save_checkpoints'], 'pruned_dict.pth.tar'))


if __name__ == '__main__':

    stream = open('./config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    prune(config)
