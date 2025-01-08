#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   run_subject_dependent.py
@Time    :   2025/01/05 00:11:19
@Author  :   HuangZhiying 
@Email    :   zhiying.huang.4g@stu.hosei.ac.jp
@Description    :   run subject dependent model
'''

import numpy as np
from sklearn.model_selection import KFold

from dataload import get_subjects_data
import config

import numpy as np

def cross_validation_split_by_channel(data_dict, n_splits=5):
    """
    进行5折交叉验证，并确保每折的数据在最后都展平并保持窗口和标签的一一对应。
    
    参数:
    data_dict: get_subjects_data的输出，包含每个subject、video和channel的数据。
    n_splits: 交叉验证的折数，默认为5。
    
    返回:
    splits: 包含每折训练和测试数据的列表，每个元素是一个字典，包含'train'和'test'键。
    """
    splits = []

    # 将数据按channel分割为列表
    all_channels = []
    for subject, videos in data_dict.items():
        for video, channels in videos.items():
            for channel, segments in channels.items():
                all_channels.append((subject, video, channel, segments))

    # 计算每折的大小
    fold_size = len(all_channels) // n_splits

    for i in range(n_splits):
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        # 选择当前折的测试集
        test_channels = all_channels[i * fold_size:(i + 1) * fold_size]
        train_channels = all_channels[:i * fold_size] + all_channels[(i + 1) * fold_size:]

        # 处理训练集
        for subject, video, channel, segments in train_channels:
            for segment, label in segments:
                train_data.append(segment)
                train_labels.append(label)

        # 处理测试集
        for subject, video, channel, segments in test_channels:
            for segment, label in segments:
                test_data.append(segment)
                test_labels.append(label)

        # 将数据和标签转换为numpy数组
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        # 将数据和标签打包成字典
        split = {
            'train': (train_data, train_labels),
            'test': (test_data, test_labels)
        }
        splits.append(split)

    return splits

if __name__ == "__main__":
    # 示例用法
    data_dict = get_subjects_data(list(range(1, 41)), config.EEG_channel_list)
    splits = cross_validation_split_by_channel(data_dict)

    # 每折的数据可以直接传入深度学习模型进行训练
    for i, split in enumerate(splits):
        train_data, train_labels = split['train']
        test_data, test_labels = split['test']
        print(f"Fold {i+1}:")
        print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
        print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")