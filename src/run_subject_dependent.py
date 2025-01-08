#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   run_subject_dependent.py
@Time    :   2025/01/06 19:25:14
@Author  :   HuangZhiying 
@Email    :   zhiying.huang.4g@stu.hosei.ac.jp
@Description    :   None
'''

import random
import numpy as np

from dataload import get_subjects_data

def cross_validation_split_by_channel(data_dict, fold_num = 1, shuffle = False):
    
    # Using list to temporarily store the data
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    for subject, videos in data_dict.items():
        for video, channels in videos.items():
            for channel, segments in channels.items():

                if shuffle:
                    random.shuffle(segments) # random shuffle the segments, do not use KFold-validation

                # 80% for training, 20% for testing
                test_start_index = int(len(segments) * (0.2*(fold_num-1)))
                test_end_index = int(len(segments) * (0.2*fold_num))

                test_segments = segments[test_start_index:test_end_index]
                train_segments = segments[:test_start_index] + segments[test_start_index:test_end_index] + segments[test_end_index:]

                for segment in range(len(train_segments)):
                    x_train_list.append(train_segments[segment][0])
                    y_train_list.append(train_segments[segment][1])

                for segment in range(len(test_segments)):
                    x_test_list.append(test_segments[segment][0])
                    y_test_list.append(test_segments[segment][1])

                continue

    x_train = np.stack(x_train_list, axis=0)
    x_test = np.stack(x_test_list, axis=0)
    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    for i in range(1, 6):
        cross_validation_split_by_channel(get_subjects_data([1], list(range(1, 18))), fold_num = i)