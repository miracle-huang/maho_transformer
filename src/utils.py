#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2025/01/03 18:36:27
@Author  :   HuangZhiying 
@Email    :   zhiying.huang.4g@stu.hosei.ac.jp
@Description    :   Utility functions
'''

from scipy.io import loadmat
import numpy as np
import sys

import config

def load_mat(filename, want):
    '''
    Load data from .mat file
    '''
    data = loadmat(filename)
    return data[want][0]

def window_segmentation(data, window_size = 10, sliding_size = 5):
    '''
    data: pandas DataFrame
    window_size: window size in seconds
    sliding_size: sliding size in seconds

    segmented_signals: list of segmented signals
    '''

    segmented_signals = []
    sampling_rate = 128 # sampling rate = 128 Hz
    window_data_size = window_size * sampling_rate
    sliding_step = sliding_size * sampling_rate
    
    start = 0
    while start + window_data_size <= len(data):
        end = start + window_data_size
        signal_ = data[start:end].copy()
        if len(signal_) == window_data_size:
            segmented_signals.append(signal_)
        start += sliding_step
    
    return segmented_signals

def calculate_average_label(subjects, channels, emotion_type = config.arousal):
    '''
    input:
    subjects: list of subjects
    channels: list of channels (EEG_channel_list: 1-14, ECG_channel_list: 15-16, GSR_channel_list: 17)
    emotion_type: one of 12 emotions in amigos dataset
    
    output:
    average_label: average label'''

    print("run calculate_average_label")

    label_list = [] # store all labels to calculate the average label

    for subject in subjects:
        # get all channels data and label data from list of subjects
        if subject < 10:
            subject = '0' + str(subject)
        else:
            subject = str(subject)
        subject_channels_data = load_mat("amigo/data_preprocessed/Data_Preprocessed_P" + str(subject) + "/Data_Preprocessed_P" + str(subject), "joined_data")
        subject_labels_data = load_mat("amigo/data_preprocessed/Data_Preprocessed_P" + str(subject) + "/Data_Preprocessed_P" + str(subject), "labels_selfassessment")
        
        for video in range(1, 21):
            # ergodic 20 videos
            video_data = subject_channels_data[video-1]
            video_labels = subject_labels_data[video-1][0]
            if video_data.any() and video_labels.any():

                # collect the non-NaN data channels, if all channels are NaN, skip this video
                nan_data_channels = 0 
                
                for channel in channels:
                    channel_data = video_data[channel-1]
                    if not np.isnan(channel_data).any():
                        try:
                            label = video_labels[emotion_type]
                            label_list.append(label)
                        except IndexError as e:
                            print(f"IndexError: Subject {subject}, Video {video}, Emotion Type {emotion_type}")
                            print(f"subject_labels_data shape: {subject_labels_data.shape}")
                            print(f"video: {video}")
                            sys.exit(1)
                        break
                    else:
                        nan_data_channels += 1
                
                if nan_data_channels == len(channels):
                    # print(f"Subject {subject}: all channels in Video {video} data contains NaN.")
                    continue
            elif not video_labels.any():
                # print(f"Subject {subject}: Video {video} label contains NaN.")
                continue
            else:
                # print(f"Subject {subject}: Video {video} data contains NaN.")
                continue
    
    average_label = sum(label_list) / len(label_list)
    greater_count, lesser_count = count_labels(label_list, average_label)
    print(f"Average label: {average_label}")
    print(f"Greater count: {greater_count}, Lesser count: {lesser_count}")
    return average_label

def count_labels(label_list, average_label):
    '''
    input:
    label_list: list of labels
    average_label: average label
    
    output:
    greater_count: count of labels greater than average label
    lesser_count: count of labels lesser than average label
    '''
    greater_count = 0
    lesser_count = 0

    for label in label_list:
        if label > average_label:
            greater_count += 1
        elif label < average_label:
            lesser_count += 1

    return greater_count, lesser_count

def print_dict_structure_only_keys(d, indent=0):
    """
    Recursively prints the key structure and data type of the dictionary, but does not display the details.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print('  ' * indent + f"{key}: dict")
            print_dict_structure_only_keys(value, indent + 1)
        elif isinstance(value, list):
            print('  ' * indent + f"{key}: list[{type(value[0]).__name__}]" if value else f"{key}: list")
        else:
            print('  ' * indent + f"{key}: {type(value).__name__}")

if __name__ == "__main__":
    subjects = list(range(1, 41))
    channels = config.EEG_channel_list
    arousal_ave = calculate_average_label(subjects, channels)
            