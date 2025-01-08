#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   dataload.py
@Time    :   2025/01/03 02:50:59
@Author  :   HuangZhiying 
@Email    :   zhiying.huang.4g@stu.hosei.ac.jp
@Description    :   Load data from .mat file, and return the data you want.
'''

import numpy as np

import config
from utils import window_segmentation
from utils import load_mat
from utils import calculate_average_label
from utils import print_dict_structure_only_keys

def get_subjects_data(subjects, channels, emotion_type = config.arousal):
    '''
    subjects: list of subjects
    channels: list of channels (EEG_channel_list: 1-14, ECG_channel_list: 15-16, GSR_channel_list: 17)
    emotion_type: one of 12 emotions in amigos dataset
    '''

    average_label = calculate_average_label(subjects, channels)
    data_dict = {}

    for subject in subjects:
        # get all channels data and label data from list of subjects
        if subject < 10:
            subject = '0' + str(subject)
        else:
            subject = str(subject)
        subject_channels_data = load_mat("amigo/data_preprocessed/Data_Preprocessed_P" + str(subject) + "/Data_Preprocessed_P" + str(subject), "joined_data")
        subject_labels_data = load_mat("amigo/data_preprocessed/Data_Preprocessed_P" + str(subject) + "/Data_Preprocessed_P" + str(subject), "labels_selfassessment")

        subject_data = {}
        
        for video in range(1, 21):
            # ergodic 20 videos

            # Transpose the matrix. Dimension 0 is 17 channels (17, 12225)
            video_data = subject_channels_data[video-1].T # videos start from 1
            video_labels = subject_labels_data[video-1][0] # videos start from 1

            if video_data.any() and video_labels.any():
                video_data_dict = {}

                # if video_data.any() and subject_labels_data[video-1].any():
                for channel in channels:
                    # ergodic 17 channels
                    channel_data = video_data[channel-1] # channels start from 1

                    if not np.isnan(channel_data).any():
                        segmented_channel_data = window_segmentation(channel_data)
                        labels = [1 if video_labels[emotion_type] > average_label else 0] * len(segmented_channel_data)
                        video_data_dict[channel] = list(zip(segmented_channel_data, labels))
                    else:
                        print(f"Subject {subject}: Video {video} Channel {channel} data contains NaN.")

                subject_data[video] = video_data_dict
            elif not video_labels.any():
                print(f"Subject {subject}: Video {video} label contains NaN.")
                # continue
            else:
                print(f"Subject {subject}: Video {video} data is empty.")
                # continue

        data_dict[subject] = subject_data

    return data_dict

if __name__ == "__main__":
    print("This is dataload.py")
    get_subjects_data(list(range(1, 2)), config.EEG_channel_list)