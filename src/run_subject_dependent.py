#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   run_subject_dependent.py
@Time    :   2025/01/06 19:25:14
@Author  :   HuangZhiying 
@Email    :   zhiying.huang.4g@stu.hosei.ac.jp
@Description    :   None
'''

from dataload import get_subjects_data

def cross_validation_split_by_channel(data_dict, n_splits=5):
    
    all_channels = []

    for subject, videos in data_dict.items():
        for video, channels in videos.items():
            for channel, segments in channels.items():
                for segment in segments:
                    segment_data = segment[0]
                    lable_data = segment[1]
                    print(segment_data.shape, lable_data)

if __name__ == "__main__":
    cross_validation_split_by_channel(get_subjects_data([1], list(range(1, 18))))