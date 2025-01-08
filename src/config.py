import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

EEG_channel_list = list(range(1, 15))
ECG_channel_list = [15, 16]
GSR_channel_list = [17]

arousal = 0

# Deeplearning parameters
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")