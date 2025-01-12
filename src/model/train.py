import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from model_utils import EarlyStopping
from model_utils import MyDataset

def train(x_train, y_train, x_val, y_val, epoch, model, transform = True):

    # Set model to training mode
    model = model.to(config.device) # move model to device (GPU or CPU)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 0.00001)

    # set dataloader
    train_dataset = MyDataset(x_train, y_train, transform)
    train_size = int(len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 4, shuffle = False, drop_last = True)
    val_dataset = MyDataset(x_val, y_val, transform)
    val_size = int(len(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 4, shuffle = False, drop_last = True)


    for e in range(epoch):
        train_loss_sum = 0
        train_acc_sum = 0
        val_loss_sum = 0
        val_acc_sum = 0
        model.train()
        for batch, (x, y) in enumerate(train_dataloader):
            x = x.unsqueeze(1)
            x, y = x.to(config.device, dtype = torch.float32), y.to(config.device, dtype = torch.long)

            optimizer.zero_grad()
            pred = model(x)
            #print(pred)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            pred = pred.argmax(1)
            train_acc_sum += pred.eq(y.view_as(pred)).sum().item()

        model.eval()
        with torch.no_grad():
            for x, y in val_dataset:
                x = x.unsqueeze(1)
                x, y = x.to(config.device, dtype = torch.float32), y.to(config.device, dtype = torch.long)
                pred = model(x)
                val_loss_sum += loss_fn(pred, y).item()
                pred = pred.argmax(1)
                val_acc_sum += pred.eq(y.view_as(pred)).sum().item()

        train_loss = train_loss_sum / train_size
        train_acc = train_acc_sum / train_size
        val_loss = val_loss_sum / val_size
        val_acc = val_acc_sum / val_size
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early Stop!")
            break
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        print(f"{e + 1} / {epoch} | Train Loss : {train_loss}, Acc : {100 * train_acc}, Val Loss : {val_loss}, Acc : {100 * val_acc}")