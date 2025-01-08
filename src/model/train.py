import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config

def train(dataloader, epoch, model, optimizer):
    size = len(dataloader.dataset)

    model = model.to(config.device) # move model to device (GPU or CPU)
    loss_fn = nn.CrossEntropyLoss()

    for e in range(epoch):
        train_loss_sum = 0
        train_acc_sum = 0
        val_loss_sum = 0
        val_acc_sum = 0
        model.train()
        for batch, (x, y) in enumerate(dataloader):
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
            for x, y in test_dataloader:
                x = x.unsqueeze(1)
                x, y = x.to(config.device, dtype = torch.float32), y.to(config.device, dtype = torch.long)
                pred = model(x)
                val_loss_sum += loss_fn(pred, y).item()
                pred = pred.argmax(1)
                val_acc_sum += pred.eq(y.view_as(pred)).sum().item()

        train_loss = train_loss_sum / len(dataloader)
        train_acc = train_acc_sum / train_size
        val_loss = val_loss_sum / len(dataloader)
        val_acc = val_acc_sum / test_size
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