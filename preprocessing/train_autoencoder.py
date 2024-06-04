import numpy as np
import pandas as pd 
import torch
import torch.nn as nn

def train_autoencoder(model, train_data, test_data, optimizer, criterion, n_epoch = 50):
    for epoch in range(n_epoch):
        output = model.forward(train_data)
        loss = criterion(output, train_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                test_output = model.forward(test_data)
                test_loss = nn.MSELoss()(test_output, test_data)
                print(f'Test Loss: {test_loss.item():.4f}')
            print(f'Epoch [{epoch+1}/{n_epoch}], Loss: {loss.item():.4f}')
    return model
