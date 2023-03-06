import os
import torch
from torch import nn, cuda, tensor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(17, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,128),

        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class FeatureDataset(Dataset):
 
    def __init__(self,file_name):
        all_data=pd.read_csv(file_name)
        x=all_data.iloc[:,0:17].values
        y=all_data.iloc[:,17].values

        self.x_train=torch.tensor(x,dtype=torch.float32)
        self.y_train=torch.tensor(y)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]


def main():

    # data processing
    fd = FeatureDataset('../feature_data/features.csv')
    train_loader = torch.utils.data.DataLoader(fd)
    # allData.replace([np.inf, -np.inf], np.nan, inplace=True)
    # allData = allData.dropna()
    # X = allData.drop(["label"],axis=1)
    # X = X.drop(["radius"],axis=1)
    # cols = X.keys()
    # y = allData["label"]
    # X = pd.DataFrame(X, columns = cols)
    # torch_tensor = tensor(X[:].values)
    # print(torch_tensor)
    device = "cuda" if cuda.is_available() else "cpu"
    print(device)
    loss_func=nn.NLLLoss()
    # model definition
    model = NeuralNetwork().to(device)
    # model structure
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epochs = 10
    for e in range(epochs):
        running_loss = 0
        for features, labels in train_loader:
            output = model(features)
            loss = loss_func(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(loss)

if __name__ == "__main__":
    main()