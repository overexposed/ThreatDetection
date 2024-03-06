import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import optim
from tqdm import tqdm


def oneHotEncode(df,colNames):
    for col in colNames:
        if( df[col].dtype == np.dtype('object') or True):
            dummies = pd.get_dummies(df[col],prefix=col)
            df = pd.concat([df,dummies],axis=1)

            #drop the encoded column
            df.drop([col],axis = 1 , inplace=True)
    return df

def preprocessing(df):
    # df["label"][df["label"]=="benign"] = 0
    # df["label"][df["label"]=="outlier"] = 1
    # df["label"][df["label"]=="malicious"] = 2
    # df["label"] = df["label"].astype(np.int16)

    # Normalization 
    df["dest_port"] = df["dest_port"] / df["dest_port"].max()
    df["src_port"] = df["src_port"] / df["src_port"].max()
    df["dest_port"][df["dest_port"].isna()] = -1
    df["src_port"][df["src_port"].isna()] = -1

    # # Binary Decision
    # df["dest_port"][~df["dest_port"].isna()] = 0
    # df["src_port"][~df["src_port"].isna()] = 0

    # df["dest_port"][df["dest_port"].isna()] = 1
    # df["src_port"][df["src_port"].isna()] = 1

class ThreatDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.features = self.df.drop(columns = ['label']).values
        df_enc = oneHotEncode(self.df, ["label"])
        self.label = df_enc[["label_benign", "label_malicious", "label_outlier"]].to_numpy()
        # self.label = self.df.label.values
        
    def __len__(self):
        return (len(self.df))
    
    def __getitem__(self, idx):
        return (torch.tensor(self.features[idx], dtype=torch.float), torch.tensor(self.label[idx], dtype=torch.float))
    
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(15, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def test(dataloader, model):
    size = len(test_loader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(test_loader):
            pred = model(X)
            test_loss += loss_fn(pred, y).item()  # View target labels as column vector
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



path = "Dataset/2020/06/2020.06.19/2020.06.19.csv"
df = pd.read_csv(path)
preprocessing(df)


X = df.drop(columns = ['label'])
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
df_train = pd.concat([pd.DataFrame(X_train), y_train], axis=1)
df_test = pd.concat([pd.DataFrame(X_test), y_test], axis=1)

train_loader = DataLoader(ThreatDataset(df_train), batch_size=64, shuffle=True)
test_loader = DataLoader(ThreatDataset(df_test), batch_size=64, shuffle=False)

model = NeuralNetwork().float()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Train the model (adjusting for the different feature size)
epochs = 5
size = len(train_loader.dataset)
for t in (range(epochs)):
    for batch, (X, y) in tqdm(enumerate(train_loader), total=size/64):
        # Compute prediction and loss
        X = X.view(-1, 15)  # Reshape input to match the feature size
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_val, current = loss.item(), batch * len(X)
    print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
    print(f"Epoch:{t+1}/{epochs}")

test(test_loader, model)

torch.save(model.state_dict(), 'model_binaryPort.pt')
