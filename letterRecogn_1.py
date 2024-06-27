import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as f
import numpy as np

learningRate=1e-1
output_file='prediction.csv'
device= 'cuda' if torch.cuda.is_available() else 'cpu'

class NumbersDataset(Dataset):
    def __init__(self,csv):
        df=pd.read_csv(csv)
    
        if 'label' in df.columns:
            self.y=np.array(df['label'])
            self.X=np.array(df.drop(columns='label')).astype(np.float32)
        else:
            self.X=np.array(df).astype(np.float32)
            self.y=None
        self.X = self.X.reshape((-1, 28, 28))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index): 
        X=torch.tensor(self.X[index])
        if self.y is not None:
            y=torch.tensor(self.y[index])
        else:
            y=torch.zeros(1)
        return X,y

train_dataset=NumbersDataset('numbers.csv') 
Dloader_train=DataLoader(train_dataset,batch_size=64,shuffle=True)

test_dataset=NumbersDataset('test.csv')
Dloader_test=DataLoader(test_dataset,batch_size=1,shuffle=False)



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(16)
        self.pool=nn.AvgPool2d(kernel_size=2)
        self.conv2=nn.Conv2d(16,32,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(32)
        
        self.fc1=nn.Linear(32*7*7,64)
        self.dropout=nn.Dropout(0.5)
        self.fc2=nn.Linear(64,10)

    def forward(self,x):
        x=self.pool(f.relu(self.bn1(self.conv1(x))))
        x=self.pool(f.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x=f.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x)
        return x

model=NeuralNetwork().to(device)
loss_fn=nn.CrossEntropyLoss()
optim=torch.optim.SGD(model.parameters(),lr=learningRate,momentum=0.1)
scheduler=torch.optim.lr_scheduler.StepLR(optim,step_size=10,gamma=0.1)

def trainLoop(dataloader, model, loss_fn, optim):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.view(-1, 1, 28, 28).float()
        X,y=X.to(device),y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optim.step()
        optim.zero_grad()
        #scheduler.step()

        if batch % 100 == 0:
            loss_val = loss.item()
            current = batch * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
        
def testLoop(dataloader,model,loss_fn):
    model.eval()
    label=1
    with torch.no_grad(),open ('prediction.csv','w') as f:
        f.write('ImageID,Label\n')
        for X,y in dataloader:
            X=X.view(-1,1,28,28).float()
            X,y=X.to(device),y.to(device)
            pred=model(X)
            probab=nn.Softmax(dim=1)(pred)
            y_pred=probab.argmax(1)
            f.write(f"{label},{y_pred.item()}\n")
            label+=1
            
epochs=50

for epoch in range(epochs):
    trainLoop(Dloader_train, model, loss_fn, optim)
    
testLoop(Dloader_test, model, loss_fn)