from torch import nn
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import os
import torchvision.io
import torch
import torch.nn.functional as F

device='cuda'

#https://tough-airmail-c2a.notion.site/IOAI-24-9d474249b60640ec880556e8597839d7?pvs=4
class ImageData(Dataset):
    def __init__(self,labels_dir,img_dir,transform=None,target_transform=None):
        self.image_labels=pd.read_csv(labels_dir,dtype={0:str})
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform
        
    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        image_label=str(self.image_labels.iloc[index,0])+'.png'
        img_path=os.path.join(self.img_dir,image_label)
        image=torchvision.io.read_image(img_path).to(torch.float32)
        label=self.image_labels.iloc[index,1]
        if self.transform:
            image=self.transform(image)
        if self.target_transform:
            label=self.target_transform(label)
        return image, label
  

learningRate=0.1
momentum=0.9

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten=nn.Flatten()
        # self.stack = nn.Sequential(
        #     nn.Linear(3*224*224, 512),
        #     nn.InstanceNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.InstanceNorm1d(256),
        #     nn.Sigmoid(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 1)  # Assuming binary classification
        # )
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(16, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x=self.sigmoid(x)
        return x

labels_dir="D:\\Descarcari\\unibuc-brain-ad\\data\\train_labels.txt"
img_dir='D:\\Descarcari\\unibuc-brain-ad\\data\\data'
train_dataset=ImageData(labels_dir,img_dir)
labels_dir="D:\\Descarcari\\unibuc-brain-ad\\data\\validation_labels.txt"
train_dataLoader=DataLoader(train_dataset,batch_size=64)
test_dataset=ImageData(labels_dir,img_dir)
test_dataLoader=DataLoader(test_dataset,batch_size=1)

model=NeuralNetwork().to(device)
loss_fn=nn.BCEWithLogitsLoss()
optim=torch.optim.SGD(model.parameters(),learningRate,momentum,weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)

def trainLoop(dataLoader,model,loss_fn,optim):
    model.train()
    size=len(dataLoader.dataset)
    for batch,(X,y) in enumerate(dataLoader):
        X,y=X.to(device),y.to(device)
        pred=model(X)
        pred=pred.squeeze()
        y=y.squeeze().float()
        pred=torch.sigmoid(pred)
        loss=loss_fn(pred,y)
        loss.backward()
        optim.step()
        optim.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        scheduler.step()

def testLoop(dataLoader, model, loss_fn):
    model.eval()
    total, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataLoader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred=torch.sigmoid(pred)
            #pred = torch.sigmoid(pred).squeeze()  # Apply sigmoid to get probabilities
            predicted_classes = (pred > 0.5).long()  # Threshold probabilities to get binary predictions
            # print(f"{predicted_classes}---{y}")
            total += y.size(0)
            correct += (predicted_classes == y).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    

trainLoop(train_dataLoader,model,loss_fn,optim)
testLoop(test_dataLoader,model,loss_fn)


            