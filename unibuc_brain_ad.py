from torch import nn
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import os
import torchvision.io
import torch

device='cuda'

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
  

learningRate=1e-1 
momentum=0.1

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        self.stack=nn.Sequential(
            nn.Linear(3*224*224,1024),
            nn.Sigmoid(),
            nn.Linear(1024,512),
            nn.Sigmoid(),
            nn.Linear(512,128),
            nn.Sigmoid(),
            nn.Linear(128,32),
            nn.Sigmoid(),
            nn.Linear(32,1),
        )
    
    def forward(self,x):
        x=self.flatten(x)
        logits=self.stack(x)
        return logits

labels_dir="D:\\Descarcari\\unibuc-brain-ad\\data\\train_labels.txt"
img_dir='D:\\Descarcari\\unibuc-brain-ad\\data\\data'
train_dataset=ImageData(labels_dir,img_dir)
labels_dir="D:\\Descarcari\\unibuc-brain-ad\\data\\validation_labels.txt"
train_dataLoader=DataLoader(train_dataset,batch_size=1)
test_dataset=ImageData(labels_dir,img_dir)
test_dataLoader=DataLoader(test_dataset,batch_size=1)

model=NeuralNetwork().to(device)
loss_fn=nn.BCEWithLogitsLoss()
optim=torch.optim.SGD(model.parameters(),learningRate,momentum)

def trainLoop(dataLoader,model,loss_fn,optim):
    model.train()
    size=len(dataLoader.dataset)
    for batch,(X,y) in enumerate(dataLoader):
        X,y=X.to(device),y.to(device)
        pred=model(X)
        pred=pred.squeeze()
        y=y.squeeze().float()
        loss=loss_fn(pred,y)
        
        loss.backward()
        optim.step()
        optim.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
def testLoop(dataLoader, model, loss_fn):
    model.eval()
    total, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataLoader:
            X,y=X.to(device),y.to(device)
            pred = model(X)
            predicted_classes = torch.argmax(pred, dim=1)
            total += y.size(0)
            correct += (predicted_classes == y).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

trainLoop(train_dataLoader,model,loss_fn,optim)
testLoop(test_dataLoader,model,loss_fn)


            