from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset,DataLoader
from skimage.io import imread_collection
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F

device='cuda' if torch.cuda.is_available() else 'cpu'

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip()
])

#768x768
learningRate=1e-1
momentum=0.9

class ImageData(Dataset):
    def __init__(self,images_dir,transform=None,target_transform=None):
        self.images_coll=imread_collection(images_dir)
        self.transform=transform
        self.target_transform=target_transform
    
    def __len__(self):
        return len(self.images_coll)
    
    def __getitem__(self, index):
        image=self.images_coll[index]
        file_path=self.images_coll.files[index]
        file_name=os.path.basename(file_path)
        label=0
        if 'lungaca' in file_name:
            label=0
        elif 'lungn' in file_name:
            label=1
        elif 'lungssc' in file_name:
            label=2
        if self.transform:
            image=self.transform(image)
        image=image.to(torch.float32)/225.0
        return image,label
        
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(64)
        self.pool=nn.AvgPool2d(kernel_size=3,stride=2,padding=0)
        #floor((dimensiune-kernel+2*padding)/stride)+1
        
        self.conv2=nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(64)
        
        self.conv3=nn.Conv2d(64,32,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(32)
        
        self.conv4=nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.bn4=nn.BatchNorm2d(32)
        
        self.conv5=nn.Conv2d(32,16,kernel_size=3,stride=1,padding=1)
        self.bn5=nn.BatchNorm2d(16)
        
        self.fc1=nn.Linear(16*23*23,256)
        self.dropout=nn.Dropout(0.1)
        self.fc2=nn.Linear(256,3)
    
    def forward(self,x):
        x=self.pool(F.relu(self.bn1(self.conv1(x)))) #floor((768-3+2*0)/2)+1=383
        x=self.pool(F.relu(self.bn2(self.conv2(x)))) #floor((383-3)/2)+1=191
        x=self.pool(F.relu(self.bn3(self.conv3(x)))) #floor((191-3)/2)+1=95
        x=self.pool(F.relu(self.bn4(self.conv4(x)))) #floor((95-3)/2)+1=47
        x=self.pool(F.relu(self.bn5(self.conv5(x)))) #floor((47-3)/2)+1=23
        x=x.view(x.size(0),-1) # x va fi de forma batchSize*neuronii de output de la CNN(8)*23*23 adica 16*8*23*23
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x)
        return x
        

images_dir='D:\\Descarcari\\archive _lunh\\lung\\Train\\All\\*.jpeg'
train_dataset=ImageData(images_dir,transform)
DLoaderTrain=DataLoader(train_dataset,batch_size=16,shuffle=True)
test_dir='D:\\Descarcari\\archive _lunh\\lung\\Test\\All\\*.jpeg'
test_dataset=ImageData(test_dir,transform)
DLoaderTest=DataLoader(test_dataset,batch_size=1)


model=NeuralNetwork().to(device)
loss_fn=nn.CrossEntropyLoss()
optim=torch.optim.SGD(model.parameters(),learningRate,momentum,weight_decay=0.01)

def TrainLoop(dataloader,model,loss_fn,optim):
    model.train()
    size=len(dataloader.dataset)
    for batch,(X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optim.step()
        
        if batch % 100 == 0:
            loss_val = loss.item()
            current = batch * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

def TestLoop(dataloader,moddel,loss_fn):
    model.eval()
    correct=0
    with torch.no_grad():
        for X,y in dataloader:
            X,y=X.to(device),y.to(device)
            pred=model(X)
            probab=nn.Softmax(dim=1)(pred)
            y_pred=probab.argmax(1)
            if y_pred==y:
                correct+=1
            print(probab)
    
    print(correct*100/len(dataloader.dataset))

for i in range(1,5):
    TrainLoop(DLoaderTrain,model,loss_fn,optim)
TestLoop(DLoaderTest,model,loss_fn)