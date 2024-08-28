# Initialising Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import v2
import torchsummary
import torchmetrics

print("CUDA" if torch.cuda.is_available() else "No CUDA")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
print(device)

torch.manual_seed(42)

class conv_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,64, 5)
        self.batch1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64,128, 5)
        self.batch2 = torch.nn.BatchNorm2d(128)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.flat = torch.nn.Flatten()
        self.hc1 = torch.nn.Linear(21632,512)
        self.hc2 = torch.nn.Linear(512,32)
        self.out = torch.nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.pool(self.batch1(F.relu(self.conv1(x))))
        x = self.pool(self.batch2(F.relu(self.conv2(x))))
        x = F.relu(self.hc1(self.flat(x)))
        x = F.tanh(self.hc2(x))
        x = F.softmax(self.out(x), dim=1)
        return x


class skin_cancer_classifier():
    def __init__(self, model, device, cpu):
        self.model = model
        self.device = device
        self.cpu = cpu
    
    def summary(self):
        self.model.to(self.device)
        torchsummary.summary(self.model, input_size=(3, 64, 64), device = self.device)

    def train(self, dataloader, lr, epochs):
        self.model.to(self.device)
        criterion = torch.nn.BCELoss()
        optim = torch.optim.Adam(self.model.parameters(), lr = lr)
        accuracy = torchmetrics.classification.BinaryAccuracy().to(self.device)

        avg_epoch_loss = []
        avg_epoch_acc = []

        for e in range(epochs):
            print(f'Epoch: {e+1}\n')
            losseslist = []
            acclist = []
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.reshape(-1,1).to(self.device)

                outputs = self.model(images)

                loss = criterion(outputs.to(torch.float64), labels.to(torch.float64))
                optim.zero_grad()
                loss.backward()
                optim.step()

                acclist.append(accuracy(outputs, labels))
                losseslist.append(loss.item())

            acclist_np = [x.to(self.cpu).detach().numpy() for x in acclist]
            avg_epoch_acc.append(np.mean(acclist_np))

            avg_epoch_loss.append(np.mean(losseslist))
            

            print(f"Losses: {losseslist}")
            print(f"Accuracies: {acclist}")
        

        plt.plot(range(epochs),avg_epoch_loss, c="blue") 
        plt.ylabel("Average Loss")
        plt.show()
        plt.plot(range(epochs),avg_epoch_acc, c="orange")
        plt.ylabel("Average Accuracy")
        plt.show()
    
    def test(self, dataloader):
        accuracy = torchmetrics.classification.BinaryAccuracy().to(self.device)
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.reshape(-1,1).to(self.device)

            outputs = self.model(images)
            print(accuracy(outputs, labels))


transforms = v2.Compose([
    v2.Resize(size = (64, 64), antialias=True),
    v2.RandomHorizontalFlip(p = 0.5),
    v2.RandomVerticalFlip(p = 0.5),
    v2.RandomPerspective(distortion_scale=0.6, p=0.25),
    v2.ToDtype(torch.float64, scale=True),
    torchvision.transforms.ToTensor()
])

dataset_train = torchvision.datasets.ImageFolder("train_cancer/train", transform=transforms)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)
 
dataset_test = torchvision.datasets.ImageFolder("train_cancer/test", transform=transforms)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=32, shuffle=True)

model = conv_model()
model_ = skin_cancer_classifier(model, device, cpu)
model_.summary()
model_.train(dataloader_train, 1e-1, 50)
model_.test(dataloader_test)

