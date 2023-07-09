import  torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Linear(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Linear(6, 16, 5)
        self.fc1 = nn.Linear(512, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = nn.functional.relu(self.fc1(x))
        return x

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(nn.parameter(), lr=1e-4, momentum=0.9)

def train(net, dataloader, criterion, optimizer):
    train_loss = 0.0
    for index, data in enumerate(dataloader):
        inputs, labels = data
        optimizer.zero_grad()
        output = net(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss
        acc = torch.equal(labels, output)
        print(train_loss, acc)
def test(net, dataloader, criterion, optimizer):
    with torch.no_grad():
        test_loss = 0.0
        for index, data in enumerate(dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, labels)
            # loss.backward()
            optimizer.step()
            test_loss += loss
            acc = torch.equal(labels, output)
            print(test_loss, acc)
