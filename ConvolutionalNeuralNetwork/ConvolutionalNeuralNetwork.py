import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F #for the activation function
import torchvision.transforms as transforms

#device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Hyper parameters
epochs=15
batch_size=100
learning_rate=0.0005
num_classes=10
#get the data
train_set = torchvision.datasets.MNIST(root="../../data",train=True,transform=transforms.ToTensor())
test_set = torchvision.datasets.MNIST(root="../../data",train=False,transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)

#define the model - We are creating a convolutional net
class ConvNet(nn.Module):
    def __init__(self,num_classes):
        super(ConvNet,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,16,kernel_size=4,padding=2,stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(16,32,padding=2,kernel_size=4,stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc =nn.Linear(7*7*32,num_classes)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0),-1)
        out = self.fc(out)
        return out
    
#instantiate the model class
model = ConvNet(num_classes=num_classes)

#declare a loss function and an optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#train the model over multiple iterations
total_steps = len(train_loader)
for epoch in range(epochs):
    for i,(images,labels) in enumerate(train_loader):
        #forward pass
        outputs = model(images)
        loss = loss_function(outputs,labels)
        #back propragation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)%200==0:
            print("Epoch[{}/{}] Step[{}/{}] Loss:{:.4f}"
                  .format(epoch+1,epochs,i+1,total_steps,loss.item()))

model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for images,labels in test_loader:
        outputs = model(images)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
    print("Total correctly matched images : {} %".format(100*correct/total))



