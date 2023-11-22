import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

#load the data
train_dataset = torchvision.datasets.MNIST(root='../../data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data',train=False,transform=transforms.ToTensor(),download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,shuffle=True,batch_size=batch_size)

#fully connected neural network with one hidden layer
class NeuralNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNetwork,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
model = NeuralNetwork(input_size=input_size,hidden_size=hidden_size,num_classes=num_classes)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        images = images.reshape(-1,input_size)
        #forward pass
        outputs = model(images)
        loss = criterion(outputs,labels)
        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_steps, loss.item()))
            
#Testing model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Network accuracy over 10000 test images : {}%".format(100* correct/total))

#save the model checkpoint
torch.save(model.state_dict(),'FeedForwardNeuralNetwork.ckpt')