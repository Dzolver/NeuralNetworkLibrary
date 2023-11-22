import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.003

train_dataset = torchvision.datasets.MNIST(root="../../data",transform = transforms.ToTensor(),download = True,train=True)
test_dataset = torchvision.datasets.MNIST(root="../../data",transform=transforms.ToTensor(),download=True,train=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,shuffle=False,batch_size=batch_size)

class BiRNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(BiRNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size*2,num_classes)

    def forward(self,x):
        #Set initial hidden states
        h0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size)
        c0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size)
        #Forward propagate LSTM
        out, _ =self.lstm(x,(h0,c0))
        #Decode the hidden state of the last time step
        out = self.fc(out[:,-1,:])
        return out

model = BiRNN(input_size,hidden_size,num_layers,num_classes)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

#Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        #Forward Pass
        images = images.reshape(-1,sequence_length,input_size)
        outputs = model(images)
        loss = criterion(outputs,labels)
        #backwards and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
#Testing the model
with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_loader:
        images = images.reshape(-1,sequence_length,input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print("Testing accuracy of the model on 10000 test images : {}%".format(100*correct/total))

#Save the model checkpoint
torch.save(model.state_dict(),'BiDirectionalRNN.ckpt')
