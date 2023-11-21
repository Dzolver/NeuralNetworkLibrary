import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input_size = 28
sequence_length = 28
hidden_size = 128
num_layers = 2
num_classes =10
batch_size = 100
num_epochs = 10
learning_rate =0.002

training_set = torchvision.datasets.MNIST(root="../../data",transform=transforms.ToTensor(),train=True,download=True)
test_set = torchvision.datasets.MNIST(root="../../data",transform=transforms.ToTensor(),train=False,download=True)
train_load = torch.utils.data.DataLoader(dataset=training_set,batch_size=100,shuffle=True)
test_load = torch.utils.data.DataLoader(dataset=test_set,batch_size=100,shuffle=True)

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(RNN,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size,num_classes)
    
    def forward(self,x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        out,_ = self.lstm(x,(h0,c0))
        out = self.fc(out[:,-1,:])
        return out

model = RNN(input_size,hidden_size,num_layers,num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

def get_accuracy(logit,target,batch_size):
    corrects = (torch.max(logit,1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

train_running_loss = 0.0
train_acc=0.0
total_step = len(train_load)
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_load):
        images = images.reshape(-1,sequence_length,input_size)
        
        outputs = model(images)
        loss = criterion(outputs,labels)
        
        optimizer.zero_grad
        loss.backward()
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(outputs,labels,batch_size)

        if (i+1)%200==0:
            print("Epoch[{}/{}] Step[{}/{}] Loss:{:.4f} Train Accuracy: {:.2f}".format((epoch+1),num_epochs,(i+1),total_step,loss.item(),train_acc/i))

model.eval()
with torch.no_grad():
    correct=0
    total = 0
    for images,labels in test_load:
        images = images.reshape(-1,sequence_length,input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

    print("Test accuracy of the model on the 10000 test images: {} %".format(100*correct/total))

torch.save(model.state_dict(),'model.ckpt') 