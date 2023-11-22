import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import os
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper-parameters
latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 200
batch_size = 100
sample_dir = 'samples'
learning_rate = 0.0002
#Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
])
#get data (MNIST)
mnist = torchvision.datasets.MNIST(root='../../data',train=True,transform=transform,download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist,shuffle=True,batch_size=batch_size)

#Discriminator
D = nn.Sequential(
    nn.Linear(image_size,hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size,hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size,1),
    nn.Sigmoid()
)
G = nn.Sequential(
    nn.Linear(latent_size,hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size,hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size,image_size),
    nn.Tanh()
)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(),lr=learning_rate)
g_optimizer = torch.optim.Adam(G.parameters(),lr=learning_rate)

def denorm(x):
    out = (x + 1)/2
    return out.clamp(0,1)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

#Start training
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i,(images,_) in enumerate(data_loader):
        images = images.reshape(batch_size,-1)
        #Create the labels which are later used as input for the BCELoss function
        real_labels = torch.ones(batch_size,1)
        fake_labels = torch.zeros(batch_size,1)
        #===============================Train the discriminator===========================
        outputs = D(images)
        d_loss_real = criterion(outputs,real_labels)
        real_score = outputs
        #compute BCELoss using fake images
        #first term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size,latent_size)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs,fake_labels)
        fake_score = outputs

        #backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        #=========================Train the generator=======================================
        #compute loss with fake images
        z = torch.randn(batch_size, latent_size)
        fake_images = G(z)
        outputs = D(fake_images)
        #We train G to maximize log(D(G(z))) instead of minimizing log(1-D(G(z)))
        g_loss = criterion(outputs,real_labels)
        #backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
    
    #Save the real images
    if (epoch+1) == 1:
        images = images.reshape(images.size(0),1,28,28)
        save_image(denorm(images),os.path.join(sample_dir,'real_images.png'))
    
    #save sampled images
    fake_images = fake_images.reshape(fake_images.size(0),1,28,28)
    save_image(denorm(fake_images),os.path.join(sample_dir,'fake_images-{}.png'.format(epoch+1)))

#Save the model checkpoints
torch.save(G.state_dict(),'/GenerativeAdversarialNetwork/G.ckpt')
torch.save(D.state_dict(),'/GenerativeAdversarialNetwork/D.ckpt')