import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_size = 1.0
lr = 2e-4
weight_decay = 8e-9
beta1 = 0.5
beta2 = 0.999
batch_size = 256
epochs = 100
plot_every = 10
np.random.seed(42)


def mnist_data(train_part, path, transform):
    dataset = MNIST(path, download=True, transform=transform)
    train_part = int(train_part * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_part, len(dataset)-train_part])
    return train_dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])
train_data = mnist_data(train_size, './Demo/dataset', transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

def plotn(n, generator, device):
    generator.eval()
    noise = torch.FloatTensor(np.random.normal(0.0, 1.0, (n, 100))).to(device)
    imgs = generator(noise).detach().cpu()
    fig, ax = plt.subplots(1, n)
    for i, img in enumerate(imgs):
        ax[i].imshow(img[0])
    plt.show()

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 256)
        self.bn1 = nn.BatchNorm1d(256, momentum=0.2)
        self.linear2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512, momentum=0.2)
        self.linear3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024, momentum=0.2)
        self.linear4 = nn.Linear(1024, 784)
        self.bn4 = nn.BatchNorm1d(784, momentum=0.2)
        self.leaky_relu = nn.LeakyReLU(inplace=0.2)
        self.tanh = nn.Tanh()
    
    def forward(self, input):
        # input -> noise
        hidden1 = self.leaky_relu(self.bn1(self.linear1(input)))
        hidden2 = self.leaky_relu(self.bn2(self.linear2(hidden1)))
        hidden3 = self.leaky_relu(self.bn3(self.linear3(hidden2)))
        out = self.leaky_relu(self.bn4(self.linear4(hidden3))).view(input.shape[0], 1, 28, 28)
        return self.tanh(out)
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3  = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(inplace=0.2)
    
    def forward(self, input):
        input = input.view(input.shape[0], -1)
        hidden1 = self.leaky_relu(self.linear1(input))
        hidden2 = self.leaky_relu(self.linear2(hidden1))
        out = self.sigmoid(self.linear3(hidden2))
        return out

generator = Generator().to(device)
discriminator = Discriminator().to(device)
optim_generator = torch.optim.Adam(generator.parameters(), lr=lr, weight_decay=weight_decay)
optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = nn.BCELoss()
models = (generator, discriminator)
optimizers = (optim_discriminator, optim_generator)

def train_gan(data_loders, models, optimizers, loss_fn, epochs, plot_every, device):
    tqdm_iter = tqdm(range(epochs))
    g, d = models[0], models[1]
    optim_g, optim_d = optimizers[0], optimizers[1]

    for epoch in tqdm_iter:
        for batch in data_loders:
            train_g_loss = 0.0
            train_d_loss = 0.0

            img, _ = batch
            img = img.to(device)

            real_labels = torch.ones((img.shape[0], 1)).to(device)
            fake_labels = torch.zeros((img.shape[0], 1)).to(device)
            noise = torch.FloatTensor(np.random.normal(0.0, 0.1, (img.shape[0], 100))).to(device)

            d.train()
            d.zero_grad()

            d_real = d(img)
            d_real_loss = loss_fn(d_real, real_labels)
            g_fake = g(noise)
            d_fake = d(g_fake.detach())
            d_fake_loss = loss_fn(d_fake, fake_labels)
            d_loss = (d_real_loss + d_fake_loss) / 2.0
            d_loss.backward()
            optim_d.step()

            g.train()
            g.zero_grad()

            noise = torch.FloatTensor(np.random.normal(0.0, 1.0, (img.shape[0], 100))).to(device)
            g_fake = g(noise)
            d_preds = d(g_fake)

            g_loss = loss_fn(d_preds, real_labels)
            g_loss.backward()
            optim_g.step()

            train_g_loss += g_loss.item()
            train_d_loss += d_loss.item()
        
        if epoch % plot_every == 0 or epoch  == epochs-1:
            plotn(5, g, device)
        
        tqdm_dict = {'generator loss: ': train_g_loss, 'discriminator loss: ': train_d_loss}
        tqdm_iter.set_postfix(tqdm_dict, refresh=True)
        tqdm_iter.refresh()

    
if __name__ == '__main__':
    train_gan(train_loader, models, optimizers, loss_fn, epochs, plot_every, device)
 


        