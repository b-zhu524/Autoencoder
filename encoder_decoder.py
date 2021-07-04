import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# import numpy as np


transform = transforms.ToTensor()

mnist_data = datasets.MNIST(root='../data', train=True, download=True, transform=transform)

data_loader = DataLoader(dataset=mnist_data,
                         batch_size=64,
                         shuffle=True)

data_iter = iter(data_loader)
images, labels = next(data_iter)
# print(torch.min(images), torch.max(images))

save_model = True
load_model = False


class Conv_Autoencoder(nn.Module):
    def __init__(self):
        super(Conv_Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=1), # N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),    # N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(7, 7))   # -> N, 64, 1, 1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train(outputs):
    model = Conv_Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    num_epochs = 3
    for epoch in range(num_epochs):
        for (img, _) in data_loader:
            recon = model(img)
            loss = criterion(recon, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")
        outputs.append((epoch, img, recon))


def plot():
    outputs = []
    num_epochs = 3

    train(outputs)

    for k in range(0, num_epochs, 4):
        plt.figure(figsize=(9, 2))
        plt.gray()
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()

        for i, item in enumerate(imgs):
            if i >= 9:
                break

            plt.subplot(2, 9, i+1)
            plt.imshow(item[0])

        for i, item in enumerate(recon):
            if i >= 9:
                break
            plt.subplot(2, 9, 9+i+1)    # row_length + i + 1
            plt.imshow(item[0])

        plt.show()


if __name__ == '__main__':
    plot()
