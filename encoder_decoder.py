import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import config
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import (
    save_checkpoint,
    load_checkpoint
)

transform = transforms.ToTensor()

mnist_data = datasets.MNIST(root='../data', train=True, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=64, shuffle=True)


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


def train(outputs, model, criterion, optimizer):
    for (img, _) in data_loader:
        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    outputs.append((img, recon))


save_model = True
load_model = True


def train_loop():
    if load_model:
        load_checkpoint(checkpoint_file="my_checkpoint.pth.tar", model=config.model,
                        optimizer=config.optimizer, lr=config.lr)

    for epoch in range(config.num_epochs):
        print(f"epoch {epoch + 1}")
        train(config.outputs, model=config.model, criterion=config.criterion,
              optimizer=config.optimizer)

        if save_model:
            save_checkpoint(model=config.model, optimizer=config.optimizer)


def plot():
    for k in range(0, config.num_epochs, 4):
        plt.figure(figsize=(9, 2))
        plt.gray()
        imgs = config.outputs[k][0].detach().numpy()
        recon = config.outputs[k][1].detach().numpy()

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
    # train_loop()
    # plot()
    pass
