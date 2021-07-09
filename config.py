import torch.optim as optim
import torch.nn as nn
import encoder_decoder

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

outputs = []
num_epochs = 3
weight_decay = 1e-5

model = encoder_decoder.Conv_Autoencoder()
criterion = nn.MSELoss()
lr = 3e-4
optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-5)


save_model = True
load_model = True

transform = transforms.ToTensor()

mnist_data = datasets.MNIST(root='../data', train=True, download=True, transform=transform)

data_loader = DataLoader(dataset=mnist_data,
                         batch_size=64,
                         shuffle=True)