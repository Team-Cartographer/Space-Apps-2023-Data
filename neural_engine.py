from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())  # double check the file source
# I need to review the documentation for this, but it should properly import most of the dependencies that we need
# we need to come back to this later, but good enough for now
dataset = DataLoader(train, 32)
