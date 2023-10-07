from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
input_size: int  # TODO define this later
hidden_size: int  # TODO define this later
num_classes: int  # TODO define this later
num_epochs: int  # TODO define this later
batch_size: int  # TODO define this later
learning_rate: int = 0.001  # I have no clue what this does, but this is what the tutorial had. Change later?

train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())  # double check the file source
# I need to review the documentation for this, but it should properly import most of the dependencies that we need
# we need to come back to this later, but good enough for now
dataset = DataLoader(train, 32)

raise CodeNotWrittenError
