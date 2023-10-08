import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm 
import data_manager as dm 
from utils import *

raise ObsoleteCodeError

pkl_path = "dataset.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset): 
    def __init__(self, data, transform=transforms.ToTensor()):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class SimpleNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, out):
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_size: int = 647736 # declare this!!!
hidden_size: int = 20 # declare this!!!
output_size: int = 181 # declare this!!!
batch_size: int = 100

model = SimpleNN(input_size, hidden_size, output_size)
dataset = CustomDataset(data=dm.get_training_data())

# I have no idea what this does
criterion = nn.CrossEntropyLoss()  # Example loss function for classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Example optimizer


# Data Sets

# replaced with CustomDataset 
# train_dataset = torchvision.datasets.MNIST(root="./data",
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)

test_dataset = torchvision.datasets.MNIST(root="./data",
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)

# Data loaders
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)


# train the nn
num_epochs = 100  # Number of training epochs. Change this!!!!
for epoch in tqdm(range(num_epochs), desc="training"):
    for inputs, labels in train_loader:  # Iterate through your training data
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

    # Print the loss for this epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Training Evaluation
correct = 0
total = 0

# Set the model to evaluation mode (disables dropout and batch normalization)
model.eval()

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="iter thru test data"):  # Iterate through your testing data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')


# Saves and loads
torch.save(model.state_dict(), 'model.pth')  # Save the model
#model.load_state_dict(torch.load('model.pth'))  # Load the model


# predictions on new data
#with torch.no_grad():
#    outputs = model(new_data)


#raise CodeNotWrittenError
