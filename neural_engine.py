import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


input_size: int  # declare this!!!
hidden_size: int  # declare this!!!
output_size: int  # declare this!!!

model = SimpleNN(input_size, hidden_size, output_size)

# I have no idea what this does
criterion = nn.CrossEntropyLoss()  # Example loss function for classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Example optimizer


# Data Sets
train_dataset = torchvision.datasets.MNIST(root="./data",
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torch.datasets.MNIST(root="./data",
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)

# Data loaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)  # I have no clue what shuffle does here. I am copying verbatim off the tutorial

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)


# train the nn
num_epochs = 10  # Number of training epochs. Change this!!!!
for epoch in range(num_epochs):
    for inputs, labels in dataloader:  # Iterate through your training data
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
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')


# Saves and loads
torch.save(model.state_dict(), 'model.pth')  # Save the model
model.load_state_dict(torch.load('model.pth'))  # Load the model


# predictions on new data
with torch.no_grad():
    outputs = model(new_data)


raise CodeNotWrittenError
