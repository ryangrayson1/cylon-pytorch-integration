import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Prepare the Data
def create_training_dataset(rows, seed):
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(size=rows)
    labels = (inputs * 797 // 2) % 10
    return inputs, labels


# Define a PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, input_data, label_data):
        self.input_data = torch.tensor(input_data, dtype=torch.float32)
        self.label_data = torch.tensor(label_data, dtype=torch.long)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_sample = self.input_data[idx]
        label_sample = self.label_data[idx]
        return input_sample, label_sample

# Define a Neural Network Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def run(data):
    inputs, labels = create_training_dataset(data['rows'], 0)

    dataset = CustomDataset(inputs, labels)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)

    #  Train the Model
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 50
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))  # Add an extra dimension
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate the Model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            outputs = model(inputs.unsqueeze(1))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = (100 * correct) / total
        print("Test Accuracy: {:.2f}%".format(accuracy))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="strong scaling")
    parser.add_argument('-n', dest='rows', type=int, required=True)
    args = vars(parser.parse_args())
    args['host'] = "rivanna"
    args['task'] = 1

    run(args)
