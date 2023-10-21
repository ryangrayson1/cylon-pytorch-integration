import os
import numpy as np
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
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


# use allreduce to synthesize training results across different nodes
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def run(rank, size, rows):

    print("hello from process", rank, "of", size)

    inputs, labels = create_training_dataset(rows, seed=rank)

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

    num_epochs = 5
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))  # Add an extra dimension
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()

        print("rank", rank, "epoch:", epoch, "| loss:", epoch_loss)

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


def init_process(rank, size, rows, fn, backend='mpi'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="strong scaling")
    parser.add_argument('-n', dest='rows', type=int, required=True)
    parser.add_argument('--ntasks', dest='ntasks', type=int, default=10000)
    args = vars(parser.parse_args())

    mp.set_start_method("spawn")

    rows_per_process = args['rows'] // args['ntasks']
    processes = []
    for rank in range(args['ntasks']):
        p = mp.Process(target=init_process, args=(rank, args['ntasks'], rows_per_process, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
