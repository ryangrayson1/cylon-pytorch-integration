import time
import argparse
import pandas as pd
import numpy as np
import torch
import datasets
from mpi4py import MPI
from pycylon.frame import CylonEnv, DataFrame
from pycylon.net import MPIConfig


'''
suppose we have a simple training dataset, which is a list of tuples of (input, target) pairs.
input can be any float between 0 and 1, and target is a digit from 0 to 9.

for this experiment we'll first run a distributed sort on the dataset using cylon,
then convert the result to torch tensors, attempting to perform a zero-copy transformation
then we'll run a simple training loop on the tensors, using pytorch's distributed training mechanism
'''


'''
define a very simple neural network. the model details are not too important for this experiment.
'''
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(1, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)


'''
creates a training dataset (as a Cylon DataFrame)
with the given number of rows, and 2 columns (input, target)
input is a random float between 0 and 1
target is a digit from 0 to 9, calculated as (input * 797 // 2) % 10
this simple pattern will allow us to see how well the model is working
'''
def create_training_dataset(rows, seed) -> DataFrame:
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(size=rows)
    labels = (inputs * 797 // 2) % 10
    pd_df = pd.DataFrame({'input': inputs, 'label': labels})
    return DataFrame(pd_df)


'''
converts a Cylon DataFrame to a torch Tensor
should be zero-copy, leveraging the arrow format
'''
def df_to_tensors(df: DataFrame) -> torch.Tensor:
    # # # TODO - check if this is actually zero-copy
    # pd_df = df.to_pandas()

    # hf_dataset = datasets.Dataset.from_pandas(pd_df, split="train", features={'input': datasets.Value('float32'), 'label': datasets.ClassLabel(num_classes=10, names=[str(i) for i in range(10)])})

    # device = torch.device("cpu")
    # tensors = hf_dataset.with_format("torch", device=device)

    # print(type(tensors))
    # print(tensors)
    # print(tensors[0])

    # return tensors


    pd_df = df.to_pandas()

    terrible = []

    for row in pd_df.itertuples():
        terrible.append((torch.tensor([row.input], dtype=torch.float64), torch.tensor([row.label], dtype=torch.long)))

    return terrible



def average_gradients(env, model):
    pass
    # for param in model.parameters():
    #     dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
    #     param.grad.data /= env.world_size


def train(env, train_data):

    model = SimpleNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    loss_fn = torch.nn.CrossEntropyLoss()


    for epoch in range(10):
        epoch_loss = 0.0
        for input, label in train_data:
            if env.rank == 0:
                print("rank 0  input:", input, ", label:", label)
            optimizer.zero_grad()
            output = model(input.unsqueeze(0))
            loss = loss_fn(output, label)
            epoch_loss += loss.item()
            loss.backward()
            # average_gradients(env, model)
            optimizer.step()
        if env.rank == 0:
            print("rank 0  epoch:", epoch, "| loss:", epoch_loss)


def run(data):
    start_time = time.time_ns()

    comm = MPI.COMM_WORLD
    config = MPIConfig(comm)
    env = CylonEnv(config=config, distributed=True)

    if env.rank == 0:
        print("Starting distributed sort & train with", data['rows'], "rows and", env.world_size, "cores")

    # strong scaling
    rows_per_core = data['rows'] // env.world_size

    raw_data = create_training_dataset(rows_per_core, env.rank)
    sorted_data = raw_data.sort_values(by=['input'], env=env)

    train_data = df_to_tensors(sorted_data)

    train(env, train_data)

    env.finalize()

    end_time = time.time_ns()
    print("Total time elapsed: ", end_time - start_time, " nanoseconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="strong scaling")
    parser.add_argument('-n', dest='rows', type=int, required=True)
    args = vars(parser.parse_args())
    args['host'] = "rivanna"
    args['task'] = 1

    run(args)
