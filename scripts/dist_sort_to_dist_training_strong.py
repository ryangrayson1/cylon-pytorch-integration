import time
import argparse
import pandas as pd
from numpy.random import default_rng
from mpi4py import MPI
from pycylon.frame import CylonEnv, DataFrame
from pycylon.net import MPIConfig
from datasets import Dataset
import torch

'''
suppose we have a simple training dataset, which is a list of tuples of (input, target) pairs.
input can be any positive float up to 1e9, and target is a digit from 0 to 9.

for this experiment we'll first run a distributed sort on the dataset using cylon,
then convert the result to torch tensors, attempting to perform a zero-copy transformation
then we'll run a simple training loop on the tensors, using pytorch's distributed training mechanism
'''


'''
creates a training dataset (as a Cylon DataFrame)
with the given number of rows, and 2 columns (input, target)
input is a random positive float up to 1e9
target is a random digit from 0 to 9
'''
def create_training_dataset(rows, seed) -> DataFrame:
    rng = default_rng(seed)
    inputs = rng.floats(0, 1e9, rows)
    targets = rng.integers(0, 10, rows)
    pd_df = pd.DataFrame({'input': inputs, 'target': targets})
    return DataFrame(pd_df)


def df_to_tensor(df) -> torch.Tensor:
    pd_df = df.to_pandas()
    # print("pandas:")
    # print(pd_df)

    hf_dataset = Dataset.from_pandas(pd_df)
    # print("huggingface:")
    # print(hf_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = hf_dataset.with_format("torch", device=device)
    # print("torch tensor:")
    # print(tensor)

    return tensor


# def train(rank, tensor):
#     # TODO
#     torch.manual_seed(1234)
#     model = torch.Net()
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

#     num_batches = ceil(len(train_set.dataset) / float(bsz))
#     for epoch in range(10):
#         epoch_loss = 0.0
#         for data, target in train_set:
#             optimizer.zero_grad()
#             output = model(data)
#             loss = F.nll_loss(output, target)
#             epoch_loss += loss.item()
#             loss.backward()
#             average_gradients(model)
#             optimizer.step()
#         print('Rank ', dist.get_rank(), ', epoch ',
#             epoch, ': ', epoch_loss / num_batches)


def run(data):
    start_time = time.time_ns()

    comm = MPI.COMM_WORLD
    config = MPIConfig(comm)
    env = CylonEnv(config=config, distributed=True)

    if env.rank == 0:
        print("Starting distributed sort & train with", data['rows'], "rows and", env.world_size, "nodes")

    # strong scaling
    rows_per_node = data['rows'] / env.world_size

    raw_data = create_training_dataset(rows_per_node, data['seed'])
    sorted_data = raw_data.sort_values(by=['input'], env=env)

    # tensor = df_to_tensor(sorted_data)

    # train(env, tensor)

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
