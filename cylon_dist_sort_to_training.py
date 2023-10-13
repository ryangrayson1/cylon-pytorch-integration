import time
import argparse

import pandas as pd
from mpi4py import MPI
from numpy.random import default_rng
from pycylon.frame import CylonEnv, DataFrame
from pycylon.net import MPIConfig
from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.common.dotdict import dotdict
from cloudmesh.common.Shell import Shell
from datasets import Dataset
import torch

def cylon_sort(data=None):
    StopWatch.start(f"sort_total_{data['host']}_{data['rows']}")

    comm = MPI.COMM_WORLD

    config = MPIConfig(comm)
    env = CylonEnv(config=config, distributed=True)

    u = data['unique']

    if data['scaling'] == 'w':  # weak
        num_rows = data['rows']
        max_val = num_rows * env.world_size
    else:  # 's' strong
        max_val = data['rows']
        num_rows = int(data['rows'] / env.world_size)

    rng = default_rng(seed=env.rank)
    data1 = rng.integers(0, int(max_val * u), size=(num_rows, 2))

    df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))

    if env.rank == 0:
        print("Task# ", data['task'])

    env.barrier()
    StopWatch.start(f"sort_{data['host']}_{data['rows']}")
    t1 = time.time()
    df3 = df1.sort_values(by=[0], env=env)
    env.barrier()
    t2 = time.time()
    t = (t2 - t1)
    sum_t = comm.reduce(t)
    tot_l = comm.reduce(len(df3))

    if env.rank == 0:
        print("### ", data['scaling'], env.world_size, num_rows, max_val, sum_t, tot_l)
        StopWatch.stop(f"sort_{data['host']}_{data['rows']}")

    StopWatch.stop(f"sort_total_{data['host']}_{data['rows']}")

    if env.rank == 0:
        StopWatch.benchmark(tag=str(data))

    # env.finalize()

    return env, env.rank, df3

def df_to_tensor(df):
    print("Cylon:")
    print(df)
    pd_df = df.to_pandas()
    print("pandas:")
    print(pd_df)

    huggingface = Dataset.from_pandas(pd_df)
    print("huggingface:")
    print(huggingface)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = huggingface.with_format("torch", device=device)
    print("torch tensor:")
    print(tensor)

    return tensor


def train(rank, tensor):
    # TODO
    torch.manual_seed(1234)
    model = torch.Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="strong scaling")
    parser.add_argument('-n', dest='rows', type=int, required=True)
    parser.add_argument('-i', dest='it', type=int, default=2)
    parser.add_argument('-u', dest='unique', type=float, default=0.9, help="unique factor")
    parser.add_argument('-s', dest='scaling', type=str, default='s', choices=['s', 'w'], help="s=strong w=weak")

    args = vars(parser.parse_args())
    args['host'] = "rivanna"
    args['task'] = 1

    env, rank, srtd_df = cylon_sort(args)

    tensor = df_to_tensor(srtd_df)

    train(rank, tensor)

    env.finalize()
