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

def cylon_sort(data=None):
    StopWatch.start(f"sort_total_{data['host']}_{data['rows']}_{data['it']}")

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

    for i in range(data['it']):
        env.barrier()
        StopWatch.start(f"sort_{i}_{data['host']}_{data['rows']}_{data['it']}")
        t1 = time.time()
        df3 = df1.sort_values(by=[0], env=env)
        env.barrier()
        t2 = time.time()
        t = (t2 - t1)
        sum_t = comm.reduce(t)
        tot_l = comm.reduce(len(df3))

        if env.rank == 0:
            avg_t = sum_t / env.world_size
            print("### ", data['scaling'], env.world_size, num_rows, max_val, i, avg_t, tot_l)
            StopWatch.stop(f"sort_{i}_{data['host']}_{data['rows']}_{data['it']}")

    StopWatch.stop(f"sort_total_{data['host']}_{data['rows']}_{data['it']}")

    if env.rank == 0:
        StopWatch.benchmark(tag=str(data))

    env.finalize()

    return df3

def df_to_tensor(df):
    print(df)
    pd_df = df.to_pandas()
    print(pd_df)
    # TODO: use huggingface data loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="weak scaling")
    parser.add_argument('-n', dest='rows', type=int, required=True)
    parser.add_argument('-i', dest='it', type=int, default=10)
    parser.add_argument('-u', dest='unique', type=float, default=0.9, help="unique factor")
    parser.add_argument('-s', dest='scaling', type=str, default='s', choices=['s', 'w'],
                        help="s=strong w=weak")

    args = vars(parser.parse_args())
    args['host'] = "rivanna"
    args['task'] = 1
    srtd = cylon_sort(args)
    df_to_tensor(srtd)
