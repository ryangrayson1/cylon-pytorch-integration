Note: this experiment was designed for rivanna and is intended to be run on rivanna

Prerequisite: first, follow documentation in the Cylon repo in `target/rivanna` to get Cylon set up

Modules used
```
module load gcc/9.2.0 openmpi/3.1.6 cmake/3.23.3 python/3.7.7
```

Pip Installations
```
pip install cloudmesh-common -U
pip install openssl-python -U
python3 -m pip install urllib3==1.26.6
pip install torch -U
pip install datasets -U
```

Note: update the constants at the top of `experiment_setup.py` with your own paths

The capitalized variables at the top of `experiment_setup.py` should be updated to specify the experiment parameters

when specifying which test files to run, keep in mind that the files are expected to be python scripts that are expected to take certain command line arguments. the following should be pasted into your test files for the best experience

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="your description")
    parser.add_argument('-n', dest='rows', type=int, required=True)
    parser.add_argument('-i', dest='it', type=int, default=2)
    parser.add_argument('-u', dest='unique', type=float, default=0.9, help="unique factor")
    parser.add_argument('-s', dest='scaling', type=str, default='s', choices=['s', 'w'], help="s=strong w=weak")
    args = vars(parser.parse_args())
    args['host'] = "rivanna"
    args['task'] = 1
```

To run experiment
```bash
make clean # For cleaning
make run # for running the specified Cylon experiment python scripts
```
