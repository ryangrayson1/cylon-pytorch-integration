Prerequisite: first, follow documentation in target/rivanna to get Cylon set up

Modules
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

Note: update the constants at the top of cylon-experiment-setup.py with your own paths

To run experiment
```bash
make clean # For cleaning
make run # for running Cylon experiment
```
