#!/bin/bash
#SBATCH --job-name=h-n=02-t=37-e=e0
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=37
#SBATCH --time=15:00
#SBATCH --output=out-02-37-%j.log
#SBATCH --error=out-02-37-%j.err
#SBATCH --partition=bii
#SBATCH -A bii_dsc_community

echo "..............................................................."
module load gcc/9.2.0 openmpi/3.1.6 cmake/3.23.3 python/3.7.7
echo "..............................................................."
source /project/bii_dsc_community/rtg5xkh/ENV377/bin/activate
echo "..............................................................."
BUILD_PATH=/project/bii_dsc_community/rtg5xkh/cylon/build
echo "..............................................................."
export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH
echo "..............................................................."
which python gcc g++
echo "..............................................................."
lscpu
echo "..............................................................."
time srun --exact --nodes 2 --ntasks 74  python cylon_sort_to_tensor.py -n 1000000
echo "..............................................................."