import os
import sys
from textwrap import dedent
from cloudmesh.common.util import writefile
from cloudmesh.common.util import readfile
from cloudmesh.common.util import banner
from cloudmesh.common.console import Console

counter = 0

debug = True
debug = False

partition="bii-gpu"

partition="parallel"


# (nodes, threads, rows, partition, "exclusive")
combination = [\
    # (1,4, 5000, "parallel", "exclusive"), # always pending
    (2,37, 1000000, "parallel", ""),
    # (4,37, 35000000, "parallel", ""),
    # (6,37, 35000000, "parallel", ""),
    # (8,37, 35000000, "parallel", ""),
    # (10,37, 35000000, "parallel", ""),
    # (12,37, 35000000, "parallel", ""),
    # (14,37, 35000000, "parallel", ""),
]

'''
combination = []
for nodes in range(0,50):
  for threads in range(0,37):
    combination.append((nodes+1, threads+1, "parallel", ""))
'''

total = len(combination)
jobid="-%j"
# jobid=""

f = open("submit.log", "w")
for nodes, threads, rows, partition, exclusive in combination:
  counter = counter + 1

  if exclusive == "exclusive":
      exclusive = "#SBATCH --exclusive"
      e = "e1"
  else:
      exclusive = ""
      e = "e0"

  usable_threads = nodes * threads

  '''
  cores_per_node = nodes * threads - 2

  print (cores_per_node)

  config = readfile("raptor.in.cfg")

  config = config.replace("CORES_PER_NODE", str(cores_per_node))
  config = config.replace("NO_OF_ROWS", str(rows))


  print (config)

  cfg_filename = f"raptor-{nodes}-{threads}.cfg"

  writefile(cfg_filename, config)
  '''
  banner(f"SLURM {nodes} {threads} {counter}/{total}")
  script=dedent(f"""
  #!/bin/bash
  #SBATCH --job-name=h-n={nodes:02d}-t={threads:02d}-e={e}
  #SBATCH --nodes={nodes}
  #SBATCH --ntasks-per-node={threads}
  #SBATCH --time=15:00
  #SBATCH --output=out-{nodes:02d}-{threads:02d}{jobid}.log
  #SBATCH --error=out-{nodes:02d}-{threads:02d}{jobid}.err
  #SBATCH --partition=bii
  #SBATCH -A bii_dsc_community
  {exclusive}
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
  time srun --exact --nodes {nodes} --ntasks {usable_threads}  python cylon_sort_to_tensor.py -n {rows}
  echo "..............................................................."
  """).strip()

  print (script)
  filename = f"script-{nodes:02d}-{threads:02d}.slurm"
  writefile(filename, script)


  if not debug:

    r = os.system(f"sbatch {filename}")
    total = nodes * threads
    if r == 0:
      msg = f"{counter} submitted: nodes={nodes:02d} threads={threads:02d} total={total}"
      Console.ok(msg)
    else:
      msg = f"{counter} failed: nodes={nodes:02d} threads={threads:02d} total={total}"
      Console.error(msg)
    f.writelines([msg, "\n"])
f.close()
