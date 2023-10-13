import os
from datetime import datetime
from textwrap import dedent
from cloudmesh.common.util import writefile
from cloudmesh.common.util import readfile
from cloudmesh.common.util import banner
from cloudmesh.common.console import Console

# fill these in with your own paths
PYTHON_ENV_PATH = "/project/bii_dsc_community/rtg5xkh/ENV377"
CYLON_BUILD_PATH = "/project/bii_dsc_community/rtg5xkh/cylon/build"

# add all test files here
TEST_FILENAMES = [
	"scripts/dist_sort_to_dist_training_strong.py",
	# "scripts/cylon_dist_sort_test.py",
	# "scripts/cylon_scaling.py",
]
# or set this flag to run all files in a directory of your choice.
# if this is not empty it will be used. do not include a trailing slash.
RUN_ALL_FILES_IN_DIR = ""

# Debug mode will do the setup steps but will not submit the jobs
DEBUG = False

# set the desired partition
PARTITION = "parallel" # "bii-gpu"

# (nodes, threads, rows, partition, "exclusive")
TEST_PARAMS = [
	(2, 37, 100000, "parallel", ""),
	# (4, 37, 35000000, "parallel", ""),
	# (6, 37, 35000000, "parallel", ""),
	# (8, 37, 35000000, "parallel", ""),
	# (10, 37, 35000000, "parallel", ""),
	# (12, 37, 35000000, "parallel", ""),
	# (14, 37, 35000000, "parallel", ""),
]

'''
for nodes in range(1, 51):
	for threads in range(1, 38):
		TEST_PARAMS.append((nodes, threads, PARTITION, ""))
'''

if RUN_ALL_FILES_IN_DIR != "":
	TEST_FILENAMES.clear()
	for filename in os.listdir(RUN_ALL_FILES_IN_DIR):
		if filename.endswith(".py"):
			TEST_FILENAMES.append(RUN_ALL_FILES_IN_DIR + '/' + filename)

jobid = "%j"
timestamp = datetime.now().strftime("%H:%M:%S")
num_tests = len(TEST_PARAMS)
counter = 0
submit_file = open("submit.log", "w")
for nodes, threads, rows, partition, exclusive in TEST_PARAMS:
	counter += 1

	if exclusive == "exclusive":
		exclusive = "#SBATCH --exclusive"
		e = "e1"
	else:
		exclusive = ""
		e = "e0"

	usable_threads = nodes * threads

	for filename in TEST_FILENAMES:
		filename_no_path = filename.split("/")[-1]
		banner(f"SLURM {nodes} {threads} {counter}/{num_tests}")
		script=dedent(f"""
		#!/bin/bash
		#SBATCH --job-name=h-n={nodes:02d}-t={threads:02d}-e={e}
		#SBATCH --nodes={nodes}
		#SBATCH --ntasks-per-node={threads}
		#SBATCH --time=15:00
		#SBATCH --output={timestamp}-{filename_no_path}-{nodes:02d}-{threads:02d}-{jobid}.log
		#SBATCH --error={timestamp}-{filename_no_path}-{nodes:02d}-{threads:02d}-{jobid}.err
		#SBATCH --partition=parallel
		#SBATCH -A bii_dsc_community
		{exclusive}
		echo "..............................................................."
		module load gcc/9.2.0 openmpi/3.1.6 cmake/3.23.3 python/3.7.7
		echo "..............................................................."
		source {PYTHON_ENV_PATH}/bin/activate
		pip install datasets
		pip install torch
		echo "..............................................................."
		BUILD_PATH={CYLON_BUILD_PATH}
		echo "..............................................................."
		export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH
		echo "..............................................................."
		which python gcc g++
		echo "..............................................................."
		lscpu
		echo "..............................................................."
		time srun --exact --nodes {nodes} --ntasks {usable_threads}  python {filename} -n {rows}
		echo "..............................................................."
		""").strip()

		print(script)
		slurm_file = f"script-{nodes:02d}-{threads:02d}.slurm"
		writefile(slurm_file, script)

		if not DEBUG:
			r = os.system(f"sbatch {slurm_file}")
			total = nodes * threads
			if r == 0:
				msg = f"{counter} submitted: nodes={nodes:02d} threads={threads:02d} total={total}"
				Console.ok(msg)
			else:
				msg = f"{counter} failed: nodes={nodes:02d} threads={threads:02d} total={total}"
				Console.error(msg)
			submit_file.writelines([msg, "\n"])

submit_file.close()
