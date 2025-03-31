#!/bin/bash
#SBATCH -J Make_Syyblue_Cache  # job name
#SBATCH --time=03-00:00:00     # requested time (DD-HH:MM:SS)
#SBATCH -p batch               # running in batch jobs
#SBATCH -N 1                   # 1 node
#SBATCH -n 4                   # 8 tasks total (default 1 CPU core per task) = # of cores
#SBATCH --mem=32g              # requesting 10GB of RAM total

# Submit this job with the following command from command line interface:
# `sbatch create_cache_sjob.sh`
# You can also find batch job examples on the cluster for MALAB, R, and Python jobs at:
# /cluster/tufts/hpc/tools/slurm_scripts
# To check your active jobs in the queue:
# $ squeue -u your_utln
# or
# $ squeue -u $USER
# To cancel a specific job:
# $ scancel JOBID
# To check details of your active jobs (running or pending):
# $ scontrol show jobid -dd JOBID
# Finished Job
# Querying finished jobs helps users make better decisions on requesting resources for future jobs.
# Display job CPU and memory usage:
# $ seff JOBID

# Display job detailed accounting data:
#SBATCH --output=cacheJob.%j.%N.out #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=cacheJob.%j.%N.err #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL        # email about everything
#SBATCH --mail-user=simon.webber@tufts.edu

#[commands_you_would_like_to_exe_on_the_compute_nodes]
module load singularity
singularity exec --no-mount "hostfs" --bind /cluster/tufts/capstone25skyblue/swebbe01/sky_blue:/skyblue /cluster/tufts/capstone25skyblue/skyblue_images_hpc-gpu.sif sh -c "cd /skyblue && python /skyblue/src/createCache.py"

#Current job id 12855436