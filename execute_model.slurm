#!/bin/bash
#SBATCH -J Execute_Model       # job name
#SBATCH --time=07-00:00:00     # requested time (DD-HH:MM:SS)
#SBATCH -p preempt             # running on "mpi" partition/queue
#SBATCH -N 1                   # 1 node
#SBATCH -n 16                  # 16 tasks total (default 1 CPU core per task) = # of cores
#SBATCH --mem=64g              # requesting 64GB of RAM total
#SBATCH --gres=gpu:h100:1      # GPUs that we want: h100
# Submit this job with the following command from command line interface:
# sbatch execute_model.slurm
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
#SBATCH --output=MyJob.execute_model.%j.%N.out #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=MyJob.execute_model.%j.%N.err #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL #email optitions
#SBATCH --mail-user=aiden.auretto@tufts.edu,simon.webber@tufts.edu

#[commands_you_would_like_to_exe_on_the_compute_nodes]

# Run scripts that will copy down newest image then run it
module load singularity

/cluster/tufts/capstone25skyblue/model_executor /cluster/tufts/capstone25skyblue/skyblue_images_hpc-gpu-no-repo.sif /cluster/tufts/capstone25skyblue/aauret01/sky_blue


# Submitted batch job 13088145