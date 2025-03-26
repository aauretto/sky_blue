#!/bin/bash
#SBATCH -J Train_SkyBlue_Model # job name
#SBATCH --time=00-01:00:00     # requested time (DD-HH:MM:SS)
#SBATCH -p preempt             # running on "mpi" partition/queue
#SBATCH -N 1                   # 1 node
#SBATCH -n 8                   # 8 tasks total (default 1 CPU core per task) = # of cores
#SBATCH --mem=10g              # requesting 10GB of RAM total
#SBATCH --gres=gpu:h100:1      # GPUs that we want: h100
# Submit this job with the following command from command line interface:
# `sbatch train_model_sjob.sh`
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
#SBATCH --output=MyJob.%j.%N.out #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=MyJob.%j.%N.err #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL #email optitions
#SBATCH --mail-user=Your_Tufts_Email@tufts.edu

#[commands_you_would_like_to_exe_on_the_compute_nodes]

TIMESTAMP="$(date +"%m_%d_%H_%M")"
OUT_DIR="./"$TIMESTAMP"_output"

echo "Making log entry for timestamp: "$TIMESTAMP
mkdir ./logs/$TIMESTAMP

echo "Output dir for this run will be: "$OUT_DIR

# Run scripts that will copy down newest image then run it
echo "Pulling latest image from dockerhub"
sh pull_image.sh papillonlibre/skyblue_hpc --rm > /logs/$TIMESTAMP/pull_image.log 

echo "Running container that trains model..."
sh train_model.sh skyblue_hpc_latest.sif $OUT_DIR > /logs/$TIMESTAMP/train_model.log
