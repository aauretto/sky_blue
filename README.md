# Docker

In order to launch the container for our code, you first download [Docker Desktop] (https://www.docker.com/products/docker-desktop/)
for your OS. Docker will prompt you to create an account. Unless you intend to push or make new images, for the purposes of our application, you do not need an account as we are already hosting our container on Docker Hub.

We run our code using a Docker container that has all of our dependencies in it. `Dockerfile` is the file we use
to build that container. An image located at `docker://aauretto122/skyblue_images:hpc-gpu` will have all of our dependencies in it.

To build the image from the top level of the GitHub Repository (aka skyblue) do this command (should work for Powershell or WSL. Mac
users may need to do platform-specific things).
`docker build -t <image_name>:latest .`

Our container can be used to develop in when the image is built using:
Do this command: `docker run --gpus all -it --rm -v ${PWD}:/skyblue/ <image_name>:latest`
The above command needs to be run from the skyblue directory on your computer and will sync the container files with local ones. Changes
made to files under the /skyblue directory in the container will be reflected outside of the container too.

Our code uses the default python in the container (python 3.11), which can be run using the command `python` from inside the container.

If there are build problems try running `docker system prune` -- Should never be the problem but its basically free to try.

When we run our code on the HPC, we need to use Singularity (Docker for Computing Clusters) to run our container.
We accomplish this by doing the following:

# Pushing Docker image to remote repo:
1) Login to a docker hub account: `docker login`
2) Rename docker image to whatever you want it to be named in the repo: `docker tag <local_image>:<tag> <desired_name>:<desired_tag>`
3) Push to repo: `docker push <desired_name>:<desired_tag>`

# Pulling Docker image on HPC:
1) Log into a HPC node
2) Log onto a compute node using: `srun -N1 -n8 -t0-8 -p preempt --gres=gpu:h100:1 --pty bash`
3) `module load singularity`
4) Use Singularity to pull, convert, and the docker image
   `singularity pull docker://<username>/<container_name>`
   If you just want a container that works with our code and don't need to make any other
   modifications, pull from `docker://aauretto122/skyblue_images:hpc-gpu`
5) This creates a .sif file that acts as a runnable image using the below steps:

There are two scripts on the HPC to make pulling and running code easier. They can be used like so: 
1) Log into HPC and get a terminal
2) Run `sh pull_image.sh <docker hub username>/<Container that trains model> --rm` 
   to get the latest version of the container that will train a model
3) Run `sh train_model.sh <SIF file from 2> <directory that will house model outputs>`
* NOTE: Our codebase uses the /skyblue/persistent_files folder as a point any containerized code
        can use to save files that need to exist beyond the lifetime of the container. In (3), 
        the second argument will be mounted in as `/skyblue/persistent_files`

To run code on the HPC, we are currently using a script that launches the container and mounts in a 
git repository containing our source code. This script is called `run_code` and is used like so:
`run_code sif_file repo_root [-it]`
Where sif_file is the location of the sif file you pulled, repo root is the root of a git repo with
our code in it, and the optional -it flag will make the container interactive (give you a shell to
execute commands in once it spools up). If -it is omitted the srcipt will automatically run
`python /skyblue/src/model.py` in the container.

### Provision GPU on HPC
srun -N1 -n8 -t0-8 -p preempt --gres=gpu:h100:1 --pty bash

### HPC Cleanup
`singularity cache clean --force`
or
`rm -rf ~/.singularity`
Use this if the HPC is complaining about disk space (check quota with quota -s).

### PIREP GRID FORMAT ###
When the pireps are placed on their grid, the grid is `np.nan` everywhere else and NEG events are set to the background risk and will be later 0ed during spreading

### Satellite Data TimeStamp Notes

The AWS S3 Bucket from which GOES-16 ABI-L2 CMPIC Sattelite Images are pulled is not consistent with the TimeStamps matching when the satellite images were actually taken,
especially on earlier years such as 2017. The team believes that this is due to satellite images being post-hoc uploaded to the S3 Bucket. Therefore, the timestamps from which we expect to pull satellite data are generated using dt.datetime independently in our model pipeline to avoid any such inconsistency issues.