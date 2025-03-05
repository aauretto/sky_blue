# Docker

In order to launch the container for our code, you first download [Docker Desktop] (https://www.docker.com/products/docker-desktop/)
for your OS. Docker will prompt you to create an account. Unless you intend to push or make new images, for the purposes of our application, you do not need an account as we are already hosting our container on Docker Hub.

There are two Dockerfiles in this repository corresponing to two containers that we use:
1) Dockerfile.dev -- The dev container that we use to develop code for the project
2) Dockerfile.hpc -- The container that will be put on the HPC and train our model

In the below notes <Dockerfile> should be replaced with Dockerfile.dev/hpc depending on what image/container you want to use

To build the image from the top level of the GitHub Repository (aka skyblue) do this command Powershell if you are not a Mac user.
`docker build -t <image_name>:latest -f ./<Dockerfile> .`

To run the dev container:
Do this command: `docker run -it --rm -v ${PWD}:/skyblue/ <image_name>:latest`
The above command needs to be run from the skyblue directory on your computer and will sync the container files with local ones

To run, do python3.12 <fileName> to use the version of Python supported by the container.

If there are build problems try running `docker system prune` -- Should never be the problem but its basically free to try.

When we run our code on the HPC, we need to push to Docker Hub so that we can then pull the latest version
on the HPC. The Tufts HPC uses singularity instead of docker, so we need to run singularity pull docker://papillonlibre/tufts_capstone_skyblue:latest.

The HPC container should be pushed to a remote repo so it can be pulled on the HPC and run there.

# Pushing Docker image to remote repo:
* Repeat this each time we wanna push to HPC
1) Login to a docker hub account: `docker login`
2) Rename docker image to whatever you want it to be named in the repo: `docker tag <local_image>:<tag> <desired_name>:<desired_tag>`
3) Push to repo: `docker push <desired_name>:<desired_tag>`

### PIREP GRID FORMAT ###
When the pireps are placed on their grid, the grid is np.nan everywhere else and NEG events are set to the background risk
and will be later 0ed during spreading