# Docker

In order to launch the container for our code, you first download [Docker Desktop] (https://www.docker.com/products/docker-desktop/)
for your OS. Docker will prompt you to create an account. Unless you intend to push or make new images, for the purposes of our application, you do not need an account as we are already hosting our container on Docker Hub.

To build our image from the top level of the GitHub Repository (aka skyblue) do this command
`docker build -t skyblue:latest .`

To run the container, do this command from a Linux shell: `docker run -v ${PWD}"/src/plots":/app/src/plots skyblue:latest`

Note: whenever you make any change to the codebase and want to run the container, you will need to rebuild the image
first before you will be able to test your changes.

If there build problems try running `docker system prune`

When we run our code on the HPC, we need to push to Docker Hub so that we can then pull the latest version
on the HPC. The Tufts HPC uses singularity instead of docker, so we need to run singularity pull docker://papillonlibre/tufts_capstone_skyblue:latest.