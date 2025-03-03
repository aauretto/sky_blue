# Docker

In order to launch the container for our code, you first download [Docker Desktop] (https://www.docker.com/products/docker-desktop/)
for your OS. Docker will prompt you to create an account. Unless you intend to push or make new images, for the purposes of our application, you do not need an account as we are already hosting our container on Docker Hub.

To build our image from the top level of the GitHub Repository (aka skyblue) do this command Powershell if you are not a Mac user.
`docker build -t skyblue_dev:latest .`

To run the container, do this command: `docker run -it --rm -v ${PWD}:/skyblue/ skyblue_dev:latest`
The above command needs to be run from the skyblue directory on your computer and will sync the container files with local ones

To run, do python3.12 <fileName> to use the version of Python supported by the container.

If there are build problems try running `docker system prune` -- Should never be the problem but its basically free to try.

When we run our code on the HPC, we need to push to Docker Hub so that we can then pull the latest version
on the HPC. The Tufts HPC uses singularity instead of docker, so we need to run singularity pull docker://papillonlibre/tufts_capstone_skyblue:latest.


### PIREP GRID FORMAT ###
When the pireps are placed on their grid, the grid is np.nan everywhere else and NEG events are set to the background risk
and will be later 0ed during spreading