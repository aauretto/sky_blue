# Cmd line args:
CONTAINER_REPO=$1

if [ $# -ne 1 -a $# -ne 2 ]; then
  echo "Usage: sh hpc_workflow container_repo [--rm]"
  exit 1
fi

DEST_FILE="${CONTAINER_REPO//:/_}"
DEST_FILE="${DEST_FILE##*/}.sif"

echo "Destination is: "$DEST_FILE

# Check if destinaiton file is there, if it is and we wanna overwrite, remove it and pull
if [ -f "$DEST_FILE" ]; then
  if [[ $2 == "--rm" ]]; then
    rm $DEST_FILE
  else
    echo "File $DEST_FILE already exists. Overwrite flag (--rm) not set, not pulling a new image."
    exit 0
  fi
fi  

module load singularity
singularity pull docker://$CONTAINER_REPO
