SIF_FILE=$1 
OUTPUT_DIR=$2

if [ $# -ne 2 ]; then
  echo "Usage: sh train_model sif_file output_dir"
  exit 1
fi

module load singularity

# Make sure we create folder that will hold all of our persistent files
if [ ! -d $OUTPUT_DIR ]; then
  echo $OUTPUT_DIR" not found, creating directory "$OUTPUT_DIR
  mkdir $OUTPUT_DIR
fi

# Remove old skyblue dirs
if [ -d "./skyblue" ]; then
  echo "Old skyblue dir found. Removing and copying updated files."
  rm -rf ./skyblue
fi

mkdir ./skyblue

# Mount in temp dir to grab all project files
singularity exec --bind ./skyblue:/mnt/host_skyblue $SIF_FILE sh -c "cp -r /skyblue/* /mnt/host_skyblue"

singularity exec --nv --no-mount "hostfs" --bind ./skyblue:/skyblue,$OUTPUT_DIR:/skyblue/persistent_files $SIF_FILE sh -c "cd /skyblue && python /skyblue/src/model.py"
