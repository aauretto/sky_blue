### Script that will launch a container that trains a model.
### FOR USE ON TUFTS HPC ONLY (Or environment that has singularity)

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

singularity shell --bind $OUTPUT_DIR:/skyblue/persistent_files $SIF_FILE