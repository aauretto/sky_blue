
TOTAL_JOBS=10

for JOB_IDX in `seq 0 $((${TOTAL_JOBS}-1))`
do
    sbatch sat_cache_arr.sjob $JOB_IDX $TOTAL_JOBS
done
