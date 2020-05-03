#!/bin/bash

function run_tests() {
    step_size="$1"
    chunk_size="$2"
    num_ranks="$3"
    block_size="$4"
    output_dir="$5"

    iterations=100
    num_gpus=1

    ranks_per_node="$num_ranks"
    num_nodes=1

    if [ "$num_ranks" -eq 64 ]
    then
        ranks_per_node=32
        num_nodes=2
    fi

    time_dir="$output_dir/${step_size}_${iterations}_${num_ranks}_${block_size}"

    mkdir -p "$time_dir"

    slurm_script="$(dirname "$0")/slurmTest.sh"

    sbatch \
        --nodes="$num_nodes" \
        --ntasks-per-node="$ranks_per_node" \
        --gres=gpu:"$num_gpus" \
        --block-size="$block_size" \
        --step-size="$step_size" \
        --iterations="$iterations" \
        --time-dir="$time_dir" \
        --output="$time_dir/"log_%j.out
        "$slurm_script" -c"$chunk_size"
}

# BEGIN SCRIPT EXECUTION

output_dir=Output

# Clean up after previous tests if necessary.
# Prompt user to make sure data is not lost.
[ -d "$output"] && rm -rI "$output_dir"

# Step sizes used for weak scaling tests only.
step_sizes=(0.032 0.016 0.008 0.004 0.002 0.001 0.0005)
num_ranks_total=(1 2 4 8 16 32 64)
chunk_sizes=(64 32 16 8 2 4 1)
block_sizes=(16 32 64 128 256 512 1024)

# Index i is the same for all parallel arrays above.
for i in "${!num_ranks_total[@]}";
do
    chunks="${chunk_sizes[$i]}"
    num_ranks="${num_ranks_total[$i]}"
    block_size="${block_sizes[$i]}"

    # Strong scaling test 1.
    step_size=0.001
    run_tests "$step_size" "$chunks" "$num_ranks" "$block_sizes" "$output_dir"

    # Strong scaling test 2.
    step_size=0.0001
    run_tests "$step_size" "$chunks" "$num_ranks" "$block_sizes" "$output_dir"

    # Weak scaling test.
    step_size="${step_sizes[$i]}"
    run_tests "$step_size" "$chunks" "$num_ranks" "$block_sizes" "$output_dir"
done
