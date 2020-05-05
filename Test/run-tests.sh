#!/bin/bash

function run_tests() {
    step_size="$1"
    chunk_size="$2"
    num_ranks="$3"
    block_size="$4"
    local output_dir="$5"

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

    slurm_script="$(dirname $0)/slurmTest.sh"

    # Executable path is passed in to the slurm script.
    # Change this path if this is not where your executable is.
    exe="$(dirname $0)/../Build/mandelbrot"

    sbatch \
        --nodes="$num_nodes" \
        --ntasks-per-node="$ranks_per_node" \
        --gres=gpu:"$num_gpus" \
        --output="$time_dir/"log_%j.out \
        "$slurm_script" \
            "$exe" \
            --chunks="$chunk_size" \
            --block-size="$block_size" \
            --step-size="$step_size" \
            --iterations="$iterations" \
            --time-dir="$time_dir" \
            --delete-output
}

# BEGIN SCRIPT EXECUTION

output_dir=Output

# Clean up after previous tests if necessary.
# Prompt user to make sure data is not lost.
[ -d "$output_dir" ] && rm -rI "$output_dir"

# Step sizes used for weak scaling tests only.
step_sizes=(0.05 0.0355 0.0252 0.0179 0.0127 0.0090 0.0064)
num_ranks_total=(1 2 4 8 16 32 64)
chunk_sizes=(64 32 16 8 2 4 1)
block_sizes=(16 32 64 128 256 512 1024)

# Index i is the same for all parallel arrays above.
for i in "${!num_ranks_total[@]}";
do
    # chunks="${chunk_sizes[$i]}"
    chunks=1
    num_ranks="${num_ranks_total[$i]}"
    block_size="${block_sizes[$i]}"

    # # Strong scaling test 1.
    # step_size=0.001
    # run_tests "$step_size" "$chunks" "$num_ranks" "$block_size" "$output_dir"/StrongLow

    # # Strong scaling test 2.
    # step_size=0.0001
    # run_tests "$step_size" "$chunks" "$num_ranks" "$block_size" "$output_dir"/StrongHigh

    # Weak scaling test.
    step_size="${step_sizes[$i]}"
    run_tests "$step_size" "$chunks" "$num_ranks" "$block_size" "$output_dir"/Weak
done
