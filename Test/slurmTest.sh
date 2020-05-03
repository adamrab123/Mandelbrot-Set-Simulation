#!/bin/bash -x
#SBATCH --time=30
#SBATCH --partition=dcs

if [ "$SLURM_NPROCS" ]
then
    if [ ! "$SLURM_NTASKS_PER_NODE" ]
    then
    SLURM_NTASKS_PER_NODE=1
    fi
    SLURM_NPROCS=‘expr $SLURM_JOB_NUM_NODES \* $SLURM_NTASKS_PER_NODE‘
else
    if [ ! "$SLURM_NTASKS_PER_NODE" ]
    then
    SLURM_NTASKS_PER_NODE=‘expr $SLURM_NPROCS / $SLURM_JOB_NUM_NODES‘
    fi
fi

srun hostname -s | sort -u > /tmp/hosts.$SLURM_JOB_ID
awk "{ print \$0 \"-ib slots=$SLURM_NTASKS_PER_NODE\"; }" /tmp/hosts.$SLURM_JOB_ID>/tmp/tmp.$SLURM_JOB_ID
mv /tmp/tmp.$SLURM_JOB_ID /tmp/hosts.$SLURM_JOB_ID

module load gcc/7.4.0/1
module load spectrum-mpi
module load cuda

time mpirun -hostfile /tmp/hosts.$SLURM_JOB_ID -np $SLURM_NPROCS "$HOME/scratch/Mandelbrot-Set-Simulation/Build/mandelbrot" -d "$@"

rm /tmp/hosts.$SLURM_JOB_ID
