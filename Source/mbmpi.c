#include <mbmpi.h>

#include "args.h"

extern void cuda_init(const Arguments *args, int myranks, int numranks);
extern bool launch_mandlebrot_kernel(int num_iterations, ushort block_size);

/**
 * @brief Starts the kernel with for each rank and assigns each rank a portion of the grid
 * 
 * @param argc number of arguements
 * @param argv the arguements
 * @param dim_width the size of the x-axis of the grid
 * @param dim_height the size of the y-axis of the grid
 * @param step the step size increments of the grid
 * @param num_iterations number of iterations per point
 * @param block_size number of threads per block
 */
void start_MPI(const Arguments *args){
    int myrank, numranks;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);

    cuda_init(args, myrank, numranks);

    launch_mandlebrot_kernel(num_iterations, block_size);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}