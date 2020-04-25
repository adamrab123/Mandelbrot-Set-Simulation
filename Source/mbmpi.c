#include <mbmpi.h>

#include "args.h"
#include "kernel.h"
#include "bitmap.h"

int _get_offset(int grid_height);

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
void start_mpi(const Arguments *args) {
    MPI_Init(NULL, NULL);

    int myrank, numranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    cuda_init(myrank);

    int grid_width = (args->x_max - args->x_min) / args->step_size;
    int grid_height = (args->y_max - args->y_min) / args->step_size;

    Bitmap *bitmap = Bitmap_init(grid_width, grid_height, args->output_file, PARALLEL);

    int grid_offset_y = get_offset(grid_height);

    launch_mandelbrot_kernel(bitmap, grid_width, grid_height, grid_offset_y, args->iterations, args->block_size);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
}

/**
 * @brief Initialize variables and assign portion of grid to the current rank
 * 
 * @param dim_width the size of the x-axis of the grid
 * @param dim_height the size of the y-axis of the grid
 * @param step the step size increments of the grid
 * @param my_rank the current rank
 * @param num_ranks the total number of ranks
 */
int _get_offset(int grid_height) {
    int my_rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int offset = 0;
    if (grid_height % num_ranks == 0 || my_rank != num_ranks - 1) {
        offset = (grid_height / num_ranks) * my_rank;
    }
    else {
        int remainder = grid_height % num_ranks;
        offset = grid_height - remainder;
    }

    return offset;
}
