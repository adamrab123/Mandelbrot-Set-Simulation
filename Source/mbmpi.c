#include <mbmpi.h>

#include "args.h"
#include "kernel.h"
#include "bitmap.h"
#include "mandelbrot.h"

int _get_offset(int grid_height);
Rgb ** allocate_grid(int grid_width, int grid_height);

/**
 * @brief Starts the kernel with for each rank and assigns each rank a portion of the grid
 * 
 * @param args the command line arguements
 */
void start_mpi(const Arguments *args) {
    MPI_Init(NULL, NULL);

    int myrank, numranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    cuda_init(myrank);

    int grid_width = (args->x_max - args->x_min) / args->step_size;
    int grid_height = (args->y_max - args->y_min) / args->step_size;

    Rgb ** grid = allocate_grid(grid_width, grid_height);

    Bitmap *bitmap = Bitmap_init(grid_width, grid_height, args->output_file, PARALLEL);

    int grid_offset_y = _get_offset(grid_height);

    launch_mandelbrot_kernel(grid, bitmap, grid_width, grid_height, grid_offset_y, args->iterations, args->block_size);
    MPI_Barrier(MPI_COMM_WORLD);

    // write_grid(grid, grid_height, grid_offset_y)

    MPI_Finalize();
}

/**
 * @brief Allocates the grid of size @p grid_width by @p grid_height using cudaMallocManaged
 * 
 * @param grid_width the width of the grid
 * @param grid_height the height of the grid
 * 
 * @return grid
 */
// unsigned char ** allocate_grid(int grid_width, int grid_height){
Rgb ** allocate_grid(int grid_width, int grid_height){
    Rgb ** grid = NULL;

    // allocate rows
    grid = calloc(grid_width, sizeof(Rgb *)); 

    // allocate columns
    for (int i = 0; i < grid_width; i++){
        grid[i] = calloc(grid_height, sizeof(Rgb)); 
    }

    return grid;
}

/**
 * @brief Initialize variables and assign portion of grid to the current rank
 * 
 * @param grid_height
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
