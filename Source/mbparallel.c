#include "mbparallel.h"
#include "args.h"
#include "bitmap.h"

// Cuda functions
extern void cuda_init(int my_rank);
extern void launch_mandelbrot_kernel(Rgb ** grid, long grid_width, long grid_height, long grid_offset_y, const Args *args);

void _free_grid(Rgb **grid, long grid_width, long grid_height);
long _get_y_offset(long grid_height);
Rgb **_allocate_grid(long grid_width, long grid_height);

/**
 * @brief Starts the kernel with for each rank and assigns each rank a portion of the grid
 * 
 * @param args the command line arguements
 */
void compute_mandelbrot_parallel(const Args *args) {
    MPI_Init(NULL, NULL);

    printf("%s, %d\n", __FILE__, __LINE__);

    int myrank, numranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    cuda_init(myrank);
    printf("%s, %d\n", __FILE__, __LINE__);

    long grid_width, grid_height;
    Args_bitmap_dims(args, &grid_width, &grid_height);
    printf("%s, %d\n", __FILE__, __LINE__);

    Rgb **grid = _allocate_grid(grid_width, grid_height);
    printf("%s, %d\n", __FILE__, __LINE__);

    Bitmap *bitmap = Bitmap_init(grid_width, grid_height, args->output_file);
    printf("%s, %d\n", __FILE__, __LINE__);

    long grid_offset_y = _get_y_offset(grid_height);
    printf("%s, %d\n", __FILE__, __LINE__);

    launch_mandelbrot_kernel(grid, grid_width, grid_height, grid_offset_y, args);
    printf("%s, %d\n", __FILE__, __LINE__);
    MPI_Barrier(MPI_COMM_WORLD);

    Bitmap_write_rows(bitmap, grid, grid_offset_y, grid_height);
    printf("%s, %d\n", __FILE__, __LINE__);

    _free_grid(grid, grid_width, grid_height);

    Bitmap_free(bitmap);
    printf("%s, %d\n", __FILE__, __LINE__);

    MPI_Finalize();
    printf("%s, %d\n", __FILE__, __LINE__);
}

/**
 * @brief Allocates the grid of size @p grid_width by @p grid_height using cudaMallocManaged
 * 
 * @param grid_width the width of the grid
 * @param grid_height the height of the grid
 * 
 * @return grid
 */
Rgb **_allocate_grid(long grid_width, long grid_height){
    Rgb **grid = NULL;

    // allocate rows
    grid = calloc(grid_height, sizeof(Rgb *)); 

    // allocate columns
    for (long i = 0; i < grid_height; i++){
        grid[i] = calloc(grid_width, sizeof(Rgb)); 
    }

    return grid;
}

void _free_grid(Rgb **grid, long grid_width, long grid_height) {
    for (long i = 0; i < grid_height; i++){
        free(grid[i]);
    }

    free(grid);
}

/**
 * @brief Calculate the y index in the bitmap image grid that this process should begin calculating.
 *
 * This is purely an integer calcualtion since we are in the grid space, and does not need mpfr.
 * 
 * @param grid_height
 */
long _get_y_offset(long grid_height) {
    int my_rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    long offset = 0;
    if (grid_height % num_ranks == 0 || my_rank != num_ranks - 1) {
        offset = (grid_height / num_ranks) * my_rank;
    }
    else {
        long remainder = grid_height % num_ranks;
        offset = grid_height - remainder;
    }

    return offset;
}
