#include "balancer.h"
#include "args.h"
#include "bitmap.h"
#include "mandelbrot.h"

#ifdef PARALLEL
// Cuda functions that only exist when kernel.cu is linked in parallel mode.
extern void cuda_init(int my_rank);
extern void launch_mandelbrot_kernel(Rgb ** grid, long grid_width, long grid_height, long grid_offset_y, const Args *args);
#endif

void _free_grid(Rgb **grid, long grid_width, long grid_height);
long _get_y_offset(long grid_height);
Rgb **_allocate_grid(long grid_width, long grid_height);

#ifdef PARALLEL
/**
 * @brief Starts the kernel with for each rank and assigns each rank a portion of the grid
 * 
 * @param args the command line arguements
 */
void compute_mandelbrot_parallel(const Args *args) {
    MPI_Init(NULL, NULL);

    int myrank, numranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    cuda_init(myrank);

    long grid_width, grid_height;
    Args_bitmap_dims(args, &grid_width, &grid_height);

    Rgb **grid = _allocate_grid(grid_width, grid_height);

    Bitmap *bitmap = Bitmap_init(grid_width, grid_height, args->output_file);

    long grid_offset_y = _get_y_offset(grid_height);

    launch_mandelbrot_kernel(grid, grid_width, grid_height, grid_offset_y, args);
    MPI_Barrier(MPI_COMM_WORLD);

    Bitmap_write_rows(bitmap, grid, grid_offset_y, grid_height);

    _free_grid(grid, grid_width, grid_height);

    Bitmap_free(bitmap);

    MPI_Finalize();
}
#endif

void compute_mandelbrot_serial(const Args *args) {
    long px_width, px_height;
    Args_bitmap_dims(args, &px_width, &px_height);

    Bitmap *bitmap = Bitmap_init(px_width, px_height, args->output_file);

    for (long y = 0; y < px_height; y++) {
        Rgb **grid = _allocate_grid(px_width, 1);

        for (long x = 0; x < px_width; x++) {
            double c_real, c_imag;
            Args_bitmap_to_complex(args, x, y, &c_real, &c_imag);
            MandelbrotPoint *point = Mandelbrot_iterate(c_real, c_imag, args->iterations);

            Rgb color;

            if (point->diverged) {
                color = ColorMap_hsv_based(point->norm_iters);
            }
            else {
                color = RGB_BLACK;
            }

            grid[0][x] = color;
            free(point);
        }

        Bitmap_write_rows(bitmap, grid, y, 1);
        _free_grid(grid, px_width, 1);
    }

    Bitmap_free(bitmap);
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

#ifdef PARALLEL
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
#endif