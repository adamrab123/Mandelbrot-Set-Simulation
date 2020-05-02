#include "balancer.h"
#include "args.h"
#include "bitmap.h"
#include "mandelbrot.h"

#ifdef PARALLEL
// Cuda functions that only exist when kernel.cu is linked in parallel mode.
extern void cuda_init(int my_rank);
extern void launch_mandelbrot_kernel(Rgb ** grid, long num_rows, long num_cols, long grid_offset_y, const Args *args);
#endif

void _free_grid(Rgb **grid, long num_rows);
long _get_start_row(long num_cols);
Rgb **_allocate_grid(long num_rows, long num_cols);

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

    long num_rows, num_cols;
    Args_get_bitmap_dims(args, &num_rows, &num_cols);

    Rgb **grid = _allocate_grid(num_rows, num_cols);

    Bitmap *bitmap = Bitmap_init(num_rows, num_cols, args->output_file);

    long start_row = _get_start_row(num_cols);

    launch_mandelbrot_kernel(grid, start_row, num_rows, num_cols, args);
    MPI_Barrier(MPI_COMM_WORLD);

    Bitmap_write_rows(bitmap, grid, start_row, num_cols);

    _free_grid(grid, num_rows);

    Bitmap_free(bitmap);

    MPI_Finalize();
}
#endif

#ifndef PARALLEL
void compute_mandelbrot_serial(const Args *args) {
    long num_rows, num_cols;
    Args_get_bitmap_dims(args, &num_rows, &num_cols);

    Bitmap *bitmap = Bitmap_init(num_rows, num_cols, args->output_file);

    for (long row = 0; row < num_rows; row++) {
        // Grid will be used to contain a single row.
        Rgb **row_grid = _allocate_grid(1, num_cols);

        for (long col = 0; col < num_cols; col++) {
            double c_real, c_imag;
            Args_bitmap_to_complex(args, row, col, &c_real, &c_imag);
            MandelbrotPoint *point = Mandelbrot_iterate(c_real, c_imag, args->iterations);

            Rgb color;

            if (point->diverged) {
                color = ColorMap_hsv_based(point->norm_iters);
            }
            else {
                color = RGB_BLACK;
            }

            row_grid[0][col] = color;
            free(point);
        }

        Bitmap_write_rows(bitmap, row_grid, row, 1);
        _free_grid(row_grid, 1);
    }

    Bitmap_free(bitmap);
}
#endif

/**
 * @brief Allocates the grid of size @p num_rows by @p num_cols using cudaMallocManaged
 * 
 * @param num_rows the width of the grid
 * @param num_cols the height of the grid
 * 
 * @return grid
 */
Rgb **_allocate_grid(long num_rows, long num_cols){
    Rgb **grid = NULL;

    // allocate rows
    grid = calloc(num_rows, sizeof(Rgb *)); 

    // allocate columns
    for (long i = 0; i < num_rows; i++){
        grid[i] = calloc(num_cols, sizeof(Rgb)); 
    }

    return grid;
}

void _free_grid(Rgb **grid, long num_rows) {
    for (long i = 0; i < num_rows; i++){
        free(grid[i]);
    }

    free(grid);
}

#ifdef PARALLEL
/**
 * @brief Calculate the row in the bitmap image grid that this process should begin calculating.
 *
 * @param num_cols
 */
long _get_start_row(long num_rows) {
    int my_rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    long start_row = 0;
    if (num_rows % num_ranks == 0 || my_rank != num_ranks - 1) {
        start_row = (num_rows / num_ranks) * my_rank;
    }
    else {
        long remainder = num_rows % num_ranks;
        start_row = num_rows - remainder;
    }

    return start_row;
}
#endif