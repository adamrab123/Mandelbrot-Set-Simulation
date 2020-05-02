#include <stdio.h>
#include <stdlib.h>

#include "balancer.h"
#include "args.h"
#include "bitmap.h"
#include "mandelbrot.h"

#ifdef PARALLEL
#include <mpi.h>

// Cuda functions that only exist when kernel.cu is linked in parallel mode.
extern void cuda_init(int my_rank);
extern void launch_mandelbrot_kernel(Rgb ** grid, long num_rows, long num_cols, long grid_offset_y, const Args *args);
#endif

void _free_grid(Rgb **grid, long num_rows);
void _get_row_range(Bitmap *bitmap, long *start_row, long *end_row);
Rgb **_allocate_grid(long num_rows, long num_cols);

#ifdef PARALLEL
/**
 * @brief Starts the kernel with for each rank and assigns each rank a portion of the grid
 * 
 * @param args the command line arguements
 */
void compute_mandelbrot_parallel(const Args *args) {
    MPI_Init(NULL, NULL);

    // Dimensions of the whole bitmap.
    long bitmap_rows, bitmap_cols;
    Args_get_bitmap_dims(args, &bitmap_rows, &bitmap_cols);

    // printf("%s: %d\n", __FILE__, __LINE__);
    Bitmap *bitmap = Bitmap_init(bitmap_rows, bitmap_cols, args->output_file);
    // printf("%s: %d\n", __FILE__, __LINE__);

    // Start inclusive, end exclusive.
    long start_row, end_row;
    _get_row_range(bitmap, &start_row, &end_row);
    long grid_rows = end_row - start_row;

    Rgb **grid = _allocate_grid(grid_rows, bitmap_cols);

    if (bitmap == NULL) {
        fprintf(stderr, "Error opening file %s\n", args->output_file);
        exit(EXIT_FAILURE);
    }

    if (bitmap == NULL) {
        fprintf(stderr, "Error opening file %s\n", args->output_file);
        exit(EXIT_FAILURE);
    }

    // printf("%s: %d\n", __FILE__, __LINE__);

    launch_mandelbrot_kernel(grid, start_row, grid_rows, bitmap_cols, args);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("%s: %d\n", __FILE__, __LINE__);

    int result = Bitmap_write_rows(bitmap, grid, start_row, grid_rows);
    printf("%s: %d\n", __FILE__, __LINE__);

    if (result != 0) {
        fprintf(stderr, "Error writing to file %s\n", args->output_file);
        exit(EXIT_FAILURE);
    }

    _free_grid(grid, grid_rows);
    printf("%s: %d\n", __FILE__, __LINE__);

    result = Bitmap_free(bitmap);

    if (result != 0) {
        fprintf(stderr, "Error closing file %s\n", args->output_file);
        exit(EXIT_FAILURE);
    }

    MPI_Finalize();
}
#endif

#ifndef PARALLEL
void compute_mandelbrot_serial(const Args *args) {
    long num_rows, num_cols;
    Args_get_bitmap_dims(args, &num_rows, &num_cols);

    Bitmap *bitmap = Bitmap_init(num_rows, num_cols, args->output_file);

    if (bitmap == NULL) {
        fprintf(stderr, "Error opening file %s\n", args->output_file);
        exit(EXIT_FAILURE);
    }

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

        int result = Bitmap_write_rows(bitmap, row_grid, row, 1);

        if (result != 0) {
            fprintf(stderr, "Error writing to file %s\n", args->output_file);
            exit(EXIT_FAILURE);
        }

        _free_grid(row_grid, 1);
    }

    int result = Bitmap_free(bitmap);

    if (result != 0) {
        fprintf(stderr, "Error closing file %s\n", args->output_file);
        exit(EXIT_FAILURE);
    }
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
void _get_row_range(Bitmap *bitmap, long *start_row, long *end_row) {
    int my_rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // If the total number of rows can be divided evenly among the ranks, each rank gets an equal number of rows.
    // If the total number of rows cannot be divided evenly among the ranks, ranks 0 to n-1 get the result of num_rows
    // // num_ranks, and rank n gets the remainder.
    if (bitmap->num_rows % num_ranks == 0 || my_rank != num_ranks - 1) {
        *start_row = (bitmap->num_rows / num_ranks) * my_rank;
        *end_row = *start_row + bitmap->num_rows / num_ranks;
    }
    else {
        long remainder = bitmap->num_rows % num_ranks;
        *start_row = bitmap->num_rows - remainder;
        *end_row = *start_row + remainder;
    }
}
#endif