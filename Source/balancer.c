#include <stdio.h>
#include <stdlib.h>

#include "balancer.h"
#include "args.h"
#include "bitmap.h"
#include "mandelbrot.h"

#ifdef PARALLEL
#include <mpi.h>
#include "kernel.h"
#endif

void _get_slice(long num_jobs, long num_workers, long worker_index, long *start, long *end);
void _compute_and_write_rows_serial(Bitmap *bitmap, const Args *args, long start_row, long num_rows, long num_cols);

#ifdef PARALLEL
/**
 * @brief Starts the kernel with for each rank and assigns each rank a portion of the grid
 * 
 * @param args the command line arguements
 */
long compute_mandelbrot_parallel(const Args *args) {
    // Dimensions of the whole bitmap.
    long bitmap_rows, bitmap_cols;
    Args_get_bitmap_dims(args, &bitmap_rows, &bitmap_cols);

    Bitmap *bitmap = Bitmap_init(bitmap_rows, bitmap_cols, args->output_file);
    if (bitmap == NULL) {
        fprintf(stderr, "Error opening file %s\n", args->output_file);
        exit(EXIT_FAILURE);
    }

    int my_rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Initialize the CUDA instance for this rank
    cuda_init(my_rank);

    // Start inclusive, end exclusive.
    long bitmap_start_row, bitmap_end_row;
    // num_jobs, num_workers, worker_index, out start, out end.
    _get_slice(bitmap->num_rows, num_ranks, my_rank, &bitmap_start_row, &bitmap_end_row);
    // Among different processes, this value will either be n or n+1.
    long all_grid_rows = bitmap_end_row - bitmap_start_row;
    long grid_cols = bitmap_cols;

    for (int i = 0; i < args->chunks; i++) {
        // Ensure that every process makes the same number of writes (required by collective IO),
        // Even if they have slightly different numbers of rows to write.
        // Args has been validated to make sure that the number of writes is <= number of rows.

        // num_jobs, num_workers, worker_index, out start, out end.
        long grid_start_row, grid_end_row;
        _get_slice(all_grid_rows, args->chunks, i, &grid_start_row, &grid_end_row);
        long sub_grid_rows = grid_end_row - grid_start_row;

        // This method has error handling if the cuda malloc call fails.
        Rgb *grid = (Rgb *)cuda_malloc(sub_grid_rows * grid_cols * sizeof(Rgb));

        // The row that this subgrid starts on out of the whole bitmap image.
        long absolute_start_row = bitmap_start_row + grid_start_row;

        // Start row is used to determine which point in the whole image it is calculating, not where to write it in the
        // grid.
        launch_mandelbrot_kernel(grid, absolute_start_row, sub_grid_rows, grid_cols, args);

        int result = Bitmap_write_rows(bitmap, grid, absolute_start_row, sub_grid_rows);

        if (result != 0) {
            fprintf(stderr, "Error writing to file %s\n", args->output_file);
            exit(EXIT_FAILURE);
        }

        cuda_free(grid);
    }

    long bytes_written = Bitmap_bytes_written(bitmap);
    int result = Bitmap_free(bitmap);

    if (result != 0) {
        fprintf(stderr, "Error closing file %s\n", args->output_file);
        exit(EXIT_FAILURE);
    }

    return bytes_written;
}
#endif

#ifndef PARALLEL
/**
 * @brief Starts the kernel with for each rank and assigns each rank a portion of the grid
 * 
 * @param args the command line arguements
 *
 * @return bytes_written the number of bytes written to the file
 */
long compute_mandelbrot_serial(const Args *args) {
    long num_rows, num_cols;
    Args_get_bitmap_dims(args, &num_rows, &num_cols);

    Bitmap *bitmap = Bitmap_init(num_rows, num_cols, args->output_file);
    if (bitmap == NULL) {
        fprintf(stderr, "Error opening file %s\n", args->output_file);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < args->chunks; i++) {
        long start_row, end_row;
        // num_jobs, num_workers, worker_index, out start, out end.
        _get_slice(num_rows, args->chunks, i, &start_row, &end_row);
        long rows_to_write = end_row - start_row;

        _compute_and_write_rows_serial(bitmap, args, start_row, rows_to_write, num_cols);
    }

    long bytes_written = Bitmap_bytes_written(bitmap);
    int result = Bitmap_free(bitmap);

    if (result != 0) {
        fprintf(stderr, "Error closing file %s\n", args->output_file);
        exit(EXIT_FAILURE);
    }

    return bytes_written;
}

/**
 * @brief Computes a subset of the rows and columns and writes them to the bitmap file.
 *
 * @param bitmap the image
 * @param args the command line arguements
 * @param start_row the offset starting row for this rank
 * @param num_rows the number of rows to be computed
 * @param num_cols the number of columns to be computed
 */
void _compute_and_write_rows_serial(Bitmap *bitmap, const Args *args, long start_row, long num_rows, long num_cols) {
    long bytes_needed = num_rows * num_cols * sizeof(Rgb);
    Rgb *grid = malloc(bytes_needed);

    if (grid == NULL) {
        fprintf(stderr, "Unable to allocate %ld bytes of heap memory, exiting.\n", bytes_needed);
        exit(EXIT_FAILURE);
    }

    for (long grid_row = 0; grid_row < num_rows; grid_row++) {
        for (long grid_col = 0; grid_col < num_cols; grid_col++) {

            double c_real, c_imag;
            // Offset the row so it is the row of the whole image, not the local grid.
            Args_bitmap_to_complex(args, start_row + grid_row, grid_col, &c_real, &c_imag);
            MandelbrotPoint *point = Mandelbrot_iterate(c_real, c_imag, args->iterations);

            Rgb color;

            if (point->diverged) {
                color = ColorMap_hsv_based(point->norm_iters);
            }
            else {
                color = RGB_BLACK;
            }

            grid[num_cols * grid_row + grid_col] = color;
            free(point);
        }
    }

    int result = Bitmap_write_rows(bitmap, grid, start_row, num_rows);

    if (result != 0) {
        fprintf(stderr, "Error writing to file %s\n", args->output_file);
        exit(EXIT_FAILURE);
    }

    free(grid);
}

#endif

/**
 * @brief Determine which part of total this process should calculate. The start inclusive, end exclusive.
 *
 * @param num_jobs the image
 * @param num_workers number of ranks
 * @param worker_index the current rank
 * @param start the starting row, inclusive
 * @param end the ending row, exclusive
 */
void _get_slice(long num_jobs, long num_workers, long worker_index, long *start, long *end) {
    long quotient = num_jobs / num_workers;
    long remainder = num_jobs % num_workers;

    if (worker_index < remainder) {
        // All processes of rank less than remainder pick up an extra value from the remainder.
        // Note that all processes before them have also picked up an extra value that must be taken into account in
        // their offset calculation.
        *start = (quotient + 1) * worker_index;
        *end = *start + (quotient + 1);
    }
    else {
        // This process does not get an extra value from the remainder, but must take into account those that did before
        // it when calculating its offset.
        *start = ((quotient + 1) * remainder) + (quotient * (worker_index - remainder));
        *end = *start + quotient;
    }
}
