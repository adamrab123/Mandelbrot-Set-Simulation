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

#ifdef PARALLEL
/**
 * @brief Starts the kernel with for each rank and assigns each rank a portion of the grid
 * 
 * @param args the command line arguements
 */
void compute_mandelbrot_parallel(const Args *args) {
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

    // Start inclusive, end exclusive.
    long start_row, end_row;
    // num_jobs, num_workers, worker_index, out start, out end.
    _get_slice(bitmap->num_rows, num_ranks, my_rank, &start_row, &end_row);
    // Among different processes, this value will either be n or n+1.
    long all_grid_rows = end_row - start_row;
    long grid_cols = bitmap_cols;

    for (int i = 0; i < args->writes_per_process; i++) {
        // Ensure that every process makes the same number of writes (required by collective IO),
        // Even if they have slightly different numbers of rows to write.
        // Args has been validated to make sure that the number of writes is <= number of rows.

        // num_jobs, num_workers, worker_index, out start, out end.
        _get_slice(all_grid_rows, args->writes_per_process, i, &start_row, &end_row);
        rows_to_write = end_row - start_row;

        // This method has error handling if the cuda malloc call fails.
        Rgb *grid = (Rgb *)cuda_malloc(rows_to_write * grid_cols * sizeof(Rgb));

        launch_mandelbrot_kernel(grid, start_row, rows_to_write, grid_cols, args);
        MPI_Barrier(MPI_COMM_WORLD);

        int result = Bitmap_write_rows(bitmap, grid, start_row, rows_to_write);

        if (result != 0) {
            fprintf(stderr, "Error writing to file %s\n", args->output_file);
            exit(EXIT_FAILURE);
        }

        cuda_free(grid);
    }

    result = Bitmap_free(bitmap);

    if (result != 0) {
        fprintf(stderr, "Error closing file %s\n", args->output_file);
        exit(EXIT_FAILURE);
    }
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
        long bytes_needed = num_cols * sizeof(Rgb);
        Rgb *row_grid = malloc(num_cols * sizeof(Rgb));

        if (row_grid == NULL) {
            fprintf(stderr, "Unable to allocate %ld bytes of heap memory, exiting.\n", bytes_needed);
            exit(EXIT_FAILURE);
        }

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

            row_grid[col] = color;
            free(point);
        }

        int result = Bitmap_write_rows(bitmap, row_grid, row, 1);

        if (result != 0) {
            fprintf(stderr, "Error writing to file %s\n", args->output_file);
            exit(EXIT_FAILURE);
        }

        free(row_grid);
    }

    int result = Bitmap_free(bitmap);

    if (result != 0) {
        fprintf(stderr, "Error closing file %s\n", args->output_file);
        exit(EXIT_FAILURE);
    }
}
#endif

// void _make_image_chunks(Rgb *pixels, long pixels_rows, long pixels_cols, long num_images) {
//     // naming convention for image chunks is args->output_file_<row>_<column>.bmp

//     long subimage_rows = pixels_rows;


//     Bitmap *bitmap = Bitmap_init(subimage_rows, subimage_cols, "foobar name");
// }

// Determine which part of total this process should calculate.
// start inclusive, end exclusive.
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
