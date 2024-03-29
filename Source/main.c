#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/stat.h>

#include "args.h"
#include "balancer.h"

#ifdef PARALLEL
#include <mpi.h>
#endif

const int CLOCK_HERTZ = 512000000;
typedef unsigned long long tick;

#ifdef PARALLEL
/**
 * @brief Provided code to get the current tick that the CPU is on.
 * 
 * @return The current tick value of the CPU.
 */
static __inline__ tick getticks(void) {
    unsigned int tbl, tbu0, tbu1;

    do {
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);

    return (((unsigned long long)tbu0) << 32) | tbl;
}

/**
 * @brief Provided code to get the current tick that the CPU is on.
 * 
 * @param args the command line arguements and/or default arguements
 * @param time_secs time taken for this rank to complete execution
 * @param bytes_written number of bytes written
 * @param my_rank the current rank
 * @param num_ranks the total amount of ranks
 */
void write_yaml(const Args *args, double time_secs, long bytes_written, int my_rank, int num_ranks) {
    const int MAX_FILE_NAME_LEN = 256;

    char *yaml_file_name = calloc(MAX_FILE_NAME_LEN, sizeof(char));
    sprintf(yaml_file_name, "%s/%d.yaml", args->time_dir, my_rank);

    FILE *yaml_file = fopen(yaml_file_name, "a");
    if (yaml_file == NULL) {
        fprintf(stderr, "Failed to create file %s\n", yaml_file_name);
        exit(EXIT_FAILURE);
    }

    // Write test data for this rank to its yaml file.
    fprintf(yaml_file, "x_min: %.15f\n", args->x_min);
    fprintf(yaml_file, "x_max: %.15f\n", args->x_max);
    fprintf(yaml_file, "y_min: %.15f\n", args->y_min);
    fprintf(yaml_file, "y_max: %.15f\n", args->y_max);
    fprintf(yaml_file, "step_size: %.15f\n", args->step_size);
    fprintf(yaml_file, "iterations: %ld\n", args->iterations);
    fprintf(yaml_file, "output_file: %s\n", args->output_file);
    fprintf(yaml_file, "block_size: %ld\n", args->block_size);
    fprintf(yaml_file, "chunks: %ld\n", args->chunks);
    fprintf(yaml_file, "time_secs: %.15f\n", time_secs);
    fprintf(yaml_file, "rank: %d\n", my_rank);
    fprintf(yaml_file, "num_ranks: %d\n", num_ranks);
    fprintf(yaml_file, "bytes_written: %ld\n", bytes_written);

    fclose(yaml_file);
}
#endif

int main(int argc, char **argv) {
    Args *args = Args_init(argc, argv);

    long num_rows, num_cols;
    Args_get_bitmap_dims(args, &num_rows, &num_cols);

    #ifdef PARALLEL
    // Start parallel execution, potentially with timer data.
    MPI_Init(&argc, &argv);

    tick start = getticks();

    int my_rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Set the max number of writes based on the image size and number of ranks.
    args->chunks = fmin(args->chunks, num_rows / num_ranks);

    long bytes_written = compute_mandelbrot_parallel(args);

    tick end = getticks();

    double time_secs = (end - start) / (double)CLOCK_HERTZ;

    if (args->time_dir != NULL) {
        // Wait for all processes to finish writing and data to be flushed.

        write_yaml(args, time_secs, bytes_written, my_rank, num_ranks);
    }

    MPI_Finalize();

    #else
    // Set the max number of writes based on the image size.
    args->chunks = fmin(args->chunks, num_rows);

    // Start serial execution.
    compute_mandelbrot_serial(args);
    #endif

    if (args->delete_output) {
        remove(args->output_file);
    }

    Args_free(args);

    return EXIT_SUCCESS;
}
