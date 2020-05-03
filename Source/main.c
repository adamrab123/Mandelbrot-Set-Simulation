#include <stdlib.h>

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
#endif

void write_yaml(const Args *args, tick time_secs, int my_rank, int num_ranks) {
    const int MAX_FILE_NAME_LEN = 256;

    char *yaml_file_name = calloc(MAX_FILE_NAME_LEN, sizeof(char));
    sprintf(yaml_file_name, "%s/%d.yaml", args->time_dir, my_rank);

    FILE *yaml_file = fopen(yaml_file_name, "a");

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

    fclose(yaml_file);
}

int main(int argc, char **argv) {
    // Hard code arguments for now.
    Args *args = Args_init(argc, argv);

    #ifdef PARALLEL
    MPI_Init(&argc, &argv);

    tick start = getticks();

    int my_rank = 0;
    int num_ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    compute_mandelbrot_parallel(args);

    tick end = getticks();

    tick time_secs = (end - start) / (double)CLOCK_HERTZ;

    if (args->time_dir != NULL) {
        write_yaml(args, time_secs, my_rank, num_ranks);
    }

    MPI_Finalize();
    #else
    compute_mandelbrot_serial(args);
    #endif

    Args_free(args);

    return EXIT_SUCCESS;
}
