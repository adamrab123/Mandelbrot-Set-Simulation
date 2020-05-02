#ifndef ARGS_H
#define ARGS_H

/**
 * @brief Contains information about the optional command 
 *      line parameters passed into our simulation
 */
typedef struct {
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    double step_size;
    long iterations;
    char *output_file;
    int block_size;
} Args;

Args *get_args(int argc, char **argv);

void Args_free(Args *self);

#ifdef __CUDACC__
__host__ __device__
#endif
void Args_get_bitmap_dims(const Args *self, long *num_rows, long *num_cols);

#ifdef __CUDACC__
__host__ __device__
#endif
void Args_bitmap_to_complex(const Args *self, long row, long col, double *c_real, double *c_imag);

#endif // ARGS_H