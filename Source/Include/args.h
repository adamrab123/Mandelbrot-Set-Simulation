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

Args *Args_init(int argc, char **argv);

void Args_free(Args *self);

#ifdef __CUDACC__
__host__ __device__
#endif
void Args_bitmap_dims(const Args *self, long *width, long *height);

#ifdef __CUDACC__
__host__ __device__
#endif
void Args_bitmap_to_complex(const Args *self, int x, int y, double *c_real, double *c_imag);

#endif // ARGS_H