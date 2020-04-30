#ifndef ARGS_H
#define ARGS_H

#include <gmp.h>
#include <mpfr.h>
#include <mpc.h>

/**
 * @brief Contains information about the optional command 
 *      line parameters passed into our simulation
 */
typedef struct {
    mpfr_t x_min;
    mpfr_t x_max;
    mpfr_t y_min;
    mpfr_t y_max;
    mpfr_t step_size;
    mpfr_prec_t prec;
    mpfr_rnd_t rnd;
    long long iterations;
    char *output_file;
    int block_size;
} Args;

Args *Args_init(int argc, char **argv);
void Args_free(Args *self);

#ifdef PARALLEL
__host__ __device__
#endif
void Args_bitmap_dims(const Args *self, long *width, long *height);

#ifdef PARALLEL
__host__ __device__
#endif
void Args_bitmap_to_complex(const Args *self, int x, int y, mpc_ptr c);

#endif // ARGS_H