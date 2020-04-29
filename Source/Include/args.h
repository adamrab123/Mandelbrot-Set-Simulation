#ifndef ARGS_H
#define ARGS_H

#include <mpfr.h>

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

Args *Args_init(int argc, const char **argv);
void Args_free(Args *self);

#endif // ARGS_H