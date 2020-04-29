#ifndef MANDELBROT_H
#define MANDELBROT_H

#include<stdarg.h>
#include<stdbool.h>
#include<mpfr.h>
#include<mpc.h>

#include "rgb.h"

// BEGIN TYPEDEFS

typedef struct {
    long long max_iters;
    mpfr_prec_t prec;
    mpfr_rnd_t rnd;
} Mandelbrot;

/**
 * @brief Contains information about a point c that has undergone some number of Mandelbrot set iterations.
 * 
 * Fields are named based on the Mandelbrot set formula z = z_prev^2 + c.
 */
typedef struct {
    int iters_performed; /** The number of Mandelbrot iterations performed on @c c. */
    mpc_t c; /** The imaginary number whose Mandelbrot info is contained in this struct. */
    mpc_t z_final; /** The value of z after @c iters_performed iterations of @c c */
    bool diverged;
    mpfr_t norm_iters;
} MandelbrotPoint;

// END TYPEDEFS

Mandelbrot *Mandelbrot_init(long long max_iters, mpfr_prec_t prec, mpfr_rnd_t rnd);
void Mandelbrot_free(Mandelbrot *self);

MandelbrotPoint *Mandelbrot_iterate(Mandelbrot *self, mpc_t c);
void MandelbrotPoint_free(MandelbrotPoint *self);

#endif // MANDELBROT_H