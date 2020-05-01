#ifndef MANDELBROT_H
#define MANDELBROT_H

#include<stdbool.h>
#include "mbcomplex.h"

// BEGIN TYPEDEFS

/**
 * @brief Contains information about a point c that has undergone some number of Mandelbrot set iterations.
 * 
 * Fields are named based on the Mandelbrot set formula z = z_prev^2 + c.
 */
typedef struct {
    long iters_performed; /** The number of Mandelbrot iterations performed on @c c. */
    long max_iters;
    double c_real;
    double c_imag;
    double z_real;
    double z_imag;
    bool diverged;
    double norm_iters;
} MandelbrotPoint;

// END TYPEDEFS

#ifdef __CUDACC__
__host__ __device__
#endif
MandelbrotPoint *Mandelbrot_iterate(double c_real, double c_image, long iterations);

#endif // MANDELBROT_H