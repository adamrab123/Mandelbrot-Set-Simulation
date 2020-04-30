#include<stdio.h>
#include<stdlib.h>
#include<complex.h>
#include<stdbool.h>
#include<math.h>

#include<assert.h>

#include "mandelbrot.h"

#ifdef __CUDACC__
__host__ __device__
#endif
void _MandelbrotPoint_set_norm_iters(MandelbrotPoint *self);

/**
 * @brief Performs @p iterations number of mandelbrot set iterations on the imaginary number c given by @p c_real and @c
 *      c_complex, and returns the resulting information.
 * 
 * @param c_real The real part of the number c to calculate Mandelbrot set information for.
 * @param c_img The imaginary part of the number c to calculate Mandelbrot set information for.
 * @param iterations The number of Mandelbrot set iterations to perform on c.
 *
 * @return An @c MB_Point instance containing information about the resulting iterations. The data should be freed after use.
 */
#ifdef __CUDACC__
__host__ __device__
#endif
MandelbrotPoint *Mandelbrot_iterate(double c_real, double c_image, long iterations) {
    bool diverged = false;
    unsigned int escape_radius = 2;

    MbComplex c = MbComplex_init(c_real, c_image);
    MbComplex z = MbComplex_init(0, 0);

    long iters_performed = 0;
    while (iters_performed < iterations && !diverged) {
        // Compute z = z^2 + c.
        MbComplex z_squared = MbComplex_mul(z, z);
        MbComplex_assign(z, MbComplex_add(z_squared, c));

        // Absolute value of z is its distance from the origin.
        diverged = (MbComplex_abs(z) > 0);

        iters_performed++;
    }

    // Return point information to the user.
    MandelbrotPoint *point = (MandelbrotPoint *)malloc(sizeof(MandelbrotPoint));

    point->iters_performed = iters_performed;

    point->c_real = c.real;
    point->c_imag = c.imag;
    point->z_real = z.real;
    point->z_imag = z.imag;

    point->diverged = diverged;

    // Uses the above information.
    _MandelbrotPoint_set_norm_iters(point);

    return point;
}

/**
 * @brief Computes the noormalized iteration count of @p point.
 *
 * Given the number of iterations performed and maximum possible iterations that could have been performed on @p point,
 * generates a value on the range (0, 1), where a higher number means more iterations were performed before the point
 * diverged.
 *
 * @param point The point whose Mandelbrot iteration information will be used in the computation.
 *
 * @return A value on the range (0, 1).
 */
#ifdef __CUDACC__
__host__ __device__
#endif
void _MandelbrotPoint_set_norm_iters(MandelbrotPoint *self) {
    if (!self->diverged) {
        // Points in the set are given the value 1.
        // They may cause NaN results if run through the log calculation below.
        self->norm_iters = 1;
    }
    else {
        // Compute the following to get a number on teh range (0, max_iter), then normalize to be on (0, 1).
        // iters_performed + 1 - log(log(|z_final|)) / log(2)
        double z_abs = MbComplex_abs(MbComplex_init(self->z_real, self->z_imag));

        self->norm_iters = self->iters_performed + 1 - log(log(z_abs)) / log(2);

        // Normalize to be on (0, 1).
        self->norm_iters /= self->max_iters;
    }
}
