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
void _MandelbrotPoint_set_norm_iters(MandelbrotPoint *self, const Mandelbrot *point);

#ifdef __CUDACC__
__host__ __device__
#endif
void _check_flags(const char *file, int line);

#ifdef __CUDACC__
__host__ __device__
#endif
Mandelbrot *Mandelbrot_init(long long max_iters, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    Mandelbrot *self = malloc(sizeof(Mandelbrot));

    self->max_iters = max_iters;
    self->prec = prec;
    self->rnd = rnd;

    return self;
}

#ifdef __CUDACC__
__host__ __device__
#endif
void Mandelbrot_free(Mandelbrot *self) {
    free(self);
}

/**
 * @brief Performs @p iterations number of mandelbrot set iterations on the imaginary number c given by @p c_real and @c
 *      c_complex, and returns the resulting information.
 * 
 * @param c_real The real part of the number c to calculate Mandelbrot set information for.
 * @param c_img The imaginary part of the number c to calculate Mandelbrot set information for.
 * @param iterations The number of Mandelbrot set iterations to perform on c.
 *
 * @return An @c MB_Point instance containing information about the resulting iterations.
 */
#ifdef __CUDACC__
__host__ __device__
#endif
MandelbrotPoint *Mandelbrot_iterate(Mandelbrot *self, mpc_t c) {
    mpc_t z;
    mpc_init2(z, self->prec);
    mpc_set_ui(z, 0, self->rnd);

    mpfr_t z_abs;
    mpfr_init2(z_abs, self->prec);
    mpfr_set_ui(z_abs, 0, self->rnd);

    bool diverged = false;
    unsigned int escape_radius = 2;

    int iters_performed = 0;
    while (iters_performed < self->max_iters && !diverged) {
        // Compute z = z^2 + c.
        mpc_sqr(z, z, self->rnd);
        mpc_add(z, z, c, self->rnd);

        // Absolute value of z is its distance from the origin.
        mpc_abs(z_abs, z, self->rnd);

        diverged = (mpfr_cmp_ui(z_abs, escape_radius) > 0);

        // Check for errors in this calculation.
        _check_flags(__FILE__, __LINE__);

        iters_performed++;
    }

    // Return point information to the user.
    MandelbrotPoint *point = malloc(sizeof(MandelbrotPoint));

    point->iters_performed = iters_performed;

    mpc_init2(point->c, self->prec);
    mpc_set(point->c, c, self->rnd);

    mpc_init2(point->z_final, self->prec);
    mpc_set(point->z_final, z, self->rnd);

    point->diverged = diverged;

    // Uses point->z_final anbd point->diverged.
    _MandelbrotPoint_set_norm_iters(point, self);

    return point;
}

// Free MPFR data contained in the point.
#ifdef __CUDACC__
__host__ __device__
#endif
void MandelbrotPoint_free(MandelbrotPoint *point) {
    mpc_clear(point->z_final);
    mpc_clear(point->c);
    free(point);
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
void _MandelbrotPoint_set_norm_iters(MandelbrotPoint *self, const Mandelbrot *mb) {
    mpfr_init2(self->norm_iters, mb->prec);

    if (!self->diverged) {
        // Points in the set are given the value 1.
        // They may cause NaN results if run through the log calculation below.
        mpfr_set_ui(self->norm_iters, 1, mb->prec);
    }
    else {
        // Compute the following to get a number on teh range (0, max_iter), then normalize to be on (0, 1).
        // iters_performed + 1 - log(log(|z_final|)) / log(2)

        mpfr_set_ui(self->norm_iters, 0, mb->rnd);

        mpfr_t log_of_2;
        mpfr_init2(log_of_2, mb->prec);
        // NOTE: This uses natural log but I think that is OK.
        mpfr_log_ui(log_of_2, 2, mb->rnd);

        // Compute -log(log(|z_final|)) / log(2).
        mpc_abs(self->norm_iters, self->z_final, mb->rnd);
        mpfr_log(self->norm_iters, self->norm_iters, mb->rnd);
        mpfr_log(self->norm_iters, self->norm_iters, mb->rnd); // THIS LINE CAUSES NAN!
        mpfr_div(self->norm_iters, self->norm_iters, log_of_2, mb->rnd);
        mpfr_neg(self->norm_iters, self->norm_iters, mb->rnd);

        // Add iters_performed + 1 to the result above.
        mpfr_add_ui(self->norm_iters, self->norm_iters, self->iters_performed + 1, mb->rnd);

        // Normalize smooth to be on (0, 1).
        mpfr_div_ui(self->norm_iters, self->norm_iters, mb->max_iters, mb->rnd);

        // mpfr_printf("Norm iters: %Rf\n", self->norm_iters);

        _check_flags(__FILE__, __LINE__);
    }
}

#ifdef __CUDACC__
__host__ __device__
#endif
void _check_flags(const char *file, int line) {
    if (mpfr_flags_test(MPFR_FLAGS_OVERFLOW)) {
        fprintf(stderr, "%s: %d Overflow in computation\n", file, line);
        exit(1);
    }
    else if (mpfr_flags_test(MPFR_FLAGS_UNDERFLOW)) {
        fprintf(stderr, "%s: %d Underflow in computation\n", file, line);
        exit(1);
    }
    else if (mpfr_flags_test(MPFR_FLAGS_NAN)) {
        fprintf(stderr, "%s, %d NaN result in computation\n", file, line);
        exit(1);
    }
    else if (mpfr_flags_test(MPFR_FLAGS_DIVBY0)) {
        fprintf(stderr, "%s, %d Divide by 0 in computation\n", file, line);
        exit(1);
    }
    else if (mpfr_flags_test(MPFR_FLAGS_ERANGE)) {
        // Cuased by a function not returning an mpfr number having an invalid result, like NaN.
        fprintf(stderr, "%s, %d Erange result in computation\n", file, line);
        exit(1);
    }
    // This is set whenever rounding occurs. Do not treat this as an error.
    // else if (mpfr_flags_test(MPFR_FLAGS_INEXACT)) {
    //     fprintf(stderr, "%s, %d Inexact result in computation\n", file, line);
    //     exit(1);
    // }
}
