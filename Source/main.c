#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include<complex.h>

#include "mandelbrot.h"

const int iters = 100;

void assert_diverged(long double real, long double img) {
    MB_Point p1 = MB_iterate_mandelbrot(real, img, iters);
    assert(MB_diverged(&p1));

    printf("%Lf + %Lf i diverged after %d iterations\n", creall(p1.c), cimagl(p1.c), p1.iters_performed);
}

void assert_converged(long double real, long double img) {
    MB_Point p1 = MB_iterate_mandelbrot(real, img, iters);
    assert( ! MB_diverged(&p1));

    printf("%Lf + %Lf i converged. %d iterations were performed\n", creall(p1.c), cimagl(p1.c), p1.iters_performed);
}

int main() {
    // All point (-2, 0) to (1/4, 0) are on the set.
    assert_converged(-1, 0);
    assert_converged(0.2222332, 0);
    assert_diverged(5, 3);

    return EXIT_SUCCESS;
}