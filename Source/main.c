#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include<complex.h>

#include "mandelbrot.h"

int main() {
    int x_min = -2;
    int x_max = 2;
    int y_min = -2;
    int y_max = 2;

    int iters = 100;

    long double step = 0.05;

    for (long double y = y_max; y > y_min; y -= step) {
        for (long double x = x_min; x < x_max; x += step) {
            MB_Point p1 = MB_iterate_mandelbrot(x, y, iters);
            if (MB_diverged(&p1)) {
                printf("   ");
            }
            else {
                printf(" * ");
            }
        }
        printf("\n");
    }

    return EXIT_SUCCESS;
}