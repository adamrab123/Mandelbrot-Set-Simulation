#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include<complex.h>
#include<math.h>

#include "mandelbrot.h"
#include "bitmap.h"

int main() {
    int x_min = -2;
    int x_max = 2;
    int y_min = -2;
    int y_max = 2;

    int iters = 100;

    long double step = 0.01;

    int px_width = round((x_max - x_min) / step);
    int px_height = round((y_max - y_min) / step);

    Bitmap *bitmap = Bitmap_init(px_width, px_height, "output.bmp", SERIAL);

    for (long double y = y_max; y > y_min; y -= step) {
        for (long double x = x_min; x < x_max; x += step) {
            MB_Point p1 = MB_iterate_mandelbrot(x, y, iters);
            Rgb color = MB_color_of(&p1, HSV_TO_RGB);

            int x_bitmap = round((creal(p1.c) + abs(x_min)) / step);
            int y_bitmap = round((cimag(p1.c) + abs(y_min)) / step);

            Bitmap_write_pixel_serial(bitmap, color, x_bitmap, y_bitmap);
        }
    }

    Bitmap_free(bitmap);

    return EXIT_SUCCESS;
}