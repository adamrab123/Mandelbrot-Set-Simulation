#include <stdlib.h>

#include "mbserial.h"
#include "bitmap.h"
#include "mandelbrot.h"

void compute_mandelbrot_serial(const Args *args) {
    long px_width, px_height;
    Args_bitmap_dims(args, &px_width, &px_height);

    Bitmap *bitmap = Bitmap_init(px_width, px_height, args->output_file);

    for (long y = 0; y < px_height; y++) {
        for (long x = 0; x < px_width; x++) {
            double c_real, c_imag;
            Args_bitmap_to_complex(args, x, y, &c_real, &c_imag);
            MandelbrotPoint *point = Mandelbrot_iterate(c_real, c_imag, args->iterations);

            Rgb color;

            if (point->diverged) {
                color = ColorMap_hsv_based(point->norm_iters);
            }
            else {
                color = RGB_BLACK;
            }

            Bitmap_write_pixel(bitmap, color, x, y);
            free(point);
        }
    }

    Bitmap_free(bitmap);
}