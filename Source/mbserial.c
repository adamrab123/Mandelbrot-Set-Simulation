#include <mpfr.h>
#include <mpc.h>

#include "mbserial.h"
#include "bitmap.h"
#include "args.h"

void compute_mandelbrot_serial(const Args *args) {
    int px_width, px_height;
    Args_bitmap_dims(args, &px_width, &px_height);

    Bitmap *bitmap = Bitmap_init(px_width, px_height, args->output_file, SERIAL);

    mpc_t c;
    mpc_init2(c, args->prec);

    Mandelbrot *mb = Mandelbrot_init(args->iterations, args->prec, args->rnd);

    for (int y = 0; y < px_height; y++) {
        for (int x = 0; x < px_width; x++) {
            bitmap_to_complex(x, y, c);
            MandelbrotPoint *point = Mandelbrot_iterate(mb, c);

            Rgb color;

            if (point->diverged) {
                double norm_iters = mpfr_get_d(point->norm_iters, args->prec);
                color = ColorMap_hsv_based(norm_iters);
            }
            else {
                color = RGB_BLACK;
            }

            Bitmap_write_pixel_serial(bitmap, color, x, y);
            MandelbrotPoint_free(point);
        }
    }

    Bitmap_free(bitmap);
    Mandelbrot_free(mb);
    mpc_clear(c);
}