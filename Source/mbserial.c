#include <stdlib.h>

#include "mbserial.h"
#include "bitmap.h"
#include "mandelbrot.h"

Rgb **_allocate_grid(long grid_width, long grid_height){
    Rgb **grid = NULL;

    // allocate rows
    grid = calloc(grid_height, sizeof(Rgb *)); 

    // allocate columns
    for (long i = 0; i < grid_height; i++){
        grid[i] = calloc(grid_width, sizeof(Rgb)); 
    }

    return grid;
}

void _free_grid(Rgb **grid, long grid_width, long grid_height) {
    for (long i = 0; i < grid_height; i++){
        free(grid[i]);
    }

    free(grid);
}

void compute_mandelbrot_serial(const Args *args) {
    long px_width, px_height;
    Args_bitmap_dims(args, &px_width, &px_height);

    Bitmap *bitmap = Bitmap_init(px_width, px_height, args->output_file);

    for (long y = 0; y < px_height; y++) {
        Rgb **grid = _allocate_grid(px_width, 1);

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

            grid[0][x] = color;
            free(point);
        }

        Bitmap_write_rows(bitmap, grid, y, 1);
        _free_grid(grid, px_width, 1);
    }

    Bitmap_free(bitmap);
}