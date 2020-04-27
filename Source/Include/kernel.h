#ifndef KERNEL_H
#define KERNEL_H

#include "bitmap.h"
#include "mandelbrot.h"

extern void cuda_init(int my_rank);
extern void launch_mandelbrot_kernel(Rgb ** grid, Bitmap *bitmap, int grid_width, int grid_height, int grid_offset_y, int iterations, int block_size);

#endif // KERNEL_H
