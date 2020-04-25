#ifndef KERNEL_H
#define KERNEL_H

extern void cuda_init(int my_rank);
extern void launch_mandelbrot_kernel(Bitmap *bitmap, int grid_width, int grid_height, int grid_offset_y, int iterations, int block_size);

#endif // KERNEL_H