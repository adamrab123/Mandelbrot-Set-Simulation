#ifndef KERNEL_H
#define KERNEL_H

#include <stdlib.h>

#include "colormap.h"
#include "args.h"

// Cuda functions that only exist when kernel.cu is linked in parallel mode.
extern void cuda_init(int my_rank);
extern void launch_mandelbrot_kernel(Rgb *grid, long num_rows, long num_cols, long grid_offset_y, const Args *args);
extern void *cuda_malloc(size_t size);
extern void cuda_free(void *mem);

#endif // KERNEL_H