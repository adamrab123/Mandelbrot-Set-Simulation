#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<string.h>
#include<cuda.h>
#include<cuda_runtime.h>

#include "mandelbrot.h"
#include "args.h"
#include "colormap.h"

/**
 * @brief Iterates on grid to generate mandelbrot set points
 * 
 * @param grid the grid
 */
__global__ void _mandelbrot_kernel(Rgb **grid, long start_row, long num_rows, long num_cols, const Args *args) {
    // Strided CUDA for loop over this process's grid, which is a portion of the entire image.
    long index = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for(; index < num_cols * num_rows; index += stride) {
        // Row and col in this process's grid.
        long row = index / num_cols;
        long col = index % num_cols;

        // Add the start_row to row so that we get the exact mandelbrot coordinates from the whole image.
        double c_real, c_imag;
        Args_bitmap_to_complex(args, start_row + row, col, &c_real, &c_imag);

        MandelbrotPoint *point = Mandelbrot_iterate(c_real, c_imag, args->iterations);

        Rgb color;
        if (point->diverged) {
            color = ColorMap_hsv_based(point->norm_iters);
        }
        else {
            color = RGB_BLACK;
        }

        // Do not use the start_row here. That gives the row for the whole image, not our section of the grid.
        grid[row][col] = color;

        free(point);
    }
}

/**
 * @brief starts the mandelbrot kernel with @p blocksize threads with each point undergoing @p num_iterations
 * 
 * @param num_iterations number of iterations per point
 * @param block_size number of threads per block
 */
extern "C" void launch_mandelbrot_kernel(Rgb ** grid, long start_row, long num_rows, long num_cols, const Args *args){
    long grid_area = num_cols * num_cols;
    int num_blocks = (grid_area + args->block_size - 1) / args->block_size;

    // Launch kernel
    _mandelbrot_kernel<<<num_blocks, args->block_size>>>(grid, start_row, num_rows, num_cols, args);
    // Synchronize threads
    cudaDeviceSynchronize();
}

extern "C" void cuda_init(int my_rank) {
	int cudaDeviceCount;
	cudaError_t cE;
	if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess )
    {
        printf(" Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount );
        exit(-1);
    }
    if( (cE = cudaSetDevice( my_rank % cudaDeviceCount )) != cudaSuccess )
    {
        printf(" Unable to have rank %d set to cuda device %d, error is %d \n", my_rank, (my_rank % cudaDeviceCount), cE);
        exit(-1);
    }
}
