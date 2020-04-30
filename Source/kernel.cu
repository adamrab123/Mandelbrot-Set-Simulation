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
__global__ void _mandelbrot_kernel(Rgb **grid, long grid_width, long grid_height, long grid_offset_y, const Args *args) {
    // Strided CUDA for loop.
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(; index < grid_width * grid_height; index += stride) {
        int grid_x = index / grid_width;
        int grid_y = grid_offset_y + index % grid_height;

        double c_real, c_imag;
        Args_bitmap_to_complex(args, grid_x, grid_y, &c_real, &c_imag);

        MandelbrotPoint *point = Mandelbrot_iterate(c_real, c_imag, args->iterations);

        grid[grid_x][grid_y] = ColorMap_hsv_based(point->norm_iters);

        MandelbrotPoint_free(point);
    }
}

/**
 * @brief starts the mandelbrot kernel with @p blocksize threads with each point undergoing @p num_iterations
 * 
 * @param num_iterations number of iterations per point
 * @param block_size number of threads per block
 */
// extern "C" void launch_mandelbrot_kernel(unsigned char ** grid, Bitmap *bitmap, int grid_width, int grid_height, int grid_offset_y, int iterations, int block_size){
extern "C" void launch_mandelbrot_kernel(Rgb ** grid, long grid_width, long grid_height, long grid_offset_y, const Args *args){
    long N = grid_width * grid_height;
    int num_blocks = (N + args->block_size - 1) / args->block_size;

    // Launch kernel
    _mandelbrot_kernel<<<num_blocks, args->block_size>>>(grid, grid_width, grid_height, grid_offset_y, args);
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
