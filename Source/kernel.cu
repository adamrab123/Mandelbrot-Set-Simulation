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
__global__ void _mandelbrot_kernel(Rgb **grid, int grid_width, int grid_height, int grid_offset_y, const Args *args) {
    Mandelbrot *mb = Mandelbrot_init(args->iterations, args->prec, args->rnd);

    // Strided CUDA for loop.
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(; index < grid_width * grid_height; index += stride) {
        int grid_x = index / grid_width;
        int grid_y = grid_offset_y + index % grid_height;

        mpc_t c;
        Args_bitmap_to_complex(args, grid_x, grid_y, c);

        MandelbrotPoint *point = Mandelbrot_iterate(mb, c);

        double norm_iters = mpfr_get_d(point->norm_iters, args->rnd);
        grid[grid_x][grid_y] = ColorMap_hsv_based(norm_iters);
    }
}

/**
 * @brief starts the mandelbrot kernel with @p blocksize threads with each point undergoing @p num_iterations
 * 
 * @param num_iterations number of iterations per point
 * @param block_size number of threads per block
 */
// extern "C" void launch_mandelbrot_kernel(unsigned char ** grid, Bitmap *bitmap, int grid_width, int grid_height, int grid_offset_y, int iterations, int block_size){
extern "C" void launch_mandelbrot_kernel(Rgb ** grid, int grid_width, int grid_height, int grid_offset_y, const Args *args){
    int N = grid_width * grid_height;
    int num_blocks = (N + args->block_size - 1) / args->block_size;

    // Launch kernel
    _mandelbrot_kernel<<<num_blocks, block_size>>>(grid, grid_width, grid_height, grid_offset_y, args);
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
