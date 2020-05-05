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
// The corresponding kernel.h header file exports the host functions declared here.
// It should not be included in this file.

/**
 * @brief Iterates on grid to generate mandelbrot set points
 * 
 * @param grid the grid of RBG values representing the image
 * @param start_row the offset starting row for this rank
 * @param num_rows the number of rows in the grid
 * @param num_cols the number of columns in the grid
 * @param args the command line arguements
 */
__global__ void _mandelbrot_kernel(Rgb *grid, long start_row, long num_rows, long num_cols, const Args *args) {
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
        grid[num_cols * row + col]= color;

        free(point);
    }
}

/**
 * @brief starts the mandelbrot kernel with the given parameters and a given block size
 * 
 * @param grid the grid of RBG values representing the image
 * @param start_row the offset starting row for this rank
 * @param num_rows the number of rows in the grid
 * @param num_cols the number of columns in the grid
 * @param args the command line arguements
 */
extern "C" void launch_mandelbrot_kernel(Rgb *grid, long start_row, long num_rows, long num_cols, const Args *args){
    long grid_area = num_rows * num_cols;
    int num_blocks = (grid_area + args->block_size - 1) / args->block_size;

    // Launch kernel
    _mandelbrot_kernel<<<num_blocks, args->block_size>>>(grid, start_row, num_rows, num_cols, args);
    // Synchronize threads
    cudaDeviceSynchronize();
}

/**
 * @brief Initializes the CUDA device for this rank
 * 
 * @param myrank the current rank
 */
extern "C" void cuda_init(int my_rank) {
	int cudaDeviceCount;
	cudaError_t cE;
	if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess )
    {
        fprintf(stderr, "Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount );
        exit(-1);
    }
    if( (cE = cudaSetDevice( my_rank % cudaDeviceCount )) != cudaSuccess )
    {
        fprintf(stderr, "Unable to have rank %d set to cuda device %d, error is %d \n", my_rank, (my_rank % cudaDeviceCount), cE);
        exit(-1);
    }
}

/**
 * @brief Dynamically allocate memory of size @p size on the device
 * 
 * @param size the size of the memory to be allocated
 *
 * @return mem the void pointer to the allocated memory
 */
extern "C" void *cuda_malloc(size_t size) {
    void *mem = NULL;
    cudaError_t error = cudaMallocManaged(&mem, size);

    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed with error code %d.\n", error);
        exit(EXIT_FAILURE);
    }

    return mem;
}

/**
 * @brief Frees the allocated memory on the device
 * 
 * @param mem the memory to be freed
 */
extern "C" void cuda_free(void *mem) {
    cudaError_t freeError = cudaFree(mem);
    if (freeError != cudaSuccess) {
        fprintf(stderr, "cudaFree failed with error code %d.", freeError);
    }
}
