#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<string.h>
#include<cuda.h>
#include<cuda_runtime.h>

// Useful variables
int x_axis = 0, y_axis = 0;
int grid_width = 0, grid_height = 0;
int step_size = 0
int total_grid_size = 0;
unsigned char ** grid = NULL;

/**
 * @brief Allocates the grid using cudaMallocManaged
 */
extern "C" void allocate_grid(){      
    // Get data sizes using axis_size * step_size
    grid_width = int(x_axis * 1.0/step_size);
    grid_height = int(y_axis * 1.0/step_size);
    // allocate rows
    int error = cudaMallocManaged( & grid, grid_width * sizeof(unsigned char *));
    // check if the allocation yielded an error
    if(error != cudaSuccess){
        printf("Error received with error %d!\n", error);
        exit(EXIT_FAILURE);
    }   

    // allocate columns
    for (int i = 0; i < grid_width; i++){
        error = cudaMallocManaged( & grid[i], grid_height * sizeof(unsigned char));
        if(error != cudaSuccess){
            printf("Error received with error %d!\n", error);
            exit(EXIT_FAILURE);
        }
        // Initialize grid to 0
        cudaMemset(grid[i], 0, grid_height);
    }
}

/**
 * @brief Initializes the grid of size @p x_axis * @p y_axis with @p step
 * 
 * @param dim_width the size of the x-axis of the grid
 * @param dim_height the size of the y-axis of the grid
 * @param step the step size increments of the grid
 */
extern "C" void init( int dim_width, int dim_height, int step )
{
    x_axis = dim_width;
    y_axis = dim_height;
    step_size = step;
	int cudaDeviceCount;
	cudaError_t cE;
	if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess )
    {
        printf(" Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount );
        exit(-1);
    }
    if( (cE = cudaSetDevice( myrank % cudaDeviceCount )) != cudaSuccess )
    {
        printf(" Unable to have rank %d set to cuda device %d, error is %d \n", myrank, (myrank % cudaDeviceCount), cE);
        exit(-1);
    }
    allocate_grid();
}

/**
 * @brief Iterates on grid to generate mandlebrot set points
 * 
 * @param grid the grid
 */
extern "C" __global__ void mandlebrot_kernel(unsigned char** grid, int num_iterations){
    // Put some math code in here
}

/**
 * @brief starts the mandlebrot kernel with @p blocksize threads with each point undergoing @p num_iterations
 * 
 * @param num_iterations number of iterations per point
 * @param block_size number of threads per block
 */
extern "C" bool launch_mandlebrot_kernel(int num_iterations, ushort block_size){
    int N = total_grid_size;
    int numBlocks = (N+block_size-1)/block_size;

    // QUESTION: do we iterate here or in kernel?

    // Launch kernel
    mandlebrot_kernel<<<numBlocks, block_size>>>(grid, num_iterations);
    // Synchronize threads
    cudaDeviceSynchronize();
    return true;
}

/**
 * @brief Frees the memory previously allocated using cudaMallocManaged
 */
extern "C" void freeCuda(){
    // free individual columns
    for (int i = 0; i < grid_width; i++){
        cudaFree(grid[i]);
    }
    // finally free the overall grid
	cudaFree(grid);
}