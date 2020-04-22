#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<string.h>
#include<cuda.h>
#include<cuda_runtime.h>

// Useful variables
int x_axis, y_axis;
int grid_width, grid_height;
int step_size;
int my_rank, num_ranks;
int start_x, end_x; //points that each rank is responsible for

/**
 * @brief Initialize variables and assign portion of grid to the current rank
 * 
 * @param dim_width the size of the x-axis of the grid
 * @param dim_height the size of the y-axis of the grid
 * @param step the step size increments of the grid
 * @param myrank the current rank
 * @param numranks the total number of ranks
 */
extern "C" void init( int dim_width, int dim_height, int step, int myrank, int numranks )
{
    x_axis = dim_width;
    y_axis = dim_height;
    step_size = step;
    grid_width = int(x_axis * 1.0/step_size);
    grid_height = int(y_axis * 1.0/step_size);
    my_rank = myrank;
    num_ranks = numranks;
    
    // Divide up responsibility of grid as best as possible
    // Case 1: Equal responsibility
    if (grid_width % num_ranks == 0){
        start_x = my_rank * (grid_width/num_ranks);
        end_x = start_x + (grid_width/num_ranks);
    }
    // Case 2: Unequal, but maximize fairness as best as possible
    else {
        int cutoff = num_ranks - (grid_width % num_ranks);
        start_x = my_rank * (grid_width/num_ranks);
        if (my_rank >= cutoff){
            if (my_rank > cutoff){
                start_x++;
            } 
            end_x = start_x + (grid_width/num_ranks) + 1;
        }
        else{
            end_x = start_x + (grid_width/num_ranks);
        }
    }

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
}

/**
 * @brief Iterates on grid to generate mandlebrot set points
 * 
 * @param grid the grid
 */
__global__ void mandlebrot_kernel(unsigned char** grid, int num_iterations, int my_rank){
    // Put some math code in here

    // Pass result to image
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

    // Launch kernel
    mandlebrot_kernel<<<numBlocks, block_size>>>(grid, num_iterations, my_rank);
    // Synchronize threads
    cudaDeviceSynchronize();
    return true;
}
