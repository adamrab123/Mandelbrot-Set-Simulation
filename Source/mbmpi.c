#include <mbmpi.h>

#include "args.h"
#include "kernel.cuh"
#include "bitmap.h"
#include "mandelbrot.h"

int _get_offset(int grid_height);

/**
 * @brief Starts the kernel with for each rank and assigns each rank a portion of the grid
 * 
 * @param args the command line arguements
 */
void start_mpi(const Arguments *args) {
    MPI_Init(NULL, NULL);

    int myrank, numranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    cuda_init(myrank);

    int grid_width = (args->x_max - args->x_min) / args->step_size;
    int grid_height = (args->y_max - args->y_min) / args->step_size;

    Rgb ** grid = allocate_grid(grid_width, grid_height);

    Bitmap *bitmap = Bitmap_init(grid_width, grid_height, args->output_file, PARALLEL);

    int grid_offset_y = _get_offset(grid_height);

    launch_mandelbrot_kernel(grid, bitmap, grid_width, grid_height, grid_offset_y, args->iterations, args->block_size);
    MPI_Barrier(MPI_COMM_WORLD);

    // write_grid(grid, grid_height, grid_offset_y)

    MPI_Finalize();
}


void _bitmap_to_complex(int x, int y, long double *real, long double *imag) {
    *real = x * step_size + x_min;
    *imag = y * step_size + y_min;
}

void _complex_to_bitmap(long double real, long double imag, int *x, int *y) {
    *x = round((real - x_min) / step_size);
    *y = round((imag - y_min) / step_size);
}


// /**
//  * @brief Writes the grid to the image
//  * 
//  * @param grid the grid representing the points to be plotted
//  * @param grid_height the height of the grid
//  * @param offset the y offset for each rank
//  */
// // void write_grid(unsigned char ** grid, int grid_height, int offset){
// void write_grid(MB_Point ** grid, int grid_height, int offset){
//     int my_rank, num_ranks;
//     MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    
//     int bitmap_x, bitmap_y;

//     if(grid_height % num_ranks == 0 || my_rank != num_ranks - 1){
//         for (int i = 0; i < grid_width; i++){
//             for(int j = 0; j < (grid_height / num_ranks); j++){
//                 MB_Point point = grid[i][offset + j]
//                 Rgb color = MB_color_of(&point, DIRECT_RGB);
//                 _complex_to_bitmap(creal(point.c), imag(point.c), &bitmap_x, &bitmap_y);
//                 Bitmap_write_pixel_parallel(bitmap, color, bitmap_x, bitmap_y);
//             }
//         }
//     }
//     else{
//         for (int i = 0; i < grid_width; i++){
//             for(int j = 0; j < (grid_height % num_ranks); j++){
//                 MB_Point point = grid[i][offset + j]
//                 Rgb color = MB_color_of(&point, DIRECT_RGB);
//                 _complex_to_bitmap(creal(point.c), imag(point.c), &bitmap_x, &bitmap_y);
//                 Bitmap_write_pixel_parallel(bitmap, color, bitmap_x, bitmap_y);
//             }
//         }
//     }
// }

/**
 * @brief Allocates the grid of size @p grid_width by @p grid_height using cudaMallocManaged
 * 
 * @param grid_width the width of the grid
 * @param grid_height the height of the grid
 * 
 * @return grid
 */
// unsigned char ** allocate_grid(int grid_width, int grid_height){
Rgb ** allocate_grid(int grid_width, int grid_height){
    Rgb ** grid = NULL;

    // allocate rows
    // int error = cudaMallocManaged( & grid, grid_width * sizeof(unsigned char *)); 
    int error = cudaMallocManaged( & grid, grid_width * sizeof(Rgb *)); // Changed this to MB_POINT

    // check if the allocation yielded an error
    if(error != cudaSuccess){
        printf("Error received with error %d!\n", error);
        exit(EXIT_FAILURE);
    }   

    // allocate columns
    for (int i = 0; i < grid_width; i++){
        // error = cudaMallocManaged( & grid[i], grid_height * sizeof(unsigned char));
        error = cudaMallocManaged( & grid[i], grid_height * sizeof(Rgb)); // Changed this to MB_POINT
        if(error != cudaSuccess){
            printf("Error received with error %d!\n", error);
            exit(EXIT_FAILURE);
        }
        // Initialize grid to 0
        // cudaMemset(grid[i], 0, grid_height);
    }

    return grid;
}

/**
 * @brief Initialize variables and assign portion of grid to the current rank
 * 
 * @param grid_height
 */
int _get_offset(int grid_height) {
    int my_rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int offset = 0;
    if (grid_height % num_ranks == 0 || my_rank != num_ranks - 1) {
        offset = (grid_height / num_ranks) * my_rank;
    }
    else {
        int remainder = grid_height % num_ranks;
        offset = grid_height - remainder;
    }

    return offset;
}
