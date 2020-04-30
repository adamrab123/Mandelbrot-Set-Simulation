#ifndef BITMAP_H
#define BITMAP_H

#include <stdio.h>

#ifdef PARALLEL
#include <mpi.h>
#endif

#include "mandelbrot.h"
#include "colormap.h"

typedef unsigned long ul;

/**
 * @brief Bitmap object containing file pointers and useful info 
 */
typedef struct {
    // Public
    char *image_file_name;
    ul height;
    ul width;

    // Private
    int _padding_size;

    #ifdef PARALLEL
    MPI_File *_file;
    #else
    FILE *_file;
    #endif
} Bitmap;

/**
 * @brief Initialize the Btimap object and .bmp output file
 * 
 * @param width Image width
 * @param height Image height
 * @param image_file_name Output file name
 * @param file_type Designates Bitmap as setup for either a @c SERIAL or @c PARALLEL computation environment
 * @return @c Bitmap* The Bitmap object to be output
 */
Bitmap *Bitmap_init(int width, int height, const char *image_file_name);

/**
 * @brief Bitmap destructor
 * 
 * @param self Bitmap object to be removed from memory
 */
void Bitmap_free(Bitmap *self);

/**
 * @brief Writes passed pixel to the output file using serial C methods
 * 
 * @param self Bitmap object
 * @param pixel Rgb enum containing color data
 * @param x Pixel 'X' coordinate (offset for image plane)
 * @param y Pixel 'Y' coordinate (offest for image plane)
 */
void Bitmap_write_pixel(Bitmap *self, Rgb pixel, long x, long y) {

/**
 * @brief Writes passed pixel rows to the output file using parallel MPI methods
 * 
 * @param self Bitmap object
 * @param pixels Array of pixel rows
 * @param num_rows Number of pixel rows
 * @param start_row 'Y' coordinate of the starting row (offset for image plane)
 */
void Bitmap_write_rows(Bitmap *self, Rgb **pixels, int num_rows, int start_row);

#endif
