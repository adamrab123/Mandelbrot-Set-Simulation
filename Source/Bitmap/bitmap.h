#ifndef BITMAP_H
#define BITMAP_H

#include <stdio.h>

/**
 * @brief Initialize the .bmp output file with a given height and width
 * 
 * @param height The image height in pixels
 * 
 * @param width The image width in pixels
 * 
 */
void init_output_file(int height, int width);

/**
 * @brief Writes passed pixel to the output file using sequential C methods
 * 
 * @param pixel Array of size 3 containing values for each color
 */
void write_pixel_to_file_sequential(unsigned char *pixel);

/**
 * @brief Writes passed pixel to the output file using parallel MPI methods
 * 
 * @param pixel Array of size 3 containing values for each color
 */
void write_pixel_to_file_parallel(unsigned char *pixel);

#endif
