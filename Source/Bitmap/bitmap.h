#ifndef BITMAP_H
#define BITMAP_H

#include <stdio.h>
#include <string.h>

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
 * 
 * @param y The distance on the y-axis that the input pixel sits on the image
 * 
 * @param x The distance on the x-axis that the input pixel sits on the image
 */
void write_pixel_to_file_sequential(unsigned char *pixel, int y, int x);

/**
 * @brief Writes passed pixel to the output file using parallel MPI methods
 * 
 * @param pixel Array of size 3 containing values for each color
 * 
 * @param y The distance on the y-axis that the input pixel sits on the image
 * 
 * @param x The distance on the x-axis that the input pixel sits on the image
 */
void write_pixel_to_file_parallel(unsigned char *pixel, int y, int x);

#endif
