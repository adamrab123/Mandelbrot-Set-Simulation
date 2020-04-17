#ifndef BITMAP_H
#define BITMAP_H

#include <stdio.h>

/**
 * @brief Takes as input a 1D integer array (and image details) and writes to output.bmp
 * 
 * @param height The final image height
 * 
 * @param width The final image width
 * 
 */
void write_array_to_bitmap(unsigned char *data, int height, int width);

#endif
