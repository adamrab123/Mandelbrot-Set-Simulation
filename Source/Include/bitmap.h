#ifndef BITMAP_H
#define BITMAP_H

#include <stdio.h>
#include <mpi.h>

/**
 * @brief Represents an RGB color with red, green, and blue integer fields on [0, 255].
 */
typedef struct {
    unsigned char red;
    unsigned char green;
    unsigned char blue;
} Rgb;

typedef struct {
    // Public.
    char *image_file_name;
    FILE *serial_file;
    MPI_File *parallel_file;
    int height;
    int width;

    // Private.
    int _padding_size;
} Bitmap;

enum FileType { SERIAL, PARALLEL };
typedef enum FileType FileType;

/**
 * @brief Initialize the .bmp output file with a given height and width
 * 
 * @param height The image height in pixels
 * 
 * @param width The image width in pixels
 * 
 */
Bitmap *Bitmap_init(int width, int height, const char *image_file_name, FileType file_type);

void Bitmap_free(Bitmap *self);

/**
 * @brief Writes passed pixel to the output file using sequential C methods
 * 
 * @param pixel Array of size 3 containing values for each color
 * 
 * @param y The distance on the y-axis that the input pixel sits on the image
 * 
 * @param x The distance on the x-axis that the input pixel sits on the image
 */
void Bitmap_write_pixel_sequential(Bitmap *self, Rgb pixel, int x, int y);

/**
 * @brief Writes passed pixel to the output file using parallel MPI methods
 * 
 * @param pixel Array of size 3 containing values for each color
 * 
 * @param y The distance on the y-axis that the input pixel sits on the image
 * 
 * @param x The distance on the x-axis that the input pixel sits on the image
 */
void Bitmap_write_pixel_parallel(Bitmap *self, Rgb pixel, int x, int y);

#endif
