#ifndef BITMAP_H
#define BITMAP_H

#include <stdio.h>

#ifdef PARALLEL
#include <mpi.h>
#endif

#include "colormap.h"

/**
 * @brief Bitmap object containing file pointers and useful info 
 */
typedef struct {
    // Public
    char *image_file_name;
    long height;
    long width;

    // Private
    int _padding_size;

    #ifdef PARALLEL
    MPI_File _file;
    #else
    FILE *_file;
    #endif
} Bitmap;

Bitmap *Bitmap_init(long num_rows, long num_cols, const char *image_file_name);
void Bitmap_free(Bitmap *self);

void Bitmap_write_pixel(Bitmap *self, Rgb pixel, long row, long col);
void Bitmap_write_rows(Bitmap *self, Rgb **pixels, long start_row, long num_rows);

#endif
