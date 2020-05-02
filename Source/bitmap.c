// Writes pixel data to .bmp file

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef PARALLEL
#include <mpi.h>
#endif

#include "bitmap.h"

// Constants
const int BYTES_PER_PIXEL = 3; // for 3 pixel colors: blue, green, red
const int FILE_HEADER_SIZE = 14; // format-required
const int INFO_HEADER_SIZE = 40; // format-required
const unsigned char PADDING[3] = {0,0,0}; // .bmp format padding array

enum WriteType { COLLECTIVE, NON_COLLECTIVE };
typedef enum WriteType WriteType;

// Function declarations
int _write_at(const Bitmap *self, long offset, const unsigned char *data, long len_data, WriteType write_type);
unsigned char *_create_bmp_file_header(const Bitmap *self);
unsigned char *_create_bmp_info_header(const Bitmap *self);

// Public methods
/**
 * @brief Initialize the Btimap object and .bmp output file
 * 
 * @param num_rows Height of the image in pixels.
 * @param num_cols Width of the image in pixels.
 * @param image_file_name Output file name
 * @param file_type Designates Bitmap as setup for either a @c SERIAL or @c PARALLEL computation environment
 * @return @c Bitmap* The Bitmap object to be output, or NULL if the file could not be created.
 */
Bitmap *Bitmap_init(long num_rows, long num_cols, const char *file_name) {
    Bitmap *self = calloc(1, sizeof(Bitmap));

    self->num_cols = num_cols;
    self->num_rows = num_rows;

    // compute padding size (math magic)
    self->_padding_size = (4 - (self->num_cols * BYTES_PER_PIXEL) % 4) % 4;

    bool write_headers = true;

    // Open file in write, binary append mode.
    // Existing file of the same name will be deleted.
    #ifdef PARALLEL
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        write_headers = true;
        remove(file_name);
    }
    // Make sure any conflicting file is removed before a process tries to open it.
    MPI_Barrier(MPI_COMM_WORLD);

    int result = MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_APPEND, MPI_INFO_NULL, &self->_file);

    if (result != MPI_SUCCESS) {
        return NULL;
    }
    #else
    self->_file = fopen(file_name, "wb+");

    if (self->_file == NULL) {
        return NULL;
    }
    #endif

    if (write_headers) {
        // create .bmp file headers
        unsigned char* file_header = _create_bmp_file_header(self);
        unsigned char* info_header = _create_bmp_info_header(self);

        // In the parallel version, only process 0 calls these, so they cannot be collective.
        int status1 = _write_at(self, 0, file_header, FILE_HEADER_SIZE, NON_COLLECTIVE);
        int status2 = _write_at(self, FILE_HEADER_SIZE, info_header, INFO_HEADER_SIZE, NON_COLLECTIVE);

        if (status1 != 0 || status2 != 0) {
            return NULL;
        }
    }

    return self;
}

/**
 * @brief Bitmap destructor.
 * 
 * @param self Bitmap object to be removed from memory
 * @return 0 on successful file closing, 1 otherwise.
 */
int Bitmap_free(Bitmap *self) {
    #ifdef PARALLEL
    int result = MPI_File_close(&self->_file);

    if (result != MPI_SUCCESS) {
        return 1;
    }
    #else
    int result = fclose(self->_file);

    if (result != 0) {
        return 1;
    }
    #endif

    free(self);

    return 0;
}

/**
 * @brief Writes passed pixel rows to the output file using parallel MPI methods.
 *
 * Uses collective IO in the parallel version, so every process must call this method.
 * 
 * @param self Bitmap object
 * @param pixels Array of pixel rows
 * @param rows_to_write Number of pixel rows
 * @param start_row The pixel row index to start writing at, with origin in the top left corner.
 */
int Bitmap_write_rows(Bitmap *self, Rgb *pixels, long start_row, long rows_to_write) {
    // compute padding needed and size of array to be written
    long pixels_data_length = rows_to_write * ((self->num_cols * BYTES_PER_PIXEL) + self->_padding_size);
    unsigned char pixels_data[pixels_data_length];

    long index = 0;
    // Rows in the bitmap are stored in reverse.
    for (long row = rows_to_write - 1; row >= 0; row--) {
        for (long col = 0; col < self->num_cols; col++) {
            // add pixel to array to be written
            pixels_data[index]      = pixels[self->num_cols * row + col].blue;
            pixels_data[index + 1]  = pixels[self->num_cols * row + col].green;
            pixels_data[index + 2]  = pixels[self->num_cols * row + col].red;

            index += 3;
        }

        // add padding to end of row in array to be written
        for (int k = 0; k < self->_padding_size; k++) {
            pixels_data[index] = PADDING[0];
            index++;
        }
    }

    long x = 0;
    long y = self->num_rows - start_row;

    // This is the offset where the block to write ends (exclusive), since the bitmap image is stored with rows in reverse order.
    // This can also be thought of as the offset form the end of the file where the first pixel viewed (last pixel in the data) is written.
    long pixel_offset = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (y * (self->num_cols * BYTES_PER_PIXEL + self->_padding_size)) + (x * BYTES_PER_PIXEL);
    // Convert offset to place in the file where the write should start.
    pixel_offset -= pixels_data_length;

    return _write_at(self, pixel_offset, pixels_data, pixels_data_length, COLLECTIVE);
}

// Private methods

// Last parameter is ignored in serial version.
int _write_at(const Bitmap *self, long offset, const unsigned char *data, long len_data, WriteType write_type) {
    // Both parallel and serial versions seek from the beginning of the file every time.
    #ifdef PARALLEL
    int result = 0;

    if (write_type == COLLECTIVE) {
        result = MPI_File_write_at_all(self->_file, offset, data, len_data, MPI_UNSIGNED_CHAR, NULL);
    }
    else {
        result = MPI_File_write_at(self->_file, offset, data, len_data, MPI_UNSIGNED_CHAR, NULL);
    }

    if (result != MPI_SUCCESS) {
        return 1;
    }
    #else

    int result1 = fseek(self->_file, offset, SEEK_SET);
    int bytes_written = fwrite(data, len_data, 1, self->_file);

    if (result1 != 0 || bytes_written == 0) {
        return 1;
    }
    #endif

    return 0;
}

/**
 * @brief Creates array containing .bmp format-required file header based in image specifications
 * 
 * @param self Bitmap object
 */
unsigned char *_create_bmp_file_header(const Bitmap *self) {
    
    // compute full file size
    int file_size = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (BYTES_PER_PIXEL * self->num_cols + self->_padding_size) * self->num_rows;

    // set file header (all from file format guidelines)
    static unsigned char file_header[] = {
        0,0, // signature
        0,0,0,0, // image file size in bytes
        0,0,0,0, // reserved
        0,0,0,0, // start of pixel array
    };

    file_header[0] = (unsigned char)('B');
    file_header[1] = (unsigned char)('M');
    file_header[2] = (unsigned char)(file_size);
    file_header[3] = (unsigned char)(file_size >> 8);
    file_header[4] = (unsigned char)(file_size >> 16);
    file_header[5] = (unsigned char)(file_size >> 24);
    file_header[10] = (unsigned char)(FILE_HEADER_SIZE + INFO_HEADER_SIZE);

    return file_header;
}

/**
 * @brief Creates array containing .bmp format-required info header based in image specifications
 * 
 * @param self Bitmap object
 */
unsigned char *_create_bmp_info_header(const Bitmap *self) {

    // set info header (all from file format guidelines)
    static unsigned char info_header[] = {
        0,0,0,0, // header size
        0,0,0,0, // image width
        0,0,0,0, // image height
        0,0, // number of color planes
        0,0, // bits per pixel
        0,0,0,0, // compression
        0,0,0,0, // image size
        0,0,0,0, // horizontal resolution
        0,0,0,0, // vertical resolution
        0,0,0,0, // colors in color table
        0,0,0,0, // important color count
    };

    info_header[0] = (unsigned char)(INFO_HEADER_SIZE);
    info_header[4] = (unsigned char)(self->num_cols);
    info_header[5] = (unsigned char)(self->num_cols >> 8);
    info_header[6] = (unsigned char)(self->num_cols >> 16);
    info_header[7] = (unsigned char)(self->num_cols >> 24);
    info_header[8] = (unsigned char)(self->num_rows);
    info_header[9] = (unsigned char)(self->num_rows >> 8);
    info_header[10] = (unsigned char)(self->num_rows >> 16);
    info_header[11] = (unsigned char)(self->num_rows >> 24);
    info_header[12] = (unsigned char)(1);
    info_header[14] = (unsigned char)(BYTES_PER_PIXEL * 8);

    return info_header;
}
