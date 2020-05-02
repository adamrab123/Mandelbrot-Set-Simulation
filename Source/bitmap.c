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

// Function declarations
int _write_at_pixel(const Bitmap *self, long row, long col, const unsigned char *data, long data_len);
int _write_at(const Bitmap *self, long offset, const unsigned char *data, long len_data);
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

        int status1 = _write_at(self, 0, file_header, FILE_HEADER_SIZE);
        int status2 = _write_at(self, FILE_HEADER_SIZE, info_header, INFO_HEADER_SIZE);

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
 * @brief Writes passed pixel to the output file using serial C methods
 * 
 * @param self Bitmap object
 * @param pixel Rgb enum containing color data
 * @param row The row if the pixel, indexed from top left corner.
 * @param col The column of the pixel, indexed from top left corner.
 */
int Bitmap_write_pixel(Bitmap *self, Rgb pixel, long row, long col) {
    unsigned char pixel_data[3] = {pixel.blue, pixel.green, pixel.red};

    return _write_at_pixel(self, row, col, pixel_data, sizeof(pixel_data));
}

/**
 * @brief Writes passed pixel rows to the output file using parallel MPI methods
 * 
 * @param self Bitmap object
 * @param pixels Array of pixel rows
 * @param rows_to_write Number of pixel rows
 * @param start_row The pixel row index to start writing at, with origin in the top left corner.
 */
int Bitmap_write_rows(Bitmap *self, Rgb **pixels, long start_row, long rows_to_write) {
    // compute padding needed and size of array to be written
    long pixels_data_length = rows_to_write * ((self->num_cols * BYTES_PER_PIXEL) + self->_padding_size);
    unsigned char pixels_data[pixels_data_length];

    long index = 0;
    for (long row = 0; row < rows_to_write; row++) {
        for (long col = 0; col < self->num_cols; col++) {
            // add pixel to array to be written
            pixels_data[index]      = pixels[row][col].blue;
            pixels_data[index + 1]  = pixels[row][col].green;
            pixels_data[index + 2]  = pixels[row][col].red;

            index += 3;
        }

        // add padding to end of row in array to be written
        for (int k = 0; k < self->_padding_size; k++) {
            pixels_data[index] = PADDING[0];
            index++;
        }
    }

    return _write_at_pixel(self, start_row, 0, pixels_data, pixels_data_length);
}

// Private methods

/**
 * @brief Calculate the offset for a given pixel based on its coords.
 *        The formula takes into account both header sizes, the bytes per pixel value
 *        for each pixel and the amount of padding present in each row.
 * 
 * @param x X coordinate of the pixel with origin in the bottom left corner.
 * @param y Y coordinate of the pixel with origin in the bottom left corner.
 */
int _write_at_pixel(const Bitmap *self, long row, long col, const unsigned char *data, long data_len) {
    // Rows and columns (origin in top left) are converted to bitmap x and y (origin in bottom left).
    long x = col;
    // long y = self->num_rows - (row + 1);
    long y = row;

    long pixel_offset = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (y * (self->num_cols * BYTES_PER_PIXEL + self->_padding_size)) + (x * BYTES_PER_PIXEL);

    return  _write_at(self, pixel_offset, data, data_len);
}

int _write_at(const Bitmap *self, long offset, const unsigned char *data, long len_data) {
    printf("writing data at offset: %ld\n", offset);

    // Both parallel and serial versions seek from the beginning of the file every time.
    #ifdef PARALLEL
    int result = MPI_File_write_at(self->_file, offset, data, len_data, MPI_UNSIGNED_CHAR, NULL);

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
