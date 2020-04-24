// Writes pixel data to .bmp file

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bitmap.h"

// Constants
const int BYTES_PER_PIXEL = 3; // for 3 pixel colors: blue, green, red
const int FILE_HEADER_SIZE = 14; // format-required
const int INFO_HEADER_SIZE = 40; // format-required
const unsigned char PADDING[3] = {0,0,0}; // .bmp format padding array

// Function declarations
void _init_parallel(Bitmap *self, const char *file_name);
void _init_serial(Bitmap *self, const char *file_name);
int _compute_pixel_offset(const Bitmap *self, int x, int y);
unsigned char *_create_bmp_file_header(const Bitmap *self);
unsigned char *_create_bmp_info_header(const Bitmap *self);

/**
 * @brief Initialize the Btimap object and .bmp output file
 * 
 * @param width Image width
 * @param height Image height
 * @param image_file_name Output file name
 * @param file_type Designates Bitmap as setup for either a @c SERIAL or @c PARALLEL computation environment
 * @return @c Bitmap* The Bitmap object to be output
 */
Bitmap *Bitmap_init(int width, int height, const char *image_file_name, FileType file_type) {
    Bitmap *self = calloc(1, sizeof(Bitmap));

    // set globals
    self->width = width;
    self->height = height;

    // compute padding size (math magic)
    self->_padding_size = (4 - (width * BYTES_PER_PIXEL) % 4) % 4;

    if (file_type == SERIAL) {
        _init_serial(self, image_file_name);
    }
    else if (file_type == PARALLEL) {
        _init_parallel(self, image_file_name);
    }

    return self;
}

/**
 * @brief Bitmap destructor
 * 
 * @param self Bitmap object to be removed from memory
 */
void Bitmap_free(Bitmap *self) {
    if (self->serial_file != NULL) {
        fclose(self->serial_file);
    }

    if (self->parallel_file != NULL) {
        MPI_File_close(self->parallel_file);
    }

    free(self);
}

/**
 * @brief Writes passed pixel to the output file using serial C methods
 * 
 * @param self Bitmap object
 * @param pixel Rgb enum containing color data
 * @param x Pixel 'X' coordinate (offset for image plane)
 * @param y Pixel 'Y' coordinate (offest for image plane)
 */
void Bitmap_write_pixel_serial(Bitmap *self, Rgb pixel, int x, int y) {
    unsigned char pixel_data[3] = {pixel.blue, pixel.green, pixel.red};

    // compute pixel offset
    int offset = _compute_pixel_offset(self, x, y);

    // write pixel data to file at appropriate location
    fseek(self->serial_file, offset, SEEK_SET);
    fwrite(pixel_data, BYTES_PER_PIXEL, 1, self->serial_file);
}

/**
 * @brief Writes passed pixel to the output file using parallel MPI methods
 * 
 * @param self Bitmap object
 * @param pixel Rgb enum containing color data
 * @param x Pixel 'X' coordinate (offset for image plane)
 * @param y Pixel 'Y' coordinate (offest for image plane)
 */
void Bitmap_write_pixel_parallel(Bitmap *self, Rgb pixel, int x, int y) {
    unsigned char pixel_data[3] = {pixel.blue, pixel.green, pixel.red};

    // compute pixel offset
    MPI_Offset offset = _compute_pixel_offset(self, x, y);

    // write pixel data to file at appropriate location
    MPI_File_write_at(self->parallel_file, offset, pixel_data, BYTES_PER_PIXEL, MPI_UNSIGNED_CHAR, NULL);
}

/**
 * @brief Initialize parallel .bmp file and store pointer in Bitmap object
 * 
 * @param self Bitmap object
 * @param file_name .bmp file name
 */
void _init_parallel(Bitmap *self, const char *file_name) {
    self->serial_file = NULL;
    MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_WRONLY, MPI_INFO_NULL, &self->parallel_file);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        // create .bmp file headers
        unsigned char* file_header = _create_bmp_file_header(self);
        unsigned char* info_header = _create_bmp_info_header(self);

        // Arguments are file, offset, data, data_size, data_type, status.
        MPI_File_write_at(self->parallel_file, 0, file_header, FILE_HEADER_SIZE, MPI_UNSIGNED_CHAR, NULL);
        MPI_File_write_at(self->parallel_file, FILE_HEADER_SIZE, info_header, INFO_HEADER_SIZE, MPI_UNSIGNED_CHAR, NULL);
    }
}

/**
 * @brief Initialize serial .bmp file and store pointer in Bitmap object
 * 
 * @param self Bitmap object
 * @param file_name .bmp file name
 */
void _init_serial(Bitmap *self, const char *file_name) {
    self->parallel_file = NULL;
    self->serial_file = fopen(file_name, "wb");

    // create .bmp file headers
    unsigned char* file_header = _create_bmp_file_header(self);
    unsigned char* info_header = _create_bmp_info_header(self);

    fwrite(file_header, 1, FILE_HEADER_SIZE, self->serial_file);
    fwrite(info_header, 1, INFO_HEADER_SIZE, self->serial_file);

    // write temp data and padding to file
    unsigned char temp_data[self->width * 3];
    memset(temp_data, 255, self->width * 3); // blank (white) image

    for (int i = 0; i < self->height; i++) {
        fwrite(temp_data, BYTES_PER_PIXEL, self->width, self->serial_file);
        fwrite(PADDING, 1, self->_padding_size, self->serial_file);
    }
}

// Methods
/**
 * @brief Calculate the offset for a given pixel based on its coords.
 *        The formula takes into account both header sizes, the bytes per pixel value
 *        for each pixel and the amount of padding present in each row.
 * 
 * @param x Pixel 'X' coordinate (offset for image plane)
 * @param y Pixel 'Y' coordinate (offest for image plane)
 */
int _compute_pixel_offset(const Bitmap *self, int x, int y) {
    return FILE_HEADER_SIZE + INFO_HEADER_SIZE + (y * (self->width * BYTES_PER_PIXEL + self->_padding_size)) + (x * BYTES_PER_PIXEL);
}

/**
 * @brief Creates array containing .bmp format-required file header based in image specifications
 * 
 * @param self Bitmap object
 */
unsigned char *_create_bmp_file_header(const Bitmap *self) {
    
    // compute full file size
    int file_size = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (BYTES_PER_PIXEL * self->width + self->_padding_size) * self->height;

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
    info_header[4] = (unsigned char)(self->width);
    info_header[5] = (unsigned char)(self->width >> 8);
    info_header[6] = (unsigned char)(self->width >> 16);
    info_header[7] = (unsigned char)(self->width >> 24);
    info_header[8] = (unsigned char)(self->height);
    info_header[9] = (unsigned char)(self->height >> 8);
    info_header[10] = (unsigned char)(self->height >> 16);
    info_header[11] = (unsigned char)(self->height >> 24);
    info_header[12] = (unsigned char)(1);
    info_header[14] = (unsigned char)(BYTES_PER_PIXEL * 8);

    return info_header;
}