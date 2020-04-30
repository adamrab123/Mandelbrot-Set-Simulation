// Writes pixel data to .bmp file

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
int _write_at_pixel(const Bitmap *self, long x, long y, unsigned char *data, long data_len);
void _write_at(const Bitmap *self, long offset, unsigned char *data, long len_data);
unsigned char *_create_bmp_file_header(const Bitmap *self);
unsigned char *_create_bmp_info_header(const Bitmap *self);

// Public methods
/**
 * @brief Initialize the Btimap object and .bmp output file
 * 
 * @param width Image width
 * @param height Image height
 * @param image_file_name Output file name
 * @param file_type Designates Bitmap as setup for either a @c SERIAL or @c PARALLEL computation environment
 * @return @c Bitmap* The Bitmap object to be output
 */
Bitmap *Bitmap_init(int width, int height, const char *file_name) {
    Bitmap *self = calloc(1, sizeof(Bitmap));

    // set globals
    self->width = width;
    self->height = height;

    // compute padding size (math magic)
    self->_padding_size = (4 - (width * BYTES_PER_PIXEL) % 4) % 4;

    self->_file = NULL;
    bool write_headers = true;

    #ifdef PARALLEL
    MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_WRONLY, MPI_INFO_NULL, self->_file);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    write_headers = (my_rank == 0);
    #else
    self->_file = fopen(file_name, "wb");
    #endif


    if (write_headers) {
        // create .bmp file headers
        unsigned char* file_header = _create_bmp_file_header(self);
        unsigned char* info_header = _create_bmp_info_header(self);

        // Arguments are file, offset, data, data_size, data_type, status.
        _write_at(self, 0, file_header, FILE_HEADER_SIZE);
        _write_at(self, FILE_HEADER_SIZE, info_header, INFO_HEADER_SIZE);

        // write temp data and padding to each file row
        unsigned char temp_data[self->width * BYTES_PER_PIXEL];
        memset(temp_data, 255, self->width * BYTES_PER_PIXEL); // blank (white) image

        for (int i = 0; i < self->height; i++) {
            _write_at_pixel(self, 0, i, temp_data, BYTES_PER_PIXEL * self->width);
            _write_at_pixel(self, self->width, i, PADDING, self->_padding_size);
        }
    }

    return self;
}

/**
 * @brief Bitmap destructor
 * 
 * @param self Bitmap object to be removed from memory
 */
void Bitmap_free(Bitmap *self) {
    #ifdef PARALLEL
    MPI_File_close(self->_file);
    #else
    fclose(self->_file);
    #endif

    free(self);
}

/**
 * @brief Writes passed pixel to the output file using serial C methods
 * 
 * @param self Bitmap object
 * @param pixel Rgb enum containing color data
 * @param x Pixel 'X' coordinate (offset for image plane)
 * @param y Pixel 'Y' coordinate (offset for image plane)
 */
void Bitmap_write_pixel(Bitmap *self, Rgb pixel, long x, long y) {
    unsigned char pixel_data[BYTES_PER_PIXEL] = {pixel.blue, pixel.green, pixel.red};

    _write_at_pixel(self, x, y, pixel_data, sizeof(pixel_data));
}

/**
 * @brief Writes passed pixel rows to the output file using parallel MPI methods
 * 
 * @param self Bitmap object
 * @param pixels Array of pixel rows
 * @param num_rows Number of pixel rows
 * @param start_row 'Y' coordinate of the starting row (offset for image plane)
 */
void Bitmap_write_rows(Bitmap *self, Rgb **pixels, long start_row, long num_rows) {
    // compute padding needed and size of array to be written
    long pixels_data_length = num_rows * ((self->width * BYTES_PER_PIXEL) + self->_padding_size);
    unsigned char pixels_data[pixels_data_length];

    int index = 0;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < self->width; j++) {

            // add pixel to array to be written
            pixels_data[index]      = pixels[i][j].blue;
            pixels_data[index + 1]  = pixels[i][j].green;
            pixels_data[index + 2]  = pixels[i][j].red;

            index += 3;
        }

        // add padding to end of row in array to be written
        for (int k = 0; k < self->_padding_size; k++) {
            pixels_data[index] = PADDING[0];
            index++;
        }
    }

    _write_at_pixel(self, 0, start_row, pixels_data, pixels_data_length);
}

// Private methods

/**
 * @brief Calculate the offset for a given pixel based on its coords.
 *        The formula takes into account both header sizes, the bytes per pixel value
 *        for each pixel and the amount of padding present in each row.
 * 
 * @param x Pixel 'X' coordinate (offset for image plane)
 * @param y Pixel 'Y' coordinate (offset for image plane)
 */
int _write_at_pixel(const Bitmap *self, long x, long y, unsigned char *data, long data_len) {
    long pixel_offset = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (y * (self->width * BYTES_PER_PIXEL + self->_padding_size)) + (x * BYTES_PER_PIXEL);
    _write_at(self, pixel_offset, data, data_len);
}

void _write_at(const Bitmap *self, long offset, unsigned char *data, long len_data) {
    // Both parallel and serial versions seek from the beginning of the file every time.
    #ifdef PARALLEL
        MPI_File_write_at(self->_file,
                            offset,
                            data,
                            len_data,
                            MPI_UNSIGNED_CHAR,
                            NULL);
    #else
        fseek(self->_file, offset, SEEK_SET);
        fwrite(data, len_data, 1, self->_file);
    #endif
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