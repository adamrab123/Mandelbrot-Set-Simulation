// Writes pixel data to .bmp file

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include "bitmap.h"

// Constants
const int bytes_per_pixel = 3; // for 3 pixel colors: blue, green, red
const int file_header_size = 14; // format-required
const int info_header_size = 40; // format-required

const char *image_file_name = "output.bmp";
const unsigned char padding[3] = {0,0,0}; // .bmp format padding array

// Globals
int _height;
int _width;
int _padding_size;

// Methods
/**
 * @brief Calculate the offset for a given pixel based on its coords.
 *        The formula takes into account both header sizes, the bytes per pixel value
 *        for each pixel and the amount of padding present in each row.
 * 
 * @param y The distance on the y-axis that the input pixel sits on the image
 * 
 * @param x The distance on the x-axis that the input pixel sits on the image
 */
int _compute_pixel_offset(int y, int x) {
    return file_header_size + info_header_size + (y * (_width * bytes_per_pixel + _padding_size)) + (x * bytes_per_pixel);
}

/**
 * @brief Creates array containing .bmp format-required file header based in image specifications
 */
unsigned char *_create_bmp_file_header() {
    
    // compute full file size
    int file_size = file_header_size + info_header_size + (bytes_per_pixel * _width + _padding_size) * _height;

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
    file_header[10] = (unsigned char)(file_header_size + info_header_size);

    return file_header;
}

/**
 * @brief Creates array containing .bmp format-required info header based in image specifications
 */
unsigned char *_create_bmp_info_header() {

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

    info_header[0] = (unsigned char)(info_header_size);
    info_header[4] = (unsigned char)(_width);
    info_header[5] = (unsigned char)(_width >> 8);
    info_header[6] = (unsigned char)(_width >> 16);
    info_header[7] = (unsigned char)(_width >> 24);
    info_header[8] = (unsigned char)(_height);
    info_header[9] = (unsigned char)(_height >> 8);
    info_header[10] = (unsigned char)(_height >> 16);
    info_header[11] = (unsigned char)(_height >> 24);
    info_header[12] = (unsigned char)(1);
    info_header[14] = (unsigned char)(bytes_per_pixel * 8);

    return info_header;
}

/**
 * @brief Initialize the .bmp output file with a given height and width
 * 
 * @param height The image height in pixels
 * 
 * @param width The image width in pixels
 */
void init_output_file(int height, int width) {

    // set globals
    _height = height;
    _width = width;

    // compute padding size (math magic)
    _padding_size = (4 - (width * bytes_per_pixel) % 4) % 4;

    // create .bmp file headers
    unsigned char* fileHeader = _create_bmp_file_header();
    unsigned char* infoHeader = _create_bmp_info_header();

    FILE *f = fopen(image_file_name, "wb");

    fwrite(fileHeader, 1, file_header_size, f);
    fwrite(infoHeader, 1, info_header_size, f);

    // write temp data and padding to file
    unsigned char temp_data[_width * 3];
    memset(temp_data, 255, _width * 3); // blank (white) image

    for (int i = 0; i < _height; i++) {
        fwrite(temp_data, bytes_per_pixel, _width, f);
        fwrite(padding, 1, _padding_size, f);
    }

    fclose(f);
}

/**
 * @brief Writes passed pixel to the output file using sequential C methods
 * 
 * @param pixel Array of size 3 containing values for each color
 * 
 * @param y The distance on the y-axis that the input pixel sits on the image
 * 
 * @param x The distance on the x-axis that the input pixel sits on the image
 */
void write_pixel_to_file_sequential(unsigned char *pixel, int y, int x) {

    FILE *f = fopen(image_file_name, "rb+");

    // compute pixel offset
    int offset = _compute_pixel_offset(y, x);

    // write pixel data to file at appropriate location
    fseek(f, offset, SEEK_SET);
    fwrite(pixel, bytes_per_pixel, 1, f);

    fclose(f);
}

/**
 * @brief Writes passed pixel to the output file using parallel MPI methods
 * 
 * @param pixel Array of size 3 containing values for each color
 * 
 * @param y The distance on the y-axis that the input pixel sits on the image
 * 
 * @param x The distance on the x-axis that the input pixel sits on the image
 */
void write_pixel_to_file_parallel(unsigned char *pixel, int y, int x) {

    MPI_File f;
    MPI_File_open(MPI_COMM_SELF, image_file_name, MPI_MODE_WRONLY, MPI_INFO_NULL, &f);

    // compute pixel offset
    MPI_Offset offset = _compute_pixel_offset(y, x);

    // write pixel data to file at appropriate location
    MPI_File_write_at(f, offset, pixel, bytes_per_pixel, MPI_UNSIGNED_CHAR, NULL);

    MPI_File_close(&f);
}
