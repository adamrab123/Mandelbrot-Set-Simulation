// Writes passed 1D C array to .bmp file

#include <stdio.h>
#include "bitmap.h"

// Constants
const int bytes_per_pixel = 3; // for 3 pixel colors: red, green. blue
const int file_header_size = 14; // format-required
const int info_header_size = 40; // format-required

const char *image_file_name = "output.bmp";

// Methods
/**
 * @brief Creates array containing .bmp format-required file header based in image specifications
 * 
 * @param height The final image pixel height
 * 
 * @param width The final image pixel width
 * 
 * @param padding_size The amount of padding pixels
 */
unsigned char *_create_bmp_file_header(int height, int width, int padding_size) {
    
    // compute full file size
    int file_size = file_header_size + info_header_size + (bytes_per_pixel * width + padding_size) * height;

    // set file header (all from file format guidelines)
    static unsigned char file_header[] = {
        0,0, /// signature
        0,0,0,0, /// image file size in bytes
        0,0,0,0, /// reserved
        0,0,0,0, /// start of pixel array
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
 * 
 * @param height The final image pixel height
 * 
 * @param width The final image pixel width
 */
unsigned char *_create_bmp_info_header(int height, int width) {

    /// set info header (all from file format guidelines)
    static unsigned char info_header[] = {
        0,0,0,0, /// header size
        0,0,0,0, /// image width
        0,0,0,0, /// image height
        0,0, /// number of color planes
        0,0, /// bits per pixel
        0,0,0,0, /// compression
        0,0,0,0, /// image size
        0,0,0,0, /// horizontal resolution
        0,0,0,0, /// vertical resolution
        0,0,0,0, /// colors in color table
        0,0,0,0, /// important color count
    };

    info_header[0] = (unsigned char)(info_header_size);
    info_header[4] = (unsigned char)(width);
    info_header[5] = (unsigned char)(width >> 8);
    info_header[6] = (unsigned char)(width >> 16);
    info_header[7] = (unsigned char)(width >> 24);
    info_header[8] = (unsigned char)(height);
    info_header[9] = (unsigned char)(height >> 8);
    info_header[10] = (unsigned char)(height >> 16);
    info_header[11] = (unsigned char)(height >> 24);
    info_header[12] = (unsigned char)(1);
    info_header[14] = (unsigned char)(bytes_per_pixel * 8);

    return info_header;
}

/**
 * @brief Takes as input a 1D integer array (and image details) and writes to output.bmp
 * 
 * @param height The final image pixel height
 * 
 * @param width The final image pixel width
 * 
 */
void write_array_to_bitmap(unsigned char *data, int height, int width) {

    // .bmp format padding array
    unsigned char padding[3] = {0, 0, 0};

    // compute padding size (math magic)
    int padding_size = (4 - (width * bytes_per_pixel) % 4) % 4;

    // create .bmp file headers
    unsigned char* fileHeader = _create_bmp_file_header(height, width, padding_size);
    unsigned char* infoHeader = _create_bmp_info_header(height, width);

    FILE* f = fopen(image_file_name, "wb");

    fwrite(fileHeader, 1, file_header_size, f);
    fwrite(infoHeader, 1, info_header_size, f);

    // write image data to file
    for (int i = 0; i < height; i++) {
        fwrite(data + (i * width * bytes_per_pixel), bytes_per_pixel, width, f);
        fwrite(padding, 1, padding_size, f);
    }

    fclose(f);
}
