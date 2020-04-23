#include <stdio.h>
#include "bitmap.h"

int main() {

    // generate sample image with primary pixel color gradient
    int width = 753;
    int height = 341;

    // init image file and container
    Bitmap *bitmap = Bitmap_init(width, height, "output.bmp", SERIAL);

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {

            Rgb pixel;

            // create pixel primary color gradient
            pixel.red = (unsigned char)((double) i / height * 255);                  // red
            pixel.green = (unsigned char)((double) j / width * 255);                 // green
            pixel.blue = (unsigned char)(((double) i + j) / (height + width) * 255); // blue

            // add pixel data to image
            Bitmap_write_pixel_serial(bitmap, pixel, j, i);
        }
    }
    
    printf("Image generated!\n");

    return 0;
}
