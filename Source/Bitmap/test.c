#include <stdio.h>
#include "bitmap.h"

int main() {

    // generate sample image with primary pixel color gradient
    int height = 341;
    int width = 753;

    unsigned char image[height][width][3];

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {

            // create pixel primary color gradient
            image[i][j][2] = (unsigned char)((double) i / height * 255);                 // red
            image[i][j][1] = (unsigned char)((double) j / width * 255);                  // green
            image[i][j][0] = (unsigned char)(((double) i + j) / (height + width) * 255); // blue
        }
    }

    write_array_to_bitmap((unsigned char *)image, height, width);
    printf("Image generated!\r\n");

    return 0;
}
