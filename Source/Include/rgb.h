#ifndef RGB_H
#define RGB_H

/**
 * @brief Represents an RGB color with red, green, and blue integer fields on [0, 255].
 */
typedef struct {
    unsigned char blue;
    unsigned char green;
    unsigned char red;
} Rgb;

static const Rgb RGB_WHITE = {255, 255, 255};
static const Rgb RGB_BLACK = {0, 0, 0};

#endif // RGB_H