#ifndef COLORMAP_H
#define COLORMAP_H

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

Rgb ColorMap_rgb_based(double norm_iters);
Rgb ColorMap_hsv_based(double norm_iters);

Rgb ColorMap_hsv_to_rgb(double hue, double saturation, double value);

#endif // COLORMAP_H
