#ifndef COLORMAP_H
#define COLORMAP_H

#include "rgb.h"

Rgb ColorMap_rgb_based(double norm_iters);
Rgb ColorMap_hsv_based(double norm_iters);

Rgb ColorMap_hsv_to_rgb(double hue, double saturation, double value);

#endif // COLORMAP_H
