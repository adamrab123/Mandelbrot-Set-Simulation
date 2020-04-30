#include<stdlib.h>
#include<math.h>

#include "colormap.h"

// Rgb _hsv_to_rgb(int hue, double saturation, double value);

/**
 * @brief Generates an RGB color directly from an @c MB_Point.
 *
 * As the number of performed iterations on a point increases, this should produce a color gradient from white to
 * yellow to red to black.
 *
 * @param point The Mandelbrot set information for a point that will be used to calculate its color.
 *
 * @return An RGB representation of @p point.
 */
#ifdef __CUDACC__
__host__ __device__
#endif
Rgb ColorMap_rgb_based(double norm_iters) {
    // Produces number on range (0, 3).
    double color_fraction = norm_iters * 3;

    Rgb color;

    if (color_fraction < 1) {
        color.red = 255;
        color.green = 255;
        color.blue = (1 - color_fraction) * 255;
    }
    else if (color_fraction >= 1 && color_fraction < 2) {
        color.red = 255;
        color.green = (2 - color_fraction) * 255;
        color.blue = 0;
    }
    else {
        // color_fraction on [2, 3).
        color.red = (3 - color_fraction) * 255;
        color.green = 0;
        color.blue = 0;
    }

    return color;
}

/**
 * @brief Generates an HSV color from @p point, and then converts it to an RGB color.
 *
 * Sets the hue to be a percentage of its maximum possible value, determined by the normalized iteration count.
 * Sets the saturation as always one.
 * Sets the value as 1 if @p point diverged, 0 otherwise.
 *
 * @param point The Mandelbrot set information for a point that will be used to calculate its color.
 *
 * @return An RGB representation of @p point converted from an HSV value.
 */
#ifdef __CUDACC__
__host__ __device__
#endif
Rgb ColorMap_hsv_based(double norm_iters) {
    // Hues on [0, 360].
    // Seems like any value below 60 is converted to solid red, always the same shade.
    int hue360_min = 0;
    int hue360_max = 360;

    // Hues on [0, 1] using the above values.
    double hue_min = hue360_min / 360.0;
    double hue_max = hue360_max / 360.0;

    double hue = hue_min + (hue_max - hue_min) * norm_iters;
    int saturation = 1;
    int value = 1;

    return ColorMap_hsv_to_rgb(hue, saturation, value);
}

/**
 * @brief Converts an HSV color to an RGB color.
 *
 * This code was ported from python's syscolor.hsv_to_rgb() method.
 * Original source code found here: https://github.com/python/cpython/blob/master/Lib/colorsys.py
 *
 * @param hue The hue parameter of the color to convert, on the range [0, 1].
 * @param saturation The saturation parameter of the color to convert, on the range [0, 1].
 * @param value The value (or brightness) parameter of the color to convert, on the range [0, 1].
 *
 * @return An RGB color representation that is identical to the provided HSV color representation.
 */
#ifdef __CUDACC__
__host__ __device__
#endif
Rgb ColorMap_hsv_to_rgb(double hue, double saturation, double value) {
    double d_red = 0;
    double d_green = 0;
    double d_blue = 0;

    // If saturation is basically zero.
    if (fabs(saturation) < 0.001) {
        d_red = value;
        d_green = value;
        d_blue = value;
    }
    else {
        // Truncate result.
        int i = floor(hue * 6.0);
        double f = (hue*6.0) - i;
        double p = value*(1.0 - saturation);
        double q = value*(1.0 - saturation*f);
        double t = value*(1.0 - saturation*(1.0-f));

        i = i % 6;

        if (i == 0) {
            d_red = value;
            d_green = t;
            d_blue = p;
            // return value, t, p
        }
        else if (i == 1) {
            d_red = q;
            d_green = value;
            d_blue = p;
            // return q, value, p
        }
        else if (i == 2) {
            d_red = p;
            d_green = value;
            d_blue = t;
            // return p, value, t
        }
        else if (i == 3) {
            d_red = p;
            d_green = q;
            d_blue = value;
            // return p, q, value
        }
        else if (i == 4) {
            d_red = t;
            d_green = p;
            d_blue = value;
            // return t, p, value
        }
        else if (i == 5) {
            d_red = value;
            d_green = p;
            d_blue = q;
            // return value, p, q
        }
    }

    Rgb result;
    result.red = floor(d_red * 255);
    result.green = floor(d_green * 255);
    result.blue = floor(d_blue * 255);
    return result;
}
