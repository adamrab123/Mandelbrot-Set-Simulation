#include<stdlib.h>
#include<complex.h>
#include<stdbool.h>
#include<math.h>

#include "mandelbrot.h"

MB_Rgb _MB_rgb_color(const MB_Point *point);
MB_Rgb _MB_hsv_to_rgb(int hue, double saturation, double value);
MB_Rgb _MB_hsv_color(const MB_Point *point);
double _MB_normalized_color(const MB_Point *point);

MB_Point MB_is_mandelbrot(long double c_real, long double c_img, int iterations) {
    MB_Complex c = c_real + c_img * I;
    MB_Complex z = 0;
    MB_Complex z_next = 0;

    bool diverged = false;
    int escape_radius = 2;

    int iters_performed = 0;
    while (iters_performed < iterations && !diverged) {
        z_next = cpowl(z, 2) + c;
        z = z_next;

        // cabsl(z) is the distance from z to the origin.
        diverged = cabsl(z) > escape_radius;

        iters_performed++;
    }

    MB_Point info;
    info.final_z = z;
    info.iters_performed = iters_performed;
    info.max_iters = iterations;
    info.c = c;

    return info;
}

MB_Rgb MB_color_of(const MB_Point *point, MB_ColorMap conversion) {
    MB_Rgb color;

    if (conversion == HSV_TO_RGB) {
        color = _MB_hsv_color(point);
    }
    else if (conversion == DIRECT_RGB) {
        color = _MB_rgb_color(point);
    }

    return color;
}

/*
One possible way to map escape iterations to color is to scale the values and treat the first third as "x 0 0", the
next third as "255 x 0" and the final third as "255 255 x". This will produce a gradient from black to red to yellow
to white.
*/
MB_Rgb _MB_rgb_color(const MB_Point *point) {
    // Produces number on range (0, 3).
    double color_fraction = _MB_normalized_color(point) * 3;

    MB_Rgb color;

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

/*
HSV colors:
HUE
    Red falls between 0 and 60 degrees.
    Yellow falls between 61 and 120 degrees.
    Green falls between 121-180 degrees.
    Cyan falls between 181-240 degrees.
    Blue falls between 241-300 degrees.
    Magenta falls between 301-359 degrees.

SATURATION
Saturation describes the amount of gray in a particular color, from 0 to 100 percent (or 0 to 1).
0 is grey, 1 is a primary color.

VALUE (OR BRIGHTNESS)
Value works in conjunction with saturation and describes the brightness or intensity of the color, from 0-100 percent
(or 0 to 1).
0 is black, 1 is brightest.

// Compute the ratio of possible iterations before it diverged, and take that percent of the hue value.
hue = (mu // MAX_ITER) * 360
// No grey.
saturation = 1
// Black if in set, full brightness if not.
value = 1 if m < MAX_ITER else 0
*/
MB_Rgb _MB_hsv_color(const MB_Point *point) {
    double color_percent = _MB_normalized_color(point);

    int hue = floor(color_percent * 360);
    int saturation = 1;
    int value = 1;

    // If the point did not diverge after all iterations finished, it is in the set.
    // Color it black.
    if (point->iters_performed == point->max_iters) {
        value = 0;
    }

    return _MB_hsv_to_rgb(hue, saturation, value);
}

// hue on [0, 359]
// others on [0, 1]
MB_Rgb _MB_hsv_to_rgb(int hue, double saturation, double value) {
    double c = value * saturation;
    double x = c * (1 - abs(hue / 60 % 2 - 1));
    double m = value - c;

    // c, x, 0 become c1, c2, c3.
    int c1 = (c + m) * 255;
    int c2 = (x + m) * 255;
    int c3 = (m) * 255;

    MB_Rgb color;

    int norm_hue = hue / 60;

    if (norm_hue == 0) {
        color.red = c1;
        color.green = c2;
        color.blue = c3;
    }
    else if (norm_hue == 1) {
        color.red = c2;
        color.green = c1;
        color.blue = c3;
    }
    else if (norm_hue == 2) {
        color.red = c3;
        color.green = c1;
        color.blue = c2;
    }
    else if (norm_hue == 3) {
        color.red = c3;
        color.green = c2;
        color.blue = c1;
    }
    else if (norm_hue == 4) {
        color.red = c2;
        color.green = c3;
        color.blue = c1;
    }
    else if (norm_hue == 5) {
        color.red = c1;
        color.green = c3;
        color.blue = c2;
    }

    return color;
}

// Produces numbers on range (0, 1).
// Higher iterations performed before escaping gives higher number.
double _MB_normalized_color(const MB_Point *point) {
    // Smooth is on (0, max_iter).
    int smooth = point->iters_performed + 1 - logl(logl(cabsl(point->final_z))) / logl(2);
    // Normalize smooth to be on (0, 1).
    return smooth / point->max_iters;
}
