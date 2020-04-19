#ifndef MANDELBROT_H
#define MANDELBROT_H

#include<stdbool.h>
#include<complex.h>

// BEGIN TYPEDEFS

typedef long double complex MB_Complex;

/**
 * @brief Contains information about a point c that has undergone some number of Mandelbrot set iterations.
 * 
 * Fields are named based on the Mandelbrot set formula z = z_prev^2 + c.
 */
typedef struct {
    MB_Complex c; /** The imaginary number whose Mandelbrot info is contained in this struct. */
    int iters_performed; /** The number of Mandelbrot iterations performed on @c c. */
    int max_iters; /** The maximum number of iterations that could have been performed on @c c. */
    MB_Complex z_final; /** The value of z after @c iters_performed iterations of @c c */
} MB_Point;

/**
 * @brief Represents an RGB color with red, green, and blue integer fields on [0, 255].
 */
typedef struct {
    int red;
    int green;
    int blue;
} MB_Rgb;

/**
 * @brief Represents conversion from an @c MB_Point to an @c MB_Rgb color that can be passed to @c MB_color_of.
 */
enum MB_ColorMap {
    HSV_TO_RGB, /** Uses a normalized iteration count to generate an HSV color, and converts this to an RGB color.  */
    DIRECT_RGB /** Converts a normalized iteration count directly to an RGB color. */
};

typedef enum MB_ColorMap MB_ColorMap;

// END TYPEDEFS

MB_Point MB_iterate_mandelbrot(long double c_real, long double c_img, int iterations);

MB_Rgb MB_color_of(const MB_Point *point, MB_ColorMap conversion);

#endif // MANDELBROT_H