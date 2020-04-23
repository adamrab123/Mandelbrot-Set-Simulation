#include<stdlib.h>
#include<complex.h>
#include<stdbool.h>
#include<math.h>

#include "mandelbrot.h"

Rgb _MB_rgb_color(const MB_Point *point);
Rgb _MB_hsv_to_rgb(int hue, double saturation, double value);
Rgb _MB_hsv_color(const MB_Point *point);
double _MB_normalized_iterations(const MB_Point *point);

/**
 * @brief Performs @p iterations number of mandelbrot set iterations on the imaginary number c given by @p c_real and @c
 *      c_complex, and returns the resulting information.
 * 
 * @param c_real The real part of the number c to calculate Mandelbrot set information for.
 * @param c_img The imaginary part of the number c to calculate Mandelbrot set information for.
 * @param iterations The number of Mandelbrot set iterations to perform on c.
 *
 * @return An @c MB_Point instance containing information about the resulting iterations.
 */
MB_Point MB_iterate_mandelbrot(long double c_real, long double c_img, int iterations) {
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
    info.z_final = z;
    info.iters_performed = iters_performed;
    info.max_iters = iterations;
    info.c = c;

    return info;
}

/**
 * @brief Returns an RGB color representation of @p point, using the specified conversion method to generate the color.
 * 
 * @param point The point whose Mandelbrot set information will be converted into a color.
 * @param conversion The method to generate the color.
 *
 * @return An RGB color representation of @p point using its Mandelbrot set information.
 */
Rgb MB_color_of(const MB_Point *point, MB_ColorMap conversion) {
    Rgb color;

    if (conversion == HSV_TO_RGB) {
        color = _MB_hsv_color(point);
    }
    else if (conversion == DIRECT_RGB) {
        color = _MB_rgb_color(point);
    }

    return color;
}

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
Rgb _MB_rgb_color(const MB_Point *point) {
    // Produces number on range (0, 3).
    double color_fraction = _MB_normalized_iterations(point) * 3;

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
Rgb _MB_hsv_color(const MB_Point *point) {
    double color_percent = _MB_normalized_iterations(point);

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

/**
 * @brief Converts an HSV color to an RGB color.
 * 
 * @param hue The hue parameter of the color to convert, on the range [0, 359].
 * @param saturation The saturation parameter of the color to convert, on the range [0, 1].
 * @param value The value (or brightness) parameter of the color to convert, on the range [0, 1].
 *
 * @return An RGB color representation that is identical to the provided HSV color representation.
 */
Rgb _MB_hsv_to_rgb(int hue, double saturation, double value) {
    double c = value * saturation;
    double x = c * (1 - abs(hue / 60 % 2 - 1));
    double m = value - c;

    // c, x, 0 become c1, c2, c3.
    int c1 = (c + m) * 255;
    int c2 = (x + m) * 255;
    int c3 = (m) * 255;

    Rgb color;

    int norm_hue = hue / 60;

    if (norm_hue == 0) {
        // Red hue.
        color.red = c1;
        color.green = c2;
        color.blue = c3;
    }
    else if (norm_hue == 1) {
        // Yellow hue.
        color.red = c2;
        color.green = c1;
        color.blue = c3;
    }
    else if (norm_hue == 2) {
        // Green hue.
        color.red = c3;
        color.green = c1;
        color.blue = c2;
    }
    else if (norm_hue == 3) {
        // Cyan hue.
        color.red = c3;
        color.green = c2;
        color.blue = c1;
    }
    else if (norm_hue == 4) {
        // Blue hue.
        color.red = c2;
        color.green = c3;
        color.blue = c1;
    }
    else if (norm_hue == 5) {
        // Magenta hue.
        color.red = c1;
        color.green = c3;
        color.blue = c2;
    }

    return color;
}

/**
 * @brief Computes the noormalized iteration count of @p point.
 *
 * Given the number of iterations performed and maximum possible iterations that could have been performed on @p point,
 * generates a value on the range (0, 1), where a higher number means more iterations were performed before the point
 * diverged.
 *
 * @param point The point whose Mandelbrot iteration information will be used in the computation.
 *
 * @return A value on the range (0, 1).
 */
double _MB_normalized_iterations(const MB_Point *point) {
    // Smooth is on (0, max_iter).
    int smooth = point->iters_performed + 1 - logl(logl(cabsl(point->z_final))) / logl(2);
    // Normalize smooth to be on (0, 1).
    return (double)smooth / point->max_iters;
}

/**
 * @brief Determines whether @p point diverged after its maximum number of iterations of the Mandlebrot set calculation.
 *
 * @param point The point to determine divergence for.
 *
 * @return @c true if @p point diverged, @c false otherwise.
 */
bool MB_diverged(const MB_Point *point) {
    return point->iters_performed < point->max_iters;
}
