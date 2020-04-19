#ifndef MANDELBROT_H
#define MANDELBROT_H

#include<stdbool.h>
#include<complex.h>

// BEGIN TYPEDEFS

typedef long double complex MB_Complex;

typedef struct {
    int iters_performed;
    int max_iters;
    MB_Complex final_z;
    MB_Complex c;
} MB_Point;

typedef struct {
    int red;
    int green;
    int blue;
} MB_Rgb;

enum MB_ColorMap { HSV_TO_RGB, DIRECT_RGB };
typedef enum MB_ColorMap MB_ColorMap;

// END TYPEDEFS

MB_Point MB_is_mandelbrot(long double c_real, long double c_img, int iterations);

MB_Rgb MB_color_of(const MB_Point *point, MB_ColorMap conversion);

#endif // MANDELBROT_H