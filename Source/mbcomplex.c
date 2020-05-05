#include <math.h>
#include "mbcomplex.h"

/**
 * @brief Creates a MbComplex point representing a complex point from the parameters
 * 
 * @param real the real coordinate
 * @param imag the imaginary coordinate
 *
 * @return result the MbComplex point from the coordinates given
 */
#ifdef __CUDACC__
__host__ __device__
#endif
MbComplex MbComplex_init(Part real, Part imag) {
    MbComplex result;
    result.real = real;
    result.imag = imag;

    return result;
}

/**
 * @brief Computes the sum of 2 MbComplex points
 * 
 * @param num1 a MbComplex point
 * @param num2 a MbComplex point
 *
 * @return result the sum of the 2 complex points as a MbComplex
 */
#ifdef __CUDACC__
__host__ __device__
#endif
MbComplex MbComplex_add(MbComplex num1, MbComplex num2) {
    MbComplex result;
    result.real = num1.real + num2.real;
    result.imag = num1.imag + num2.imag;

    return result;
}

/**
 * @brief Computes the product of 2 MbComplex points
 * 
 * @param num1 a MbComplex point
 * @param num2 a MbComplex point
 *
 * @return result the product of the 2 complex points as a MbComplex
 */
#ifdef __CUDACC__
__host__ __device__
#endif
MbComplex MbComplex_mul(MbComplex num1, MbComplex num2) {
    MbComplex result;
    result.real = (num1.real * num2.real) - (num1.imag * num2.imag);
    result.imag = (num1.real * num2.imag) + (num1.imag * num2.real);

    return result;
}

/**
 * @brief Computes the absolute value of a MbComplex point
 * 
 * @param num a MbComplex point
 *
 * @return the absolute value of the complex points as a Part
 */
#ifdef __CUDACC__
__host__ __device__
#endif
Part MbComplex_abs(MbComplex num) {
    return sqrt(pow(num.real, 2) + pow(num.imag, 2));
}

/**
 * @brief Assigns the @p lvalue to the contents of @p rvalue
 * 
 * @param lvalue a MbComplex point
 * @param rvalue a MbComplex point
 */
#ifdef __CUDACC__
__host__ __device__
#endif
void MbComplex_assign(MbComplex *lvalue, MbComplex rvalue) {
    lvalue->real = rvalue.real;
    lvalue->imag = rvalue.imag;
}