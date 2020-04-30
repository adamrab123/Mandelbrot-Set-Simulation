#include <math.h>
#include "mbcomplex.h"

#ifdef __CUDACC__
__host__ __device__
#endif
MbComplex MbComplex_init(Part real, Part imag) {
    MbComplex result;
    result.real = real;
    result.imag = imag;

    return result;
}

#ifdef __CUDACC__
__host__ __device__
#endif
MbComplex MbComplex_add(MbComplex num1, MbComplex num2) {
    MbComplex result;
    result.real = num1.real + num2.real;
    result.imag = num1.imag + num2.imag;

    return result;
}

#ifdef __CUDACC__
__host__ __device__
#endif
MbComplex MbComplex_mul(MbComplex num1, MbComplex num2) {
    MbComplex result;
    result.real = (num1.real * num2.real) - (num1.imag * num2.imag);
    result.imag = (num1.real * num2.imag) + (num1.imag * num2.real);

    return result;
}

#ifdef __CUDACC__
__host__ __device__
#endif
Part MbComplex_abs(MbComplex num) {
    return sqrtf(pow(num.real, 2) + pow(num.imag, 2));
}

#ifdef __CUDACC__
__host__ __device__
#endif
void MbComplex_assign(MbComplex lvalue, MbComplex rvalue) {
    lvalue.real = rvalue.real;
    lvalue.imag = rvalue.imag;
}