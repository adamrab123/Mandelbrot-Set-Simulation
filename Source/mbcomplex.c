#include <math.h>
#include "mbcomplex.h"


MbComplex MbComplex_add(const MbComplex *num1, const MbComplex *num2) {
    MbComplex result;
    result.real = num1->real + num2->real;
    result.imag = num1->imag + num2->imag;

    return result;
}


MbComplex MbComplex_mul(const MbComplex *num1, const MbComplex *num2) {
    MbComplex result;
    result.real = (num1->real * num2->real) - (num1->imag * num2->imag);
    result.imag = (num1->real * num2->imag) + (num1->imag * num2->real);

    return result;
}

Part MbComplex_abs(const MbComplex *num) {
    return sqrtf(powf(num->real, 2) + powf(num->imag, 2));
}