#include <math.h>
#include "mbcomplex.h"

MbComplex MbComplex_init(Part real, Part imag) {
    MbComplex result;
    result.real = real;
    result.imag = imag;

    return result;
}

MbComplex MbComplex_add(MbComplex num1, MbComplex num2) {
    MbComplex result;
    result.real = num1.real + num2.real;
    result.imag = num1.imag + num2.imag;

    return result;
}

MbComplex MbComplex_mul(MbComplex num1, MbComplex num2) {
    MbComplex result;
    result.real = (num1.real * num2.real) - (num1.imag * num2.imag);
    result.imag = (num1.real * num2.imag) + (num1.imag * num2.real);

    return result;
}

Part MbComplex_abs(MbComplex num) {
    return sqrtf(pow(num.real, 2) + pow(num.imag, 2));
}

void MbComplex_assign(MbComplex lvalue, MbComplex rvalue) {
    lvalue.real = rvalue.real;
    lvalue.imag = rvalue.imag;
}