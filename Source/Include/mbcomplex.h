#ifndef MBCOMPLEX_H
#define MBCOMPLEX_H

typedef double Part;

typedef struct {
    Part real;
    Part imag;
} MbComplex;

MbComplex MbComplex_add(const MbComplex *num1, const MbComplex *num2);
MbComplex MbComplex_mul(const MbComplex *num1, const MbComplex *num2);
Part MbComplex_abs(const MbComplex *num);

#endif