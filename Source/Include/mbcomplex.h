#ifndef MBCOMPLEX_H
#define MBCOMPLEX_H

typedef double Part;

typedef struct {
    Part real;
    Part imag;
} MbComplex;

MbComplex MbComplex_init(Part real, Part imag);
MbComplex MbComplex_add(MbComplex num1, MbComplex num2);
MbComplex MbComplex_mul(MbComplex num1, MbComplex num2);
Part MbComplex_abs(MbComplex num);
void MbComplex_assign(MbComplex lvalue, MbComplex rvalue);

#endif