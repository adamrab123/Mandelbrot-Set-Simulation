#ifndef MBCOMPLEX_H
#define MBCOMPLEX_H

typedef double Part;

typedef struct {
    Part real;
    Part imag;
} MbComplex;

#ifdef __CUDACC__
__host__ __device__
#endif
MbComplex MbComplex_init(Part real, Part imag);
#ifdef __CUDACC__
__host__ __device__
#endif
MbComplex MbComplex_add(MbComplex num1, MbComplex num2);
#ifdef __CUDACC__
__host__ __device__
#endif
MbComplex MbComplex_mul(MbComplex num1, MbComplex num2);
#ifdef __CUDACC__
__host__ __device__
#endif
Part MbComplex_abs(MbComplex num);
#ifdef __CUDACC__
__host__ __device__
#endif
void MbComplex_assign(MbComplex *lvalue, MbComplex rvalue);

#endif