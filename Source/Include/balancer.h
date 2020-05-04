#ifndef MBMPI_H
#define MBMPI_H

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<string.h>

#include "args.h"

#ifdef PARALLEL
#include<mpi.h>
long compute_mandelbrot_parallel(const Args *args);
#endif

long compute_mandelbrot_serial(const Args *args);

#endif // MBMPI_H