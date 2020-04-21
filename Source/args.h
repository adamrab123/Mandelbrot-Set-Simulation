#ifndef ARGS_H
#define ARGS_H
  
#include <stdio.h>  
#include <unistd.h>  
#include <stdlib.h>
#include <string.h>

/**
 * @brief Contains information about the optional command 
 *      line parameters passed into our simulation
 * 
 */
typedef struct {
    int x_min;
    int x_max;
    int y_min;
    int y_max;
    float steps;
    int iterations;
    char* output_file;
    int block_size;
} Arguments;

  
Arguments getArgs(int argc, char *argv[]);

#endif // ARGS_H
