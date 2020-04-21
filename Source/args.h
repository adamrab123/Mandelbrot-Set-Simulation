// Program to illustrate the getopt() 
// function in C 
  
#include <stdio.h>  
#include <unistd.h>  
#include <stdlib.h>
#include <string.h>

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
    