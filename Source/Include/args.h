#ifndef ARGS_H
#define ARGS_H

/**
 * @brief Contains information about the optional command 
 *      line parameters passed into our simulation
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

#endif // ARGS_H