#include <errno.h>
#include<ctype.h>
#include<stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <stdio.h>
#include <math.h>

#include "args.h"

double _parse_double(const char *str, bool *error);
long _parse_long(const char *str, bool *error);
bool _args_valid(Args *self);
void _init_defaults(Args *self);

/**
 * @brief Parses the given argc,argv and modifies program parameters 
 * 
 * Uses the getopt function to parse the argv sequence provided from the 
 *      main program call to alter our simulation parameters. Paramters
 *      are set to their defaults before parsing user input.
 *
 * @param argc This is the argc passed in from the original main function
 * @param argv This is the argv passed in from the original main function
 *
 * @return An @c Arguments instance with user provided parameter inputts
 */
Args *Args_init(int argc, char **argv) {
    Args *self = (Args *)malloc(sizeof(Args));

    _init_defaults(self);

    int c;
    bool parse_error = false;

    struct option long_options[] = {
        // These options don’t set a flag.
        // We distinguish them by their indices.
        {"x-min", required_argument, 0, 'x'},
        {"x-max", required_argument, 0, 'X'},
        {"y-min", required_argument, 0, 'y'},
        {"y-max", required_argument, 0, 'Y'},
        {"step-size", required_argument, 0, 's'},
        {"output-file", required_argument, 0, 'o'},
        {"iterations", required_argument, 0, 'i'},
        {"chunks", required_argument, 0, 'c'},
        {"delete-output", no_argument, (int *)&self->delete_output, 'd'},
        #ifdef PARALLEL
        {"block-size", required_argument, 0, 'b'},
        {"time-dir", required_argument, 0, 't'},
        #endif
        {0, 0, 0, 0}
    };

    // : after char means requires arg.
    // :: after char means optional arg.
    // Nothing after char means no arg.
    char option_string[30];
    strcpy(option_string, "x:X:y:Y:s:o:i:c:d");

    #ifdef PARALLEL
    strcat(option_string, "b:t:");
    #endif

    while (true) {
        int option_index = 0;

        c = getopt_long(argc, argv, option_string,
                        long_options, &option_index);

        // Detect the end of the options.
        if (c == -1) {
            break;
        }

        switch (c) {
            case 0: {
                // For long options without a corresponding short option.
                // We do not have any of these.

                // If this option set a flag, we have already taken care of it.
                if (long_options[option_index].flag != 0) {
                    break;
                }
            }
            case 'x': {
                self->x_min = _parse_double(optarg, &parse_error);
                break;
            }
            case 'X': {
                self->x_max = _parse_double(optarg, &parse_error);
                break;
            }
            case 'y': {
                self->y_min = _parse_double(optarg, &parse_error);
                break;
            }
            case 'Y': {
                self->y_max = _parse_double(optarg, &parse_error);
                break;
            }
            case 's': {
                self->step_size = _parse_double(optarg, &parse_error);
                break;
            }
            case 'i': {
                self->iterations = _parse_long(optarg, &parse_error);
                break;
            }
            case 'o': {
                // No verification for file name here, it will be done when the file is opened.
                free(self->output_file);
                self->output_file = (char *)calloc(strlen(optarg) + 1, sizeof(char));
                strcpy(self->output_file, optarg);
                break;
            }
            case 'c': {
                self->chunks = _parse_long(optarg, &parse_error);
                break;
            }
            case 'd': {
                self->delete_output = true;
                break;
            }

            #ifdef PARALLEL
            case 'b': {
                self->block_size = _parse_long(optarg, &parse_error);
                break;
            }
            case 't': {
                self->time_dir= (char *)calloc(strlen(optarg) + 1, sizeof(char));
                strcpy(self->time_dir, optarg);
                break;
            }
            #endif

            default: {
                parse_error = true;
            }
        }
    }

    // Dispaly errors for extra command line arguments (not options beginning with - or --).
    if (optind < argc) {
        while (optind < argc) {
            printf("Unknown argument '%s'\n", argv[optind]);
            optind++;
        }

        parse_error = true;
    }

    // Wait until the end to exit so all bad command line arguments can be processed and displayed.
    if (parse_error || !_args_valid(self)) {
        exit(EXIT_FAILURE);
    }

    return self;
}

/**
 * @brief Frees the memory used in an @c Arguments structure
 * 
 * @param self @c Arguments structure to be freed
 */
void Args_free(Args *self) {
    free(self->output_file);

    #ifdef PARALLEL
    free(self->time_dir);
    #endif

    free(self);
}

/**
 * @brief Writes the number of rows and cols to input variables 
 * based on data present in nput @c Arguments structure
 * 
 * @param self @c Arguments structure (y_max, y_min, x_max, x_min, and step_size are used)
 * @param num_rows Variable to have row count written to
 * @param num_cols Variable to have column count written to
 */
#ifdef __CUDACC__
__host__ __device__
#endif
void Args_get_bitmap_dims(const Args *self, long *num_rows, long *num_cols) {
    *num_rows = round((self->y_max - self->y_min) / self->step_size);
    *num_cols = round((self->x_max - self->x_min) / self->step_size);
}

/**
 * @brief 
 * 
 * @param self @c Arguments structure
 * @param row Bitmap point row value
 * @param col Bitmap point column value
 * @param c_real Variable to have converted row value written to
 * @param c_imag Variable to have converted column value written to
 */
#ifdef __CUDACC__
__host__ __device__
#endif
void Args_bitmap_to_complex(const Args *self, long row, long col, double *c_real, double *c_imag) {
    *c_real = (col * self->step_size) + self->x_min;
    *c_imag = self->y_max - (row * self->step_size);
}

double _parse_double(const char *str, bool *error) {
    char *end_ptr;
    double result = strtod(str, &end_ptr);

    errno = 0;
    if (errno == ERANGE) {
        fprintf(stderr, "Number %s is out of bounds.\n", str);
        *error = true;
    }
    else if (end_ptr == str) {
        fprintf(stderr, "%s is not a valid number.\n", str);
        *error = true;
    }

    return result;
}

long _parse_long(const char *str, bool *error) {
    char *end_ptr;
    long result = strtol(str, &end_ptr, 10);

    errno = 0;
    if (errno == ERANGE) {
        fprintf(stderr, "Integer %s is out of bounds.\n", str);
        *error = true;
    }
    else if (end_ptr == str) {
        fprintf(stderr, "%s is not a valid integer.\n", str);
        *error = true;
    }

    return result;
}

/**
 * @brief Check if input arguments are valid
 * 
 * @param self @c Arguments structure
 * @return true
 * @return false
 */
bool _args_valid(Args *self) {
    double x_range = self->x_max - self->x_min;
    double y_range = self->y_max - self->y_min;

    bool args_valid = true;

    if (x_range <= 0) {
        fprintf(stderr, "Minimum x value must be less than maximum x value.\n");
        args_valid = false;
    }

    if (y_range <= 0) {
        fprintf(stderr, "Minimum y value must be less than maximum y value.\n");
        args_valid = false;
    }

    if (self->step_size <= 0) {
        fprintf(stderr, "Step size must be greater than 0.\n");
        args_valid = false;
    }

    if (self->step_size >= x_range) {
        fprintf(stderr, "Step size cannot be larger than the range of x values.\n");
        args_valid = false;
    }

    if (self->step_size >= y_range) {
        fprintf(stderr, "Step size cannot be larger than the range of y values.\n");
        args_valid = false;
    }

    if (self->iterations <= 0) {
        fprintf(stderr, "Maximum number of iterations must be positive.\n");
        args_valid = false;
    }

    if (self->chunks <= 0) {
        fprintf(stderr, "Compute chunks per process must be positive.\n");
        args_valid = false;
    }

    #ifdef PARALLEL
    if (self->block_size <= 0) {
        fprintf(stderr, "Thread block size must be positive.\n");
        args_valid = false;
    }
    #endif

    return args_valid;
}

/**
 * @brief Initialize @c Arguments structure with default values
 * 
 * @param self @c Arguments structure
 */
void _init_defaults(Args *self) {
    self->x_min = -2.1;
    self->x_max = 1;
    self->y_min = -1.5;
    self->y_max = 1.5;
    self->step_size = 0.01;
    self->iterations = 100;
    self->output_file = (char *)calloc(100, sizeof(char));
    strcpy(self->output_file, "output.bmp");
    self->chunks = 1;
    self->delete_output = false;
    #ifdef PARALLEL
    self->block_size = 64;
    self->time_dir = NULL;
    #endif
}