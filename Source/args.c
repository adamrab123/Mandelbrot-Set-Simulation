#include <errno.h>
#include<ctype.h>
#include<stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <stdio.h>

#include "args.h"

double _parse_double(const char *str, bool *error);
long _parse_long(const char *str, bool *error);
bool _args_valid(Args *self);

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

    // Set all parameters to their default values.
    self->x_min = -2.1;
    self->x_max = 1;
    self->y_min = -1.5;
    self->y_max = 1.5;
    self->step_size = 0.01;
    self->iterations = 100;
    self->output_file = (char *)calloc(100, sizeof(char));
    strcpy(self->output_file, "output.bmp");
    self->block_size = 1;
    self->chunks = 1;

    int c;
    bool parse_error = false;

    while (1) {
        static struct option long_options[] = {
            // These options don’t set a flag.
            // We distinguish them by their indices.
            {"x-min", required_argument, 0, 'x'},
            {"x-max", required_argument, 0, 'X'},
            {"y-min", required_argument, 0, 'y'},
            {"y-max", required_argument, 0, 'Y'},
            {"step-size", required_argument, 0, 's'},
            {"output-file", required_argument, 0, 'o'},
            {"block-size", required_argument, 0, 'b'},
            {"iterations", required_argument, 0, 'i'},
            {"chunks", required_argument, 0, 'c'},
            {0, 0, 0, 0}
        };
        int option_index = 0;

        c = getopt_long(argc, argv, ":x:X:y:Y:s:o:b:i:c:lx",
                        long_options, &option_index);

        // Detect the end of the options.
        if (c == -1) {
            break;
        }

        switch (c)
        {
            case 0: {
                // For long options without a corresponding short option.
                // We do not have any of these, so do nothing.
                break;
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
            case 'b': {
                self->block_size = _parse_long(optarg, &parse_error);
                break;
            }
            case 'c': {
                self->chunks = _parse_long(optarg, &parse_error);
                break;
            }
            case '?': {
                const char *option = argv[optind - 1];
                if (optind == 1) {
                    // Prevent from displaying program name as invalid option.
                    option = argv[optind];
                }

                fprintf(stderr, "Unknown option '%s'\n", option);
                parse_error = true;
                break;
            }
            default: {
                // If a parse error was already encountered, it cannot tell whether required args were provided.
                if (!parse_error) {
                    const char *option = argv[optind - 1];
                    if (optind == 1) {
                        // Prevent from displaying program name as requiring an argument.
                        option = argv[optind];
                    }

                    fprintf(stderr, "Option '%s' requires an argument.\n", option);
                    parse_error = true;
                }
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
        exit(1);
    }

    return self;
}

void Args_free(Args *self) {
    free(self->output_file);
    free(self);
}

// Calculate the width and height of the bitmap image based on the input parameters.
#ifdef __CUDACC__
__host__ __device__
#endif
void Args_get_bitmap_dims(const Args *self, long *num_rows, long *num_cols) {
    *num_rows = round((self->y_max - self->y_min) / self->step_size);
    *num_cols = round((self->x_max - self->x_min) / self->step_size);
}

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

    if (self->block_size <= 0) {
        fprintf(stderr, "Thread block size must be positive.\n");
        args_valid = false;
    }

    if (self->chunks <= 0) {
        fprintf(stderr, "Image chunks per process must be positive.\n");
        args_valid = false;
    }

    return args_valid;
}