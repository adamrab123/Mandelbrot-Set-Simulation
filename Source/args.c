#include<ctype.h>
#include<stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <stdio.h>

#include "args.h"

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
                self->x_min = atof(optarg);
                break;
            }
            case 'X': {
                self->x_max = atof(optarg);
                break;
            }
            case 'y': {
                self->y_min = atof(optarg);
                break;
            }
            case 'Y': {
                self->y_max = atof(optarg);
                break;
            }
            case 's': {
                // The steps conversion requires adding a null byte to the end of the array before converting
                int length = sizeof(optarg) / sizeof(optarg[0]);
                char *word = (char*)calloc(length + 1, sizeof(char));
                strcpy(word, optarg);
                word[length] = '\0';
                self->step_size = atof(word);
                break;
            }
            case 'i': {
                self->iterations = atof(optarg);
                break;
            }
            case 'o': {
                self->output_file = optarg;
                break;
            }
            case 'b': {
                self->block_size = atof(optarg);
                break;
            }
            case 'c': {
                self->chunks = atof(optarg);
                break;
            }
            case '?': {
                // if (optopt == 'c') {
                //     fprintf(stderr, "Option -%c requires an argument\n", optopt);
                // }
                // else if (isprint(optopt)) {
                //     // If the character entered is printable, display it back to the user.
                //     fprintf(stderr, "Unknown option '-%c'\n", optopt);
                // }
                // else {
                //     // Print the hex code of the character if it is not printable.
                //     fprintf(stderr, "Unknown option character '\\x%x'\n", optopt);
                // }

                fprintf(stderr, "Unknown option '%s'\n", argv[optind - 1]);
                // if (isprint(optopt)) {
                //     // If the character entered is printable, display it back to the user.
                //     fprintf(stderr, "Unknown option '-%c'\n", optopt);
                // }
                // else {
                //     fprintf(stderr, "Unknown option '%s'\n", argv[optind - 1]);
                // }

                parse_error = true;
                break;
            }
            default: {
                fprintf(stderr, "Option '%s' requires an argument.\n", argv[optind - 1]);
                // if (isprint(optopt)) {
                //     // If the character entered is printable, display it back to the user.
                //     fprintf(stderr, "Option '-%c' requires an argument.\n", optopt);
                // }
                // else {
                //     fprintf(stderr, "Option '%s' requires an argument.\n", argv[optind - 1]);
                // }
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
    if (parse_error) {
        printf("usage\n");
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
