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
    // setting all parameters to their default values
    double x_min = -2.1;
    double x_max = 1;
    double y_min = -1.5;
    double y_max = 1.5;
    double steps = 0.01;
    long iterations = 100;
    char *output_file = (char *)calloc(1, 100);
    strcpy(output_file, "output.bmp");
    int block_size = 100;
    int chunk = 1;

    int c;
    // defining what parameters getopt needs to look for
    while (1)
    {
        static struct option long_options[] =
            {
                /* These options set a flag. */
                // {"verbose", no_argument,       &verbose_flag, 1},
                // {"brief",   no_argument,       &verbose_flag, 0},
                /* These options don’t set a flag.
             We distinguish them by their indices. */
                {"x-min", required_argument, 0, 'x'},
                {"x-max", required_argument, 0, 'X'},
                {"y-min", required_argument, 0, 'y'},
                {"y-max", required_argument, 0, 'Y'},
                {"step-size", required_argument, 0, 's'},
                {"output-file", required_argument, 0, 'o'},
                {"block-size", required_argument, 0, 'b'},
                {"iterations", required_argument, 0, 'i'},
                {"chunk", required_argument, 0, 'c'},
                {0, 0, 0, 0}};
        /* getopt_long stores the option index here. */
        int option_index = 0;

        c = getopt_long(argc, argv, ":x:X:y:Y:s:o:b:i:c:lx",
                        long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
            break;

        switch (c)
        {
        case 0:
            /* If this option set a flag, do nothing else now. */
            if (long_options[option_index].flag != 0)
                break;
            printf("option %s", long_options[option_index].name);
            if (optarg)
                printf(" with arg %s", optarg);
            printf("\n");
            break;

        case 'x':
            printf("x_min set to: %s\n", optarg);
            x_min = atoi(optarg);
            break;
        case 'X':
            printf("x_max set to: %s\n", optarg);
            x_max = atoi(optarg);
            break;
        case 'y':
            printf("y_min set to: %s\n", optarg);
            y_min = atoi(optarg);
            break;
        case 'Y':
            printf("y_max set to: %s\n", optarg);
            y_max = atoi(optarg);
            break;
        case 's':
            printf("steps set to: %s\n", optarg);
            // the steps conversion requirs adding a null byte
            //      to the end of the array before converting
            int length = sizeof(optarg) / sizeof(optarg[0]);
            char *word = calloc(length + 1, sizeof(char));
            strcpy(word, optarg);
            word[length] = '\0';
            // printf("Converted word = %s\n", word);
            steps = atof(word);
            break;
        case 'i':
            printf("iteratios set to: %s\n", optarg);
            iterations = atoi(optarg);
            break;
        case 'o':
            printf("output_file set to: %s\n", optarg);
            output_file = optarg;
            break;
        case 'b':
            printf("block_size set to: %s\n", optarg);
            block_size = atoi(optarg);
            break;
        case 'c':
            printf("chunk set to: %s\n", optarg);
            chunk = atoi(optarg);
            break;

        default:
            abort();
        }
    }

    /* Instead of reporting ‘--verbose’
     and ‘--brief’ as they are encountered,
     we report the final status resulting from them. */
    // if (verbose_flag)
    //   puts ("verbose flag is set");

    /* Print any remaining command line arguments (not options). */
    if (optind < argc)
    {
        printf("non-option ARGV-elements: ");
        while (optind < argc)
            printf("%s ", argv[optind++]);
        putchar('\n');
    }

    Args *arg = (Args *)malloc(sizeof(Args));

    arg->x_min = x_min;
    arg->x_max = x_max;
    arg->y_min = y_min;
    arg->y_max = y_max;
    arg->step_size = steps;
    arg->iterations = iterations;
    arg->output_file = output_file;
    arg->block_size = block_size;
    arg->chunk = chunk;

    return arg;
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
