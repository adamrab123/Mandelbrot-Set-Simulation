#include <mpfr.h>
#include <mpc.h>
#include <getopt.h>

#include "args.h"

Args *Args_init(int argc, const char **argv) {
    Args *self = malloc(sizeof(Args));

    // The base that numbers entered at the command line are in.
    int base = 10;

    // 256 bit mantissa.
    self->prec = 256;
    // Round to nearest.
    self->rnd = MPFR_RNDN;

    mpfr_inits2(self->prec, self->x_min, self->x_max, self->y_min, self->y_max, self->step_size, NULL);
    mpfr_set_str(self->x_min, "-2.1", base, self->rnd);
    mpfr_set_str(self->x_max, "1", base, self->rnd);
    mpfr_set_str(self->y_min, "-1.5", base, self->rnd);
    mpfr_set_str(self->y_max, "-1.5", base, self->rnd);
    mpfr_set_str(self->step_size, "0.01", base, self->rnd);

    self->iterations = 100;
    self->output_file = "output.bmp";
    self->block_size = 4;
}

void Args_free(Args *self) {
    mpfr_clear(self->x_min);
    mpfr_clear(self->x_max);
    mpfr_clear(self->y_min);
    mpfr_clear(self->y_max);
    mpfr_clear(self->step_size);

    free(self);
}

// Calculate the width and height of the bitmap image based on the input parameters.
void Args_bitmap_dims(const Args *self, int *width, int *height) {
    mpfr_t temp;
    mpfr_init2(temp, self->prec);

    // Calculate width = (x_max - x_min) / step_size
    mpfr_sub(temp, self->x_max, self->x_min, self->rnd);
    mpfr_div(temp, temp, self->step_size, self->rnd);
    *width = mpfr_get_ui(temp, self->rnd);

    // Calculate height = (y_max - y_min) / step_size
    mpfr_sub(temp, self->y_max, self->y_min, self->rnd);
    mpfr_div(temp, temp, self->step_size, self->rnd);
    *height = mpfr_get_ui(temp, self->rnd);

    mpfr_clear(temp);
}

void Args_bitmap_to_complex(const Args *self, int x, int y, mpc_ptr c) {
    mpc_init2(c, self->prec);

    mpfr_t c_real;
    mpfr_init_set_ui(c_real, x, self->rnd);
    mpfr_mul(c_real, c_real, self->step_size, self->rnd);
    mpfr_add(c_real, c_real, self->x_min, self->rnd);

    mpfr_t c_imag;
    mpfr_init_set_ui(c_imag, y, self->rnd);
    mpfr_mul(c_imag, c_imag, self->step_size, self->rnd);
    mpfr_add(c_imag, c_imag, self->y_min, self->rnd);

    mpc_set_fr_fr(c, c_real, c_imag, self->rnd);
    mpfr_clears(c_real, c_imag, NULL);
}

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
// Args get_args(int argc, char *argv[]) { 

//     // setting all parameters to their default values
//     int x_min = -2;
//     int x_max = 2;
//     int y_min = -2;
//     int y_max = 2;
//     float steps = 0.001;
//     int iterations = 20;
//     char* output_file = "output.bmp";
//     int block_size = 100;

//     int opt;
//     // defining what parameters getopt needs to look for
//     while((opt = getopt(argc, argv, ":x:X:y:Y:s:i:b:o:lx")) != -1) { 
//         // iterating through possible matches
//         // for int options, convert to int. 
//         // for float convert to float, etc 
//         switch(opt) {  
//             case 'x':
//             	printf("x_min set to: %s\n", optarg);
//             	x_min = atoi(optarg);
//                 break;
//             case 'X': 
//             	printf("x_max set to: %s\n", optarg);
//             	x_max = atoi(optarg);
//                 break;
//             case 'y':
//             	printf("y_min set to: %s\n", optarg);
//             	y_min = atoi(optarg);
//                 break;
//             case 'Y': 
//             	printf("y_max set to: %s\n", optarg);
//             	y_max = atoi(optarg);
//                 break;
//             case 's':
//                 printf("steps set to: %s\n", optarg);
//                 // the steps conversion requirs adding a null byte 
//                 //      to the end of the array before converting
//                 int length = sizeof(optarg) / sizeof(optarg[0]);
//                 char* word = calloc(length+1, sizeof(char));
//                 strcpy(word,optarg);
//                 word[length] = '\0';
//                 printf("Converted word = %s\n", word);
//                 steps = atof(word);
//                 break;
//             case 'i':
//                 printf("iteratios set to: %s\n", optarg);
//                 iterations = atoi(optarg);
//                 break;
//             case 'o':
//                 printf("output_file set to: %s\n", optarg);
//                 output_file = optarg;
//                 break;
//             case 'b':
//                 printf("block_size set to: %s\n", optarg);
//                 block_size = atoi(optarg);
//                 break;
//             case ':':
//                 printf("option needs a value\n");
//                 break;
//             case '?':
//                 printf("unknown option: %c\n", optopt);
//                 break;
//         }  
//     }
      
//     // optind is for the extra arguments 
//     // which are not parsed 
//     for(;optind < argc;optind++){      
//         printf("extra arguments: %s\n", argv[optind]);
//     } 

//     Args arg;

// 	arg.x_min = x_min;
// 	arg.x_max = x_max;
// 	arg.y_min = y_min;
// 	arg.y_max = y_max;
// 	arg.step_size = steps;
// 	arg.iterations = iterations;
// 	arg.output_file = output_file;
// 	arg.block_size = block_size;

// 	return arg;
// }