#include <stdlib.h>
#include <stdio.h>  
#include <unistd.h>  
#include <string.h>
#include <getopt.h>

#include "args.h"
#include "mbmpi.h"

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
Arguments get_args(int argc, char *argv[]) { 

    // setting all parameters to their default values
    int x_min = -2;
    int x_max = 2;
    int y_min = -2;
    int y_max = 2;
    float steps = 0.001;
    int iterations = 20;
    char* output_file = "output.bmp";
    int block_size = 100;

    int opt;
    // defining what parameters getopt needs to look for
    while((opt = getopt(argc, argv, ":x:X:y:Y:s:i:b:o:lx")) != -1) { 
        // iterating through possible matches
        // for int options, convert to int. 
        // for float convert to float, etc 
        switch(opt) {  
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
                char* word = calloc(length+1, sizeof(char));
                strcpy(word,optarg);
                word[length] = '\0';
                printf("Converted word = %s\n", word);
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
            case ':':
                printf("option needs a value\n");
                break;
            case '?':
                printf("unknown option: %c\n", optopt);
                break;
        }  
    }
      
    // optind is for the extra arguments 
    // which are not parsed 
    for(;optind < argc;optind++){      
        printf("extra arguments: %s\n", argv[optind]);
    } 

    Arguments arg;

	arg.x_min = x_min;
	arg.x_max = x_max;
	arg.y_min = y_min;
	arg.y_max = y_max;
	arg.step_size = steps;
	arg.iterations = iterations;
	arg.output_file = output_file;
	arg.block_size = block_size;

	return arg;
} 

int main(int argc, char **argv) {
    Arguments args = get_args(argc, argv);

    start_mpi(args);
}