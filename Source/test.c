#include <stdio.h>  
#include <unistd.h>  
#include <stdlib.h>
#include <string.h>
  
int main(int argc, char *argv[]) { 
    
    int x_min = -2;
    int x_max = 2;
    int y_min = -2;
    int y_max = 2;
    float steps = 0.001;
    int iterations = 20;
    char* output_file = "output.txt";
    int block_size = 100;
      

    int opt;
    while((opt = getopt(argc, argv, ":x:X:y:Y:s:i:i:b:o:lx")) != -1) {  
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

    printf(" *** DONE PARSING COMMAND LINE ARGUMENTS\n");
    printf("x_min is now %d\n", x_min);
    printf("x_max is now %d\n", x_max);
    printf("y_min is now %d\n", y_min);
    printf("y_max is now %d\n", y_max);
    printf("steps is now %.10f\n", steps);
    printf("iterations is now %d\n", iterations);
    printf("output_file is now %s\n", output_file);
    printf("block_size is now %d\n", block_size);

 
      
    return 0;
} 