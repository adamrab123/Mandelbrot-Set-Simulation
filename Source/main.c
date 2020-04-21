
#include "args.h"


int main(int argc, char *argv[]) { 

	Arguments arg = getArgs( argc, argv);

	printf("x_min is now %d\n", arg.x_min);
    printf("x_max is now %d\n", arg.x_max);
    printf("y_min is now %d\n", arg.y_min);
    printf("y_max is now %d\n", arg.y_max);
    printf("steps is now %.10f\n", arg.steps);
    printf("iterations is now %d\n", arg.iterations);
    printf("output_file is now %s\n", arg.output_file);
    printf("block_size is now %d\n", arg.block_size);
    
    return 0;
} 