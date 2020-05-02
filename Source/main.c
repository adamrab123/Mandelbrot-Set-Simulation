#include <stdlib.h>
#include "args.h"
#include "balancer.h"

int main(int argc, char **argv) {
    // Hard code arguments for now.
    Args *args = get_args(argc, argv);

    #ifdef PARALLEL
    compute_mandelbrot_parallel(args);
    #else
    compute_mandelbrot_serial(args);
    #endif

    Args_free(args);

    return EXIT_SUCCESS;
}
