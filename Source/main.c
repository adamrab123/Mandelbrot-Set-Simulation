#include <stdlib.h>
#include "args.h"

int main(int argc, char **argv) {
    // Hard code arguments for now.
    Args *args = Args_init(argc, argv);

    #ifdef PARALLEL
    #include "mbparallel.h"
    compute_mandelbrot_parallel(args);
    #else
    #include "mbserial.h"
    compute_mandelbrot_serial(args);
    #endif

    Args_free(args);

    return EXIT_SUCCESS;
}
