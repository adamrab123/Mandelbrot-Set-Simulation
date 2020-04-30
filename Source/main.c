#include <mpfr.h>

#include "args.h"
#include "mbmpi.h"

int main(int argc, char **argv) {
    // Hard code arguments for now.
    const Args *args = Args_init(argc, argv);

    compute_mandelbrot_parallel(args);

    // compute_mandelbrot_serial(args);

    Args_free(args);
}
