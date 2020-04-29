#include "args.h"
#include "mbmpi.h"

int main(int argc, char **argv) {
    // Hard code arguments for now.
    const Args *args = Args_init(argc, argv);

    start_mpi(args);

    Args_free(args);
}