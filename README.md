# Mandelbrot Set Simulation

- A C program that generates fractal images based on the Mandelbrot set using MPI and CUDA to parallelize computations and file writing.

- All paths referenced in this document are relative to the top level directory for this project.

## Building

- The program has two build modes: serial and parallel.

    - Both build modes will create a directory called `Build` that will contain the output executable called `mandelbrot`.

    - The different build modes are implemented using preprocessor directives to change which parts of the program are compiled.

    - This allows zero code duplication between the serial and parallel versions of the program.

- To build the program in parallel mode, run `make parallel` from the top level directory.

    - This will build a version of the program that parallelizes computation and output generation using MPI and CUDA.

- To build the program in serial mode, run `make serial` from the top level directory.

    - This will build a version of the program that can be run on a normal computer without MPI or CUDA support.

    - This version has no parallelization, and runs significantly slower than the parallelized version.

## Output Formatting

- In both parallel and serial modes, the program will output the visualization as a bitmap file (.bmp).

- Bitamps are an uncompressed format, so if the image is large, it may be helpful to convert it to a lossless compression format like PNG before moving or opening it.

    - The output file `output.bmp` can be converted to a PNG using the `Imagemagick` command `convert output.bmp output.png`.

    - `Imagemagick` is installed on Aimos and can be used to convert images before downloading them to view.

- The image is generated on a complex number plane, with the real axis on the x, and the complex axis on the y.

- The resolution of the image is determined by the step size, which is the distance between points on the x and y axis calculated by the program.

- The image will be *(xmax - xmin) / step* pixels wide and *(ymax - ymin) / step* pixels tall.

- See [Command Line Arguments](#command-line-arguments) for more information about altering the output image.

## Command Line Arguments

- This program takes long and short form command line arguments formatted in standard Unix fashion.

- Command line arguments are implemented using the C `getopt` library.

- All arguments are optional, and have sensible default values.

- All arguments are enabled in parallel builds, but some arguments are not supported in serial builds.

- The program will use all MPI ranks available when launched using the `mpirun` command, so this is not specified by the command line arguments.

- If arguments are used incorrectly, the program will provide an error message and exit.

### Arguments Supported in Parallel and Serial Modes

- `-x, --x-min=XMIN`
    - The minimum value on the x axis (real axis) to compute.
    - default: -2.1

- `-X, --x-max=XMAX`
    - The maximum value on the x axis (real axis) to compute.
    - default: 1

- `-y, --y-min=YMIN`
    - The minimum value on the y axis (imaginary axis) to compute.
    - default: -1.5

- `-Y, --y-max=YMAX`
    - The maximum value on the y axis (imaginary axis) to compute.
    - default: 1.5

- `-s, --step-size=STEPSIZE`
    - The amount moved between points in the x and y directions to determine the unique points that will be calculated within the x and y bounds.

    - default: 0.01

- `-i, --iterations=ITERATIONS`
    - The number of iterations of the mandelbrot set formula to perform before determining whether a number belongs in the mandelbrot set or not.

    - default: 100

- `-o, --output-file=OUTPUTFILE`
    - The name of the file to output the image to.

    - The program does not create subidrectories, so any subidrectories containing this output file must exist before the program is run.

    - default: output.bmp
    
- `-c, --chunks=CHUNKS`
    - The number of chunks per process to perform the computations and file writes in.

    - default: 1

    - See [Computation Chunking](#computation-chunking) for more information about this option.

- `-d, --delete-output`

    - Delete the output file immediately after the program finishes.

    - This is useful for running batch testing with the `-t` option, so that a lot of large image files are not generated in addition to the timing data.

### Arguments Supported in Parallel Mode Only

- `-b, --block-size=BLOCKSIZE`
    - The number of CUDA threads per block per process to use in the computation.

    - default: 64

- `-t, --time-dir=DIRECTORY`

    - Have each rank/process time its execution, assuming a hardware clock rate of 512000000Hz, and write data about its configuration, run time, and bytes written to a file called *<rank>.yaml* in the specified subidrectory.

    - The directory will not be created by the program, so it must exist before the program is run.

    - The timing is implemented using assembly code that only builds on Aimos, so this option is disabled in serial builds.

## Computation Chunking

- Without computation chunking (`--chunks=1`), compuation of the image is carried out in the following way:

    - Serial mode
        - An array the size of the entire image is allocated on the heap, and filled with RGB values of colors one by one.

        - The entire array is written in one call to the output file.

    - Parallel mode
        - If given *N* ranks, the image is divded into *N* horizontal groups, and each rank is given a group to calculate.

        - Each rank allocates an array of size equal to its group on the device, and passes this array to the device to be filled in by the threads.

            - The computation is divded evenly among the threads and carried out in parallel.

        - Once the device code for a rank finishes, it writes its group of data to the output file using one MPI collective parallel IO write call.

- The above usually provides the best performance by minimizing the amount of file writes, but can be problematic for larger files.

    - If the amount of memory needed to allocate is too large, or the amount of data being pushed to the file in a single write is too large, the program could crash.

- Compute chunking alleviates this load by allowing each process to perform the allocate, compute, write steps serially in a specified amount of subgroups, instead of as one big group.

    - Within each subgroup, computation is still parallelized at the thread level.

    - If given chunk size *C*, each process will divide its group of pixels into *C* subgroups by row.
    
    - For each subgroup, the memory is allocated, the computation is performed, the data is written, and the memory is freed before continuing computation of the next subgroup.

- If the number of chunks specified is greater than the number of rows a process has to compute, it will be set to the number of rows that process has to compute.

- Examples:

    - An invocation with 1 rank, blocksize of 1, and chunk size of 64 will compute and write to the file in 64 serial chunks.

        - Using rank and blocksize of 1 is comparable to running the program in serial mode.

    - An invocation with 1 rank, blocksize of 32, and chunk size of 64 will compute the output in 64 serial groups, each group using 32 threads to parallelize computation, and write these to the file as 64 chunks serially.
