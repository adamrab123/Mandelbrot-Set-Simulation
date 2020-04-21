# Mandelbrot Set Simulation

- Overleaf document: https://www.overleaf.com/4578711964kfrnpsghcgsg

## Floating Point Arithmetic

- A tradeoff between accuracy and performance must be made.

- We should determine what the maximum possible image size we may want to generate is, and not pick a type that has significantly more precision than that.

- Options:

    - **double**: 8 Bytes.
        - lower bound: ±2.23 x 10^-308
        - upper bound: ±1.80 x 10^308
        - significant digits: 15-18, typically 16
    
    - **long double**: 16 Bytes.
        - 128 bit quad precision float.
        - lower bound: ±3.36 x 10^-4932
        - upper bound: ±1.18 x 10^4932
        - significant digits: 33-36

        - Should hopefully be enough for our uses.

    - **double double**: Not available on Aimos.

    - **__float128 (quad precision floating point)**: Not available on Aimos.
        - Woudl be impelemented the same as long double with 16 bytes.
    
    - External libraries that are probably not worth the hassle:
        - **GMP**
        - **MPFR**

## Image Formatting

- **.ppm**: Plain text image specification that results in very large files, but can be written in ASCII without a library.

    - May require converting to another image format later anyways.

    - High resolution images may be prohibitively large.

- **Custom Encoding Converted with Python**: Have the C program output a text file of a custom encoding we create, and write a Python script to convert it to a lossless image file type like .PNG (not .JPG).

    - Requires two steps to generate image.

    - May require moving this large image file across Aimos internet connections.

- **Bitmap File format (BMP)**: Each pixel encoded in a byte, with some sort of metadata header at the beginning of the file.

    - Maybe slightly more difficult to write than ppm, but probably more compact before compression.

    - Wikipedia says zip can provide effective lossless compression for moving the data off Aimos.

## Command Line Arguments

- Use the `getopt.h` standard header file for parsing flags.

- Potential arguments to support:

    - `--xmin, -x`
        - default: -2

    - `--xmax, -X`
        - default: 2

    - `--ymin, -y`
        - default: -2

    - `--ymax, -Y`
        - default: -2

    - `--step, -s`: Counting from zero, this amount will be added in the x or y direction to determine every point that will be calculated within the x and y bounds.

        - default: 0.001

        - The image dimensions will be *(xmax - xmin) / step* and *(xmax - xmin) / step*.

    - `--iterations, -i`: The number of iterations to perform before determining whether a number belongs in the set or not.
        - default: 20

    - `--outputfile, -o`: The name of the file to output the image to.

    - `--blocksize, -b`: The number of threads per block to use in the computation.

        - default: TBD

        - The number of blocks to use will be computed from the total number of points to calculate and this value.

## Math

- The x axis is the real axis, the y axis is the imaginary axis.

- Each point on this plane represents c omplex number c.

- c is in the mandelbrot set iff Z_{n+1} = Z_n^2 + c does not diverge when iterated a high number of times.
    - Seed this function with Z_n = 0.

    - Once Z_{n+1} > 2, it is known to diverge and is automatically not in the set.
        - If this does not happen after the chosen iteration count, the number is considered a part of the set.
        - The number of iterations performed before this happens can be used to determine the color of the point c on the visual.

- The Mandelbrot set is known to by symmetric about the real axis, so we only need to compute the top half of the image.

- Operations can be implemented using the `complex.h` standard header file.
    - Make sure to use the functions defined for `long double`s.

- An example of the math without the `complex.h` header is [here](https://www.geeksforgeeks.org/fractals-in-cc/).

- Checking if we are outside radius 2 from the origin involves the point distance formula:

    - *sqrt[(x2 - x1)^2 + (y2 - y1)^2] -> sqrt[c_real^2 + c_img^2] < 2*
    - By squaring each side we can save an operation: *c_real^2 + c_img^2 < 4*.

## Coloring

- Still need to research this more.

- [A quick overview with a formula](https://linas.org/art-gallery/escape/smooth.html)

- [A more in depth article](http://www.iquilezles.org/www/articles/mset_smooth/mset_smooth.htm)

## Division of Labor

- argument parsing
- bitmap/image generation
- actual math
- thread/block divisions

- Adam: arg parsing
- Ethan: math
- John: bitmap/image generation + coloring algorithm
- Shashank: thread/block divisions

## Coding Conventions

- Doxygen blocks for public functions

- Python style naming conventions.
    - snake_case for variables and functions.
    - PascalCase for types.
    - alllowercase for file names/headers.
    - _leading_underscore for private functions.

## File sizes

- Compression results depend on the file contents.

- The tests below were performed on a 24,884,010 byte bmp file.

    - Convert bmp to png: 3,346,416 bytes.

    - bmp to zip archive: 4,220,059 bytes

- To approximate compressed file size given one dimension pixel count of a square image:

    - d = number of pixels along one dimension of the image.

    - fc = compression factor. Factor by which the compression algorithm will reduce image size.

    - compressed size in bytes = (d^2 * 3) / cf

    - Assuming fc = 4 (conservative estimate based on the test file which had cf ~= 7):

        - d = 10,000: 75,000,000 bytes (75 Mb)
        - d = 100,000: 7,500,000,000 bytes (7.5 Gb)
        - d = 1,000,000: 750,000,000,000 bytes (750 Gb)

