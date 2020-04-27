all: $(wildcard Source/*.c) $(wildcard Source/*.cu) $(wildcard Source/*.h) 
        # Build c/mpi files to one object file.
		mpicc -g -I Source/Include Source/bitmap.c -c -o Build/bitmap.o
		mpicc -g -I Source/Include Source/main.c -c -o Build/main.o
		mpicc -g -I Source/Include Source/mandelbrot.c -c -o Build/mandelbrot.o
		mpicc -g -I Source/Include Source/mbmpi.c -c -o Build/mbmpi.o
        # Build cuda files to one object file.
		nvcc -g -G -I Source/Include -c -arch=sm_70 Source/kernel.cu -o Build/kernel.o
        # Build object files to executable.
		mpicc -g Build/*.o -o Build/mandelbrot \
			-L/usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++
