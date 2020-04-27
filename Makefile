all: $(wildcard Source/*.c) $(wildcard Source/*.cu) $(wildcard Source/*.h) 
        # Build c/mpi files to one object file.
		mpicc -g -I Source/Include Source/*.c -o Build/mpi.o
        # Build cuda files to one object file.
		nvcc -g -G -I Source/Include -arch=sm_70 Source/*.cu -o Build/cuda.o
        # Build object files to executable.
		mpicc -g Build/*.o -o Build/mandelbrot \
			-L/usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++
