parallel: $(wildcard Source/*.c) $(wildcard Source/*.cu) $(wildcard Source/*.h) 
        # Build c/mpi files to one object file.
		mpicc -D PARALLEL -g -I Source/Include Source/bitmap.c -c -o Build/bitmap.o
		mpicc -D PARALLEL -g -I Source/Include Source/colormap.c -c -o Build/bitmap.o
		mpicc -D PARALLEL -g -I Source/Include Source/args.c -c -o Build/bitmap.o
		mpicc -D PARALLEL -g -I Source/Include Source/main.c -c -o Build/main.o
		mpicc -D PARALLEL -g -I Source/Include Source/mandelbrot.c -c -o Build/mandelbrot.o
		mpicc -D PARALLEL -g -I Source/Include Source/mbparallel.c -c -o Build/mbmpi.o
        # Build cuda files to one object file.
		nvcc -D PARALLEL -g -G -I Source/Include -c -arch=sm_70 Source/kernel.cu -o Build/kernel.o
        # Build object files to executable.
		mpicc -D PARALLEL -g Build/*.o -o Build/mandelbrot \
			-L/usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++

serial: $(wildcard Source/*.c) $(wildcard Source/Include*.h)
		gcc -g -lmpfr -lmpc -lm -I Source/Include Source/bitmap.c Source/colormap.c Source/main.c \
			Source/args.c Source/mandelbrot.c Source/mbserial.c -o Build/mandelbrot