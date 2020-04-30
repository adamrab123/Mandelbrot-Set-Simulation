ARGS = -std=c99 -Werror -lmpfr -lmpc -lm
NVCC_ARGS = -arch=sm_70
INCLUDE = -I Source/Include
MPFR_LIB = -I Lib/Include -L Lib
CUDA_LIB = -L/usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++
PARALLEL = -D PARALLEL
BUILD_DIR = Build
EXE = -o ${BUILD_DIR}/mandelbrot

# Use this locally.
serial: $(wildcard Source/*.c) $(wildcard Source/Include*.h)
		gcc ${ARGS} ${INCLUDE} \
			Source/bitmap.c Source/colormap.c Source/main.c \
			Source/args.c Source/mandelbrot.c Source/mbserial.c \
			${EXE}

# Use this on Aimos.
parallel: $(wildcard Source/*.c) $(wildcard Source/*.cu) $(wildcard Source/*.h) 
        # Build c/mpi files to one object file.
		mpicc ${PARALLEL} ${ARGS} -c ${INCLUDE} ${MPFR_LIB} Source/bitmap.c -o ${BUILD_DIR}/bitmap.o
		mpicc ${PARALLEL} ${ARGS} -c ${INCLUDE} ${MPFR_LIB} Source/colormap.c -o ${BUILD_DIR}/colormap.o
		mpicc ${PARALLEL} ${ARGS} -c ${INCLUDE} ${MPFR_LIB} Source/args.c -o ${BUILD_DIR}/args.o
		mpicc ${PARALLEL} ${ARGS} -c ${INCLUDE} ${MPFR_LIB} Source/main.c -o ${BUILD_DIR}/main.o
		mpicc ${PARALLEL} ${ARGS} -c ${INCLUDE} ${MPFR_LIB} Source/mandelbrot.c -o ${BUILD_DIR}/mandelbrot.o
		mpicc ${PARALLEL} ${ARGS} -c ${INCLUDE} ${MPFR_LIB} Source/mbparallel.c -o ${BUILD_DIR}/mbparallel.o

        # Build cuda files to one object file.
		nvcc -x cu -dc ${PARALLEL} -c ${INCLUDE} ${NVCC_ARGS} ${MPFR_LIB} Source/mandelbrot.c -o ${BUILD_DIR}/mandelbrot.o
		nvcc -x cu -dc ${PARALLEL} -c ${INCLUDE} ${NVCC_ARGS} ${MPFR_LIB} Source/kernel.cu -o ${BUILD_DIR}/kernel.o
		nvcc ${PARALLEL} ${INCLUDE} ${NVCC_ARGS} ${MPFR_LIB} ${BUILD_DIR}/kernel.o ${BUILD_DIR}/mandelbrot.o -o link.o
		# nvcc ${PARALLEL} -arch=sm_70 -dlink ${BUILD_DIR}/kernel.o -o link.o -lcudadevrt -lcudart
        # Build object files to executable.
		mpicc ${PARALLEL} ${ARGS} ${MPFR_LIB} ${BUILD_DIR}/*.o ${EXE} ${CUDA_LIB}

# Not really useful but included for completeness.
aimos-serial: $(wildcard Source/*.c) $(wildcard Source/Include*.h)
		gcc ${ARGS} -std=c99 ${INCLUDE} ${MPFR_LIB} \
			Source/bitmap.c Source/colormap.c Source/main.c \
			Source/args.c Source/mandelbrot.c Source/mbserial.c \
			${EXE}

clean:
	rm ${BUILD_DIR}/*
