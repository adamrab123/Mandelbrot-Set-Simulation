GCC_FLAGS = -std=c99 -Werror -lm
NVCC_FLAGS = -arch=sm_70
INCLUDE = -I Source/Include
CUDA_LIBS = -L/usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++
PARALLEL = -D PARALLEL
BUILD_DIR = Build
EXE = -o ${BUILD_DIR}/mandelbrot

# Use this locally.
serial: $(wildcard Source/*.c) $(wildcard Source/Include*.h)
		gcc ${GCC_FLAGS} ${INCLUDE} \
			Source/bitmap.c Source/colormap.c Source/main.c \
			Source/mbcomplex.c Source/args.c Source/mandelbrot.c Source/mbserial.c \
			${EXE}

# Use this on Aimos.
parallel: $(wildcard Source/*.c) $(wildcard Source/*.cu) $(wildcard Source/*.h) 
        # Build c/mpi files to one object file.
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} ${MPFR_LIB} Source/bitmap.c -o ${BUILD_DIR}/bitmap.o
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} ${MPFR_LIB} Source/main.c -o ${BUILD_DIR}/main.o
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} ${MPFR_LIB} Source/mbparallel.c -o ${BUILD_DIR}/mbparallel.o
		# These files also have device code.
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} ${MPFR_LIB} Source/mandelbrot.c -o ${BUILD_DIR}/mandelbrot.o
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} ${MPFR_LIB} Source/mbcomplex.c -o ${BUILD_DIR}/mbcomplex.o
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} ${MPFR_LIB} Source/colormap.c -o ${BUILD_DIR}/colormap.o
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} ${MPFR_LIB} Source/args.c -o ${BUILD_DIR}/args.o

        # Build cuda files to one object file.
		nvcc -x cu -dc ${PARALLEL} -c ${INCLUDE} ${NVCC_FLAGS} ${MPFR_LIB} Source/mandelbrot.c -o ${BUILD_DIR}/mandelbrot.o
		nvcc -x cu -dc ${PARALLEL} -c ${INCLUDE} ${NVCC_FLAGS} ${MPFR_LIB} Source/kernel.cu -o ${BUILD_DIR}/kernel.o
		nvcc ${PARALLEL} ${INCLUDE} ${NVCC_FLAGS} ${MPFR_LIB} ${BUILD_DIR}/kernel.o ${BUILD_DIR}/mandelbrot.o -o link.o
		# nvcc ${PARALLEL} -arch=sm_70 -dlink ${BUILD_DIR}/kernel.o -o link.o -lcudadevrt -lcudart
        # Build object files to executable.
		mpicc ${PARALLEL} ${GCC_FLAGS} ${MPFR_LIB} ${BUILD_DIR}/*.o ${EXE} ${CUDA_LIBS}

clean:
	rm ${BUILD_DIR}/*
