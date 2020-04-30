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
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} Source/bitmap.c -o ${BUILD_DIR}/bitmap.o
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} Source/main.c -o ${BUILD_DIR}/main.o
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} Source/mbparallel.c -o ${BUILD_DIR}/mbparallel.o
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} Source/args.c -o ${BUILD_DIR}/args_mpi.o
		# These files also have device code.
		# mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} ${MPFR_LIB} Source/mandelbrot.c -o ${BUILD_DIR}/mandelbrot.o
		# mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} ${MPFR_LIB} Source/mbcomplex.c -o ${BUILD_DIR}/mbcomplex.o
		# mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} ${MPFR_LIB} Source/colormap.c -o ${BUILD_DIR}/colormap.o
		# mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} ${MPFR_LIB} Source/args.c -o ${BUILD_DIR}/args.o

        # Build cuda files to one object file.
		nvcc -x cu -dc ${PARALLEL} ${INCLUDE} ${NVCC_FLAGS} Source/kernel.cu -o ${BUILD_DIR}/kernel.o
		nvcc -x cu -dc ${PARALLEL} ${INCLUDE} ${NVCC_FLAGS} Source/mandelbrot.c -o ${BUILD_DIR}/mandelbrot.o
		nvcc -x cu -dc ${PARALLEL} ${INCLUDE} ${NVCC_FLAGS} Source/mbcomplex.c -o ${BUILD_DIR}/mbcomplex.o
		nvcc -x cu -dc ${PARALLEL} ${INCLUDE} ${NVCC_FLAGS} Source/colormap.c -o ${BUILD_DIR}/colormap.o
		nvcc -x cu -dc ${PARALLEL} ${INCLUDE} ${NVCC_FLAGS} Source/args.c -o ${BUILD_DIR}/args_cuda.o

		nvcc ${PARALLEL} -arch=sm_70 -dlink \
		${BUILD_DIR}/kernel.o \
		${BUILD_DIR}/mandelbrot.o \
		${BUILD_DIR}/mbcomplex.o \
		${BUILD_DIR}/colormap.o \
		${BUILD_DIR}/args_cuda.o \
		-o ${BUILD_DIR}/link.o ${CUDA_LIBS}

        # Build object files to executable.
		mpicc ${PARALLEL} ${GCC_FLAGS} \
		${BUILD_DIR}/args_mpi.o \
		${BUILD_DIR}/args_cuda.o \
		${BUILD_DIR}/bitmap.o \
		${BUILD_DIR}/main.o \
		${BUILD_DIR}/mbparallel.o \
		${BUILD_DIR}/kernel.o \
		${BUILD_DIR}/mandelbrot.o \
		${BUILD_DIR}/mbcomplex.o \
		${BUILD_DIR}/colormap.o \
		${BUILD_DIR}/link.o \
		${EXE} ${CUDA_LIBS}

# mpi trials
		# ${BUILD_DIR}/kernel.o \
		# ${BUILD_DIR}/mandelbrot.o \
		# ${BUILD_DIR}/mbcomplex.o \
		# ${BUILD_DIR}/colormap.o \
		# ${BUILD_DIR}/args.o \

# CUDA trials:
		# nvcc -x cu -dc ${PARALLEL} -c ${INCLUDE} ${NVCC_FLAGS} ${MPFR_LIB} Source/mandelbrot.c -o ${BUILD_DIR}/mandelbrot.o
		# nvcc -x cu -dc ${PARALLEL} -c ${INCLUDE} ${NVCC_FLAGS} ${MPFR_LIB} Source/kernel.cu -o ${BUILD_DIR}/kernel.o
		# nvcc ${PARALLEL} -c ${INCLUDE} ${NVCC_FLAGS} Source/kernel.cu -o ${BUILD_DIR}/kernel.o
		# nvcc ${PARALLEL} ${INCLUDE} ${NVCC_FLAGS} ${BUILD_DIR}/kernel.o ${BUILD_DIR}/mandelbrot.o -o link.o
		# nvcc ${PARALLEL} -arch=sm_70 -dlink ${BUILD_DIR}/kernel.o -o link.o -lcudadevrt -lcudart

clean:
	rm ${BUILD_DIR}/*
