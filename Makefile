GCC_FLAGS = -std=c99 -Werror -lm
NVCC_FLAGS = -arch=sm_70
INCLUDE = -I Source/Include
CUDA_LIBS = -L/usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++
PARALLEL = -D PARALLEL
BUILD_DIR = Build
EXE = -o ${BUILD_DIR}/mandelbrot

.PHONY: clean

# Use this on Aimos.
parallel: $(wildcard Source/*.c) $(wildcard Source/*.cu) $(wildcard Source/*.h)
		mkdir -p ${BUILD_DIR}
        # Build mpi/non-cuda files to one object file each.
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} Source/bitmap.c -o ${BUILD_DIR}/bitmap.o
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} Source/main.c -o ${BUILD_DIR}/main.o
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} Source/balancer.c -o ${BUILD_DIR}/balancer.o
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE} Source/args.c -o ${BUILD_DIR}/args_mpi.o

        # Build cuda files to one object file each.
		# args.c has cuda and non-cuda functions, so it must be built twice into two different object files.
		nvcc -x cu -dc ${PARALLEL} ${INCLUDE} ${NVCC_FLAGS} Source/args.c -o ${BUILD_DIR}/args_cuda.o
		nvcc -x cu -dc ${PARALLEL} ${INCLUDE} ${NVCC_FLAGS} Source/kernel.cu -o ${BUILD_DIR}/kernel.o
		nvcc -x cu -dc ${PARALLEL} ${INCLUDE} ${NVCC_FLAGS} Source/mandelbrot.c -o ${BUILD_DIR}/mandelbrot.o
		nvcc -x cu -dc ${PARALLEL} ${INCLUDE} ${NVCC_FLAGS} Source/mbcomplex.c -o ${BUILD_DIR}/mbcomplex.o
		nvcc -x cu -dc ${PARALLEL} ${INCLUDE} ${NVCC_FLAGS} Source/colormap.c -o ${BUILD_DIR}/colormap.o

		# Link all cuda object files into one object file.
		nvcc ${PARALLEL} ${NVCC_FLAGS} -dlink \
		${BUILD_DIR}/kernel.o \
		${BUILD_DIR}/mandelbrot.o \
		${BUILD_DIR}/mbcomplex.o \
		${BUILD_DIR}/colormap.o \
		${BUILD_DIR}/args_cuda.o \
		-o ${BUILD_DIR}/link.o ${CUDA_LIBS}

		# Link all object files into one executable.
		mpicc ${PARALLEL} ${GCC_FLAGS} \
		${BUILD_DIR}/args_mpi.o \
		${BUILD_DIR}/args_cuda.o \
		${BUILD_DIR}/bitmap.o \
		${BUILD_DIR}/main.o \
		${BUILD_DIR}/colormap.o \
		${BUILD_DIR}/mbcomplex.o \
		${BUILD_DIR}/mandelbrot.o \
		${BUILD_DIR}/balancer.o \
		${BUILD_DIR}/kernel.o \
		${BUILD_DIR}/link.o \
		${EXE} ${CUDA_LIBS}

# Use this locally.
serial: $(wildcard Source/*.c) $(wildcard Source/Include*.h)
		mkdir -p ${BUILD_DIR}
		gcc ${GCC_FLAGS} ${INCLUDE} \
			Source/bitmap.c Source/colormap.c Source/main.c \
			Source/mbcomplex.c Source/args.c Source/mandelbrot.c Source/balancer.c \
			${EXE}

debug-serial: GCC_FLAGS += -g
debug-serial: serial

debug-parallel: NVCC_FLAGS += -g -G
debug-parallel: GCC_FLAGS += -g
debug-parallel: parallel

clean:
	rm ${BUILD_DIR}/*
