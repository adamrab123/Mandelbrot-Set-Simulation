BUILD_DIR = Build
EXE = -o ${BUILD_DIR}/mandelbrot
SOURCE_DIR = Source
INCLUDE_DIRS = -I ${SOURCE_DIR}/Include

GCC_FLAGS = -std=c99 -Werror -lm
NVCC_FLAGS = -arch=sm_70
CUDA_LIBS = -L/usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++
PARALLEL = -D PARALLEL

.PHONY: clean

# Use this on Aimos.
parallel: $(wildcard ${SOURCE_DIR}/*.c) $(wildcard ${SOURCE_DIR}/*.cu) $(wildcard ${SOURCE_DIR}/*.h)
		mkdir -p ${BUILD_DIR}
        # Build mpi/non-cuda files to one object file each.
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE_DIRS} ${SOURCE_DIR}/bitmap.c -o ${BUILD_DIR}/bitmap.o
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE_DIRS} ${SOURCE_DIR}/main.c -o ${BUILD_DIR}/main.o
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE_DIRS} ${SOURCE_DIR}/balancer.c -o ${BUILD_DIR}/balancer.o
		mpicc ${PARALLEL} ${GCC_FLAGS} -c ${INCLUDE_DIRS} ${SOURCE_DIR}/args.c -o ${BUILD_DIR}/args_mpi.o

        # Build cuda files to one object file each.
		# args.c has cuda and non-cuda functions, so it must be built twice into two different object files.
		nvcc -x cu -dc ${PARALLEL} ${INCLUDE_DIRS} ${NVCC_FLAGS} ${SOURCE_DIR}/args.c -o ${BUILD_DIR}/args_cuda.o
		nvcc -x cu -dc ${PARALLEL} ${INCLUDE_DIRS} ${NVCC_FLAGS} ${SOURCE_DIR}/kernel.cu -o ${BUILD_DIR}/kernel.o
		nvcc -x cu -dc ${PARALLEL} ${INCLUDE_DIRS} ${NVCC_FLAGS} ${SOURCE_DIR}/mandelbrot.c -o ${BUILD_DIR}/mandelbrot.o
		nvcc -x cu -dc ${PARALLEL} ${INCLUDE_DIRS} ${NVCC_FLAGS} ${SOURCE_DIR}/mbcomplex.c -o ${BUILD_DIR}/mbcomplex.o
		nvcc -x cu -dc ${PARALLEL} ${INCLUDE_DIRS} ${NVCC_FLAGS} ${SOURCE_DIR}/colormap.c -o ${BUILD_DIR}/colormap.o

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
serial: $(wildcard ${SOURCE_DIR}/*.c) $(wildcard ${SOURCE_DIR}/Include*.h)
		mkdir -p ${BUILD_DIR}
		gcc ${GCC_FLAGS} ${INCLUDE_DIRS} \
			${SOURCE_DIR}/bitmap.c ${SOURCE_DIR}/colormap.c ${SOURCE_DIR}/main.c \
			${SOURCE_DIR}/mbcomplex.c ${SOURCE_DIR}/args.c ${SOURCE_DIR}/mandelbrot.c ${SOURCE_DIR}/balancer.c \
			${EXE}

debug-serial: GCC_FLAGS += -g
debug-serial: serial

debug-parallel: NVCC_FLAGS += -g -G
debug-parallel: GCC_FLAGS += -g
debug-parallel: parallel

clean:
	rm ${BUILD_DIR}/*
