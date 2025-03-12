CC = gcc
CFLAGS = -lm -fopenmp -Ofast

ray_tracing:
	${CC} ray_tracing_cpu.c -o ray_tracing_serial ${CFLAGS}
	${CC} ray_tracing_cpu.c -o ray_tracing_omp -DUSE_OMP ${CFLAGS}
	nvcc -arch=compute_70 ray_tracing_gpu.cu -o ray_tracing_gpu -O3
	nvcc -I -L -lmpi -lmpi_cxx ray_tracing_multi.cu -o ray_tracing_multi -O3

clean:
	rm -f ray_tracing_serial ray_tracing_omp ray_tracing_gpu ray_tracing_multi *.txt
