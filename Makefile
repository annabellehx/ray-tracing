CC = nvcc
CFLAGS = -lm -use_fast_math -O3

ray_tracing:
	gcc ray_tracing_serial.c -o ray_tracing_serial -fopenmp -lm -Ofast
	gcc ray_tracing_omp.c -o ray_tracing_omp -fopenmp -lm -Ofast
	${CC} -arch=compute_70 ray_tracing_gpu.cu -o ray_tracing_gpu_v100 ${CFLAGS}
	${CC} -arch=compute_80 ray_tracing_gpu.cu -o ray_tracing_gpu_a100 ${CFLAGS}
	${CC} -arch=compute_75 ray_tracing_gpu.cu -o ray_tracing_gpu_rtx6000 ${CFLAGS}
	${CC} -I -L -lmpi -lmpi_cxx ray_tracing_multi.cu -o ray_tracing_multi ${CFLAGS}

clean:
	rm -f ray_tracing_serial ray_tracing_omp ray_tracing_gpu_* ray_tracing_multi *.txt *.out
