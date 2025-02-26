#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <assert.h>
#include <curand_kernel.h>
#include "mpi.h"

typedef struct Vector
{
    float x;
    float y;
    float z;
} Vector;

__device__ inline void vector_add(const Vector V, const Vector U, Vector *result)
{
    result->x = V.x + U.x;
    result->y = V.y + U.y;
    result->z = V.z + U.z;
}

__device__ inline void vector_subtract(const Vector V, const Vector U, Vector *result)
{
    result->x = V.x - U.x;
    result->y = V.y - U.y;
    result->z = V.z - U.z;
}

__device__ inline void vector_multiply(const float t, const Vector V, Vector *result)
{
    result->x = t * V.x;
    result->y = t * V.y;
    result->z = t * V.z;
}

__device__ inline void vector_divide(const float t, const Vector V, Vector *result)
{
    result->x = V.x / t;
    result->y = V.y / t;
    result->z = V.z / t;
}

__device__ inline float vector_dot_product(const Vector V, const Vector U)
{
    return V.x * U.x + V.y * U.y + V.z * U.z;
}

__device__ inline float vector_norm(const Vector V)
{
    return sqrtf(vector_dot_product(V, V));
}

__global__ void ray_tracing(float L_x, float L_y, float L_z, float W_y, float W_max, float C_x, float C_y, float C_z, float R, int ngrid, long nrays, float *d_matrix, unsigned long long *d_total_rays, int rank, int size)
{
    int idx = rank * nrays / size + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrays)
        return;

    int total_threads = gridDim.x * blockDim.x;
    long base_work = nrays / size / total_threads;
    long remainder = nrays / size % total_threads;
    long nrays_thread = (idx < remainder) ? base_work + 1 : base_work;

    const Vector L = {L_x, L_y, L_z};
    const Vector C = {C_x, C_y, C_z};

    const float W_max2 = W_max * W_max;
    const float dx = (ngrid - 1) / (2 * W_max);
    const float dz = (ngrid - 1) / (2 * W_max);
    const float view_ray_constant = R * R - vector_dot_product(C, C);
    unsigned long long total = 0;

    float phi, cos_theta, sin_theta, dot_prod, t, b;
    float view_ray_equation = 0;
    int i, j;

    curandStateXORWOW_t state;
    curand_init(123456789 + idx, 1, 0, &state);
    Vector V, W, I, N, S;
    W.x = W.z = 0;
    W.y = W_y;

    for (long k = 0; k < nrays_thread; ++k)
    {
        while (view_ray_equation <= 0 || W.x * W.x >= W_max2 || W.z * W.z >= W_max2)
        {
            phi = curand_uniform(&state) * M_PI;
            cos_theta = 2.0 * curand_uniform(&state) - 1.0;
            sin_theta = sqrtf(1 - cos_theta * cos_theta);

            V.x = sin_theta * cosf(phi);
            V.y = sin_theta * sinf(phi);
            V.z = cos_theta;

            vector_multiply(W_y / V.y, V, &W);
            dot_prod = vector_dot_product(V, C);
            view_ray_equation = dot_prod * dot_prod + view_ray_constant;
            total++;
        }

        t = dot_prod - sqrtf(view_ray_equation);
        vector_multiply(t, V, &I);

        vector_subtract(I, C, &N);
        vector_divide(vector_norm(N), N, &N);

        vector_subtract(L, I, &S);
        vector_divide(vector_norm(S), S, &S);

        b = (vector_dot_product(S, N) > 0) ? vector_dot_product(S, N) : 0;
        i = (W.x + W_max) * dx;
        j = (W.z + W_max) * dz;

        atomicAdd(&d_matrix[ngrid * i + j], b);
        view_ray_equation = 0;
    }

    atomicAdd(d_total_rays, total);
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s <NRAYS> <NGRID> <NBLOCKS> <NTHREADS_PER_BLOCK> \n", argv[0]);
        return EXIT_FAILURE;
    }

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    float *g_matrix = NULL;
    unsigned long long g_total_rays = 0;

    long NRAYS = atol(argv[1]);
    int NGRID = atoi(argv[2]);
    int NBLOCKS = atoi(argv[3]);
    int NTHREADS_PER_BLOCK = atoi(argv[4]);

    float *d_matrix, *l_matrix = (float *)calloc(NGRID * NGRID, sizeof(float));
    unsigned long long *d_total_rays, *l_total_rays = (unsigned long long *)calloc(1, sizeof(unsigned long long));
    if (rank == 0)
        g_matrix = (float *)calloc(NGRID * NGRID, sizeof(float));

    cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    float kernel_time;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    assert(cudaMalloc((void **)&d_matrix, NGRID * NGRID * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc((void **)&d_total_rays, sizeof(unsigned long long)) == cudaSuccess);

    assert(cudaMemset(d_matrix, 0, NGRID * NGRID * sizeof(float)) == cudaSuccess);
    assert(cudaMemset(d_total_rays, 0, sizeof(unsigned long long)) == cudaSuccess);

    cudaEventRecord(start_kernel, 0);
    ray_tracing<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(4, 4, -1, 2, 2, 0, 12, 0, 6, NGRID, NRAYS, d_matrix, d_total_rays, rank, size);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_kernel, 0);

    assert(cudaMemcpy(l_matrix, d_matrix, NGRID * NGRID * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(l_total_rays, d_total_rays, sizeof(unsigned long long), cudaMemcpyDeviceToHost) == cudaSuccess);

    MPI_Reduce(l_matrix, g_matrix, NGRID * NGRID, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(l_total_rays, &g_total_rays, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        FILE *file = fopen("matrix_cuda.out", "w");

        for (int i = 0; i < NGRID * NGRID; ++i)
            fprintf(file, "%.2lf ", g_matrix[i]);

        fclose(file);
        free(g_matrix);

        double stop = MPI_Wtime();
        double total_time = 1000 * (stop - start);
        cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);

        printf("\nTotal Time of Execution  : %lf (ms)\n", total_time);
        printf("Kernel Time of Execution : %lf (ms)\n", kernel_time);
        printf("Number of Accepted Rays  : %ld\n", NRAYS);
        printf("Number of Rejected Rays  : %ld\n\n", g_total_rays - NRAYS);
    }

    free(l_matrix);
    free(l_total_rays);
    cudaFree(d_matrix);
    cudaFree(d_total_rays);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
