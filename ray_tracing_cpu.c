#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "vector.h"
#include "xoshiro256.h"

int ray_tracing(double L_x, double L_y, double L_z, double W_y, double W_max, double C_x, double C_y, double C_z, double R, int ngrid, long nrays, double *matrix, long *total_rays, int NTHREADS)
{
    const Vector L = {L_x, L_y, L_z};
    const Vector C = {C_x, C_y, C_z};

    const double W_max2 = W_max * W_max;
    const double dx = (ngrid - 1) / (2 * W_max);
    const double dz = (ngrid - 1) / (2 * W_max);
    const double view_ray_constant = R * R - vector_dot_product(C, C);
    long total = 0;

#ifdef USE_OMP
#pragma omp parallel num_threads(NTHREADS)
#endif
    {
#ifdef USE_OMP
        double *p_matrix = calloc(ngrid * ngrid, sizeof(double));
#endif
        double phi, cos_phi, sin_phi, cos_theta, sin_theta;
        double dot_prod, t, b, view_ray_equation = 0;
        int i, j;

#ifdef USE_OMP
        seed_xoshiro256(123456789 + omp_get_thread_num());
#else
        seed_xoshiro256(123456789);
#endif
        Vector V, W, I, N, S;
        W.x = W.z = 0;
        W.y = W_y;

#ifdef USE_OMP
#pragma omp for reduction(+ : total)
#endif
        for (long k = 0; k < nrays; ++k)
        {
            while (view_ray_equation <= 0 || W.x * W.x >= W_max2 || W.z * W.z >= W_max2)
            {
                phi = random_double() * M_PI;
                // cos_phi = 1 - (phi * phi / 2) + (phi * phi * phi * phi / 24) - (phi * phi * phi * phi * phi * phi / 720);
                // sin_phi = phi - (phi * phi * phi / 6) + (phi * phi * phi * phi * phi / 120) - (phi * phi * phi * phi * phi * phi * phi / 5040);

                cos_theta = 2.0 * random_double() - 1.0;
                sin_theta = sqrt(1 - cos_theta * cos_theta);

                V.x = sin_theta * cos(phi);
                V.y = sin_theta * sin(phi);
                V.z = cos_theta;

                vector_multiply(W_y / V.y, V, &W);
                dot_prod = vector_dot_product(V, C);
                view_ray_equation = dot_prod * dot_prod + view_ray_constant;
                total++;
            }

            t = dot_prod - sqrt(view_ray_equation);
            vector_multiply(t, V, &I);

            vector_subtract(I, C, &N);
            vector_divide(vector_norm(N), N, &N);

            vector_subtract(L, I, &S);
            vector_divide(vector_norm(S), S, &S);

            b = (vector_dot_product(S, N) > 0) ? vector_dot_product(S, N) : 0;
            i = (W.x + W_max) * dx;
            j = (W.z + W_max) * dz;

#ifdef USE_OMP
            p_matrix[ngrid * i + j] += b;
#else
            matrix[ngrid * i + j] += b;
#endif
            view_ray_equation = 0;
        }

#ifdef USE_OMP
#pragma omp critical
        for (int i = 0; i < ngrid * ngrid; ++i)
            matrix[i] += p_matrix[i];

        free(p_matrix);
#endif
    }

    *total_rays = total;
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <NRAYS> <NGRID> <NTHREADS> \n", argv[0]);
        return EXIT_FAILURE;
    }

    long NRAYS = atol(argv[1]);
    int NGRID = atoi(argv[2]);
    int NTHREADS = atol(argv[3]);

    double *matrix = (double *)calloc(NGRID * NGRID, sizeof(double));
    long total_rays;

    double start = omp_get_wtime();
    ray_tracing(4, 4, -1, 2, 2, 0, 12, 0, 6, NGRID, NRAYS, matrix, &total_rays, NTHREADS);

    FILE *file = fopen("matrix.txt", "w");

    for (int i = 0; i < NGRID * NGRID; ++i)
        fprintf(file, "%.2lf ", matrix[i]);

    fclose(file);

    double stop = omp_get_wtime();
    double total_time = stop - start;

    printf("\nTotal Time of Execution : %lf (s)\n", total_time);
    printf("Number of Accepted Rays : %ld\n", NRAYS);
    printf("Number of Rejected Rays : %ld\n\n", total_rays - NRAYS);

    free(matrix);
    return EXIT_SUCCESS;
}
