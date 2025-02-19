#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct Vector
{
    double x;
    double y;
    double z;
} Vector;

inline void vector_add(const Vector V, const Vector U, Vector *result)
{
    result->x = V.x + U.x;
    result->y = V.y + U.y;
    result->z = V.z + U.z;
}

inline void vector_subtract(const Vector V, const Vector U, Vector *result)
{
    result->x = V.x - U.x;
    result->y = V.y - U.y;
    result->z = V.z - U.z;
}

inline void vector_multiply(const double t, const Vector V, Vector *result)
{
    result->x = t * V.x;
    result->y = t * V.y;
    result->z = t * V.z;
}

inline void vector_divide(const double t, const Vector V, Vector *result)
{
    result->x = V.x / t;
    result->y = V.y / t;
    result->z = V.z / t;
}

inline double vector_dot_product(const Vector V, const Vector U)
{
    return V.x * U.x + V.y * U.y + V.z * U.z;
}

inline double vector_norm(const Vector V)
{
    return sqrt(vector_dot_product(V, V));
}
