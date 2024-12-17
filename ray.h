#ifndef RAYH
#define RAYH
#include "vec3.h"
#include "precision_types.h"

class ray
{
    public:
        __device__ ray() {}
        __device__ ray(const vec3& a, const vec3& b) { A = a; B = b; }
        __device__ vec3 origin() const       { return A; }
        __device__ vec3 direction() const    { return B; }
        __device__ vec3 point_at_parameter(real_t t) const { return A + t*B; }

        vec3 A;
        vec3 B;
};

#endif
