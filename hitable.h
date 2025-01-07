#ifndef HITABLEH
#define HITABLEH

#include "ray.h"
#include "precision_types.h"

class material;

struct hit_record
{
    real_t t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
};

class hitable  {
    public:
        __host__ __device__ virtual bool hit(const ray& r, real_t t_min, real_t t_max, hit_record& rec) const = 0;
};

#endif
