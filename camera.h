#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>
#include "ray.h"
#include "precision_types.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ vec3 random_in_unit_disk(curandState *local_rand_state) {
    vec3 p;
    do {
        p = real_t(2.0f)*vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),0) - vec3(1,1,0);
    } while (dot(p,p) >= real_t(1.0f));
    return p;
}

class camera {
public:
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, real_t vfov, real_t aspect, real_t aperture, real_t focus_dist) { // vfov is top to bottom in degrees
        lens_radius = aperture / real_t(2.0f);
        real_t theta = vfov*((real_t)M_PI)/real_t(180.0f);
        // real_t half_height = tan(theta/2.0f);
        real_t arg = theta/real_t(2.0f);
#ifdef __CUDA_ARCH__
        real_t half_height = hsin(arg.val) / hcos(arg.val); // no tan function in nvidia fp16 math
#else
        real_t half_height = tan(arg);
#endif
        real_t half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin  - half_width*focus_dist*u -half_height*focus_dist*v - focus_dist*w;
        horizontal = real_t(2.0f)*half_width*focus_dist*u;
        vertical = real_t(2.0f)*half_height*focus_dist*v;
    }
    __device__ ray get_ray(real_t s, real_t t, curandState *local_rand_state) {
        vec3 rd = lens_radius*random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
    }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    real_t lens_radius;
};

#endif
