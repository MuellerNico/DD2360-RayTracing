#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"
#include "precision_types.h"

class sphere: public hitable  {
    public:
        __device__ sphere() {}
        __device__ sphere(vec3 cen, real_t r, material *m) : center(cen), radius(r), mat_ptr(m)  {};
        __device__ virtual bool hit(const ray& r, real_t tmin, real_t tmax, hit_record& rec) const;
        vec3 center;
        real_t radius;
        material *mat_ptr;
};

__device__ bool sphere::hit(const ray& r, real_t t_min, real_t t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    real_t a = dot(r.direction(), r.direction());
    real_t b = dot(oc, r.direction());
    real_t c = dot(oc, oc) - radius*radius;
    real_t discriminant = b*b - a*c;
    if (discriminant > 0) {
        real_t temp = (-b - real_t::sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}


#endif
