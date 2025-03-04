#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include "ray.h"
#include "hitable.h"
#include "precision_types.h"


__device__ real_t schlick(real_t cosine, real_t ref_idx) {
    real_t r0 = real_t(1.0f-ref_idx) / real_t(1.0f+ref_idx);
    r0 = r0*r0;
    return real_t(r0 + real_t(1.0f-r0)*real_t(pow((1.0f - cosine),5.0f))); // TODO: pow computed in 32bit for now
}

__device__ bool refract(const vec3& v, const vec3& n, real_t ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    real_t dt = dot(uv, n);
    real_t discriminant = real_t(1.0f) - ni_over_nt*ni_over_nt*((real_t)1.0f-dt*dt);
    if (discriminant > real_t(0)) {
#ifdef USE_FP16
        refracted = ni_over_nt*(uv - n*dt) - n*real_t::sqrt(discriminant);
#else
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
#endif
        return true;
    }
    else
        return false;
}

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = real_t(2.0f)*RANDVEC3 - vec3(1,1,1);
    } while (p.squared_length() >= real_t(1.0f));
    return p;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
     return v - real_t(2.0f)*dot(v,n)*n;
}

class material  {
    public:
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const = 0;
};

class lambertian : public material {
    public:
        __device__ lambertian(const vec3& a) : albedo(a) {}
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
             vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
             scattered = ray(rec.p, target-rec.p);
             attenuation = albedo;
             return true;
        }

        vec3 albedo;
};

class metal : public material {
    public:
        __device__ metal(const vec3& a, real_t f) : albedo(a) { if (f < real_t(1.0f)) fuzz = f; else fuzz = real_t(1.0f); }
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > real_t(0.0f));
        }
        vec3 albedo;
        real_t fuzz;
};

class dielectric : public material {
public:
    __device__ dielectric(real_t ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const ray& r_in,
                         const hit_record& rec,
                         vec3& attenuation,
                         ray& scattered,
                         curandState *local_rand_state) const  {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        real_t ni_over_nt;
        attenuation = vec3(1.0, 1.0, 1.0);
        vec3 refracted;
        real_t reflect_prob;
        real_t cosine;
        if (dot(r_in.direction(), rec.normal) > real_t(0.0f)) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrt(real_t(1.0f) - ref_idx*ref_idx*real_t((real_t(1.0f)-cosine*cosine)));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = real_t(1.0f) / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = real_t(1.0f);
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

    real_t ref_idx;
};
#endif
