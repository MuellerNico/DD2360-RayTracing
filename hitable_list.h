#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"
#include "precision_types.h"

class hitable_list: public hitable  {
    public:
        __device__ hitable_list() {}
        __device__ hitable_list(hitable **l, int n) {list = l; list_size = n; }
        __device__ virtual bool hit(const ray& r, real_t tmin, real_t tmax, hit_record& rec) const;
        hitable **list;
        int list_size;
};

__device__ bool hitable_list::hit(const ray& r, real_t t_min, real_t t_max, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        real_t closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
}

#endif
