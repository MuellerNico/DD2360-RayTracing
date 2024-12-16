#ifndef PRECISION_TYPES_H
#define PRECISION_TYPES_H

#include <cuda_fp16.h>

// comment for float32
#define USE_FP16


// by cuda:
// https://docs.nvidia.com/cuda/archive/11.7.1/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS
// hsqrt(x), hrsqrt(x), 

#ifdef USE_FP16
    struct real_t {
        __half val;

        // Constructors
        __host__ __device__ __forceinline__ real_t() : val(__float2half(0.0f)) {}
        __host__ __device__ __forceinline__ real_t(float f) : val(__float2half(f)) {}
        __host__ __device__ __forceinline__ real_t(const __half& h) : val(h) {}
        
        // Conversion
        __host__ __device__ __forceinline__ operator float() const { return __half2float(val); }
        
        // Basic arithmetic operators
        __device__ __forceinline__ real_t operator+(const real_t& other) const { 
            return real_t(__hadd(val, other.val)); 
        }
        
        __device__ __forceinline__ real_t operator-(const real_t& other) const { 
            return real_t(__hsub(val, other.val)); 
        }
        
        __device__ __forceinline__ real_t operator*(const real_t& other) const { 
            return real_t(__hmul(val, other.val)); 
        }
        
        __device__ __forceinline__ real_t operator/(const real_t& other) const { 
            return real_t(__hdiv(val, other.val)); 
        }
        
        // Compound assignment operators
        __device__ __forceinline__ real_t& operator+=(const real_t& other) { 
            val = __hadd(val, other.val); 
            return *this; 
        }
        
        __device__ __forceinline__ real_t& operator-=(const real_t& other) { 
            val = __hsub(val, other.val); 
            return *this; 
        }
        
        __device__ __forceinline__ real_t& operator*=(const real_t& other) { 
            val = __hmul(val, other.val); 
            return *this; 
        }
        
        __device__ __forceinline__ real_t& operator/=(const real_t& other) { 
            val = __hdiv(val, other.val); 
            return *this; 
        }
        
        // Comparison operators
        __device__ __forceinline__ bool operator<(const real_t& other) const { 
            return __hlt(val, other.val); 
        }
        
        __device__ __forceinline__ bool operator>(const real_t& other) const { 
            return __hgt(val, other.val); 
        }
        
        __device__ __forceinline__ bool operator<=(const real_t& other) const { 
            return __hle(val, other.val); 
        }
        
        __device__ __forceinline__ bool operator>=(const real_t& other) const { 
            return __hge(val, other.val); 
        }
        
        // math
        __device__ __forceinline__ static real_t sqrt(const real_t& x) { 
            return real_t(hsqrt(x.val)); 
        }

        __device__ __forceinline__ static real_t rsqrt(const real_t& x) { 
            return real_t(hrsqrt(x.val)); // 1/sqrt(x)
        }
        
    };

    // Non-member atomic operation
    __device__ __forceinline__ void atomic_add(real_t* address, real_t val) {
        atomicAdd(&(address->val), val.val);
    }

#else
    struct real_t {
        float val;

        // Constructors
        __host__ __device__ __forceinline__ real_t() : val(0.0f) {}
        __host__ __device__ __forceinline__ real_t(float f) : val(f) {}
        
        // Conversion
        __host__ __device__ __forceinline__ operator float() const { return val; }
        
        // Basic arithmetic operators
        __device__ __forceinline__ real_t operator+(const real_t& other) const { 
            return real_t(val + other.val); 
        }
        
        __device__ __forceinline__ real_t operator-(const real_t& other) const { 
            return real_t(val - other.val); 
        }
        
        __device__ __forceinline__ real_t operator*(const real_t& other) const { 
            return real_t(val * other.val); 
        }
        
        __device__ __forceinline__ real_t operator/(const real_t& other) const { 
            return real_t(val / other.val); 
        }
        
        // Compound assignment operators
        __device__ __forceinline__ real_t& operator+=(const real_t& other) { 
            val += other.val; 
            return *this; 
        }
        
        __device__ __forceinline__ real_t& operator-=(const real_t& other) { 
            val -= other.val; 
            return *this; 
        }
        
        __device__ __forceinline__ real_t& operator*=(const real_t& other) { 
            val *= other.val; 
            return *this; 
        }
        
        __device__ __forceinline__ real_t& operator/=(const real_t& other) { 
            val /= other.val; 
            return *this; 
        }
        
        // Comparison operators
        __device__ __forceinline__ bool operator<(const real_t& other) const { 
            return val < other.val; 
        }
        
        __device__ __forceinline__ bool operator>(const real_t& other) const { 
            return val > other.val; 
        }
        
        __device__ __forceinline__ bool operator<=(const real_t& other) const { 
            return val <= other.val; 
        }
        
        __device__ __forceinline__ bool operator>=(const real_t& other) const { 
            return val >= other.val; 
        }
        
        // math
        __device__ __forceinline__ static real_t sqrt(const real_t& x) { 
            return real_t(sqrtf(x.val)); 
        }
        
    };

    // Non-member atomic operation
    __device__ __forceinline__ void atomic_add(real_t* address, real_t val) {
        atomicAdd(&(address->val), val.val);
    }

#endif

// conversion
__host__ __device__ __forceinline__ real_t to_real(float x) { return real_t(x); }
__host__ __device__ __forceinline__ float from_real(real_t x) { return float(x); }

#endif // PRECISION_TYPES_H