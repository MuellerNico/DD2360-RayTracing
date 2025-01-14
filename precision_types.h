#ifndef PRECISION_TYPES_H
#define PRECISION_TYPES_H

#include <cuda_fp16.h>
#include <fstream>

// comment for float32
// #define USE_FP16

// by cuda:
// https://docs.nvidia.com/cuda/archive/11.7.1/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS

#ifdef USE_FP16

/// @brief floating point type wrapper for NVIDIA __half type. Supports basic arithmetic and logical operations. CPU and GPU compatible through the use of __CUDA_ARCH__ macro. Every operation is performed in FP32 on the CPU.
struct real_t
{
    __half val;

    // Constructors
    __host__ __device__ __forceinline__ real_t() : val(__float2half(0.0f)) {}
    __host__ __device__ __forceinline__ real_t(float f) : val(__float2half(f)) {}
    __host__ __device__ __forceinline__ real_t(double f) : val(__float2half(f)) {}
    __host__ __device__ __forceinline__ real_t(int i) : val(__float2half(float(i))) {}
    __host__ __device__ __forceinline__ real_t(const __half& h) : val(h) {}

    // Conversion
    __host__ __device__ __forceinline__ operator float() const { return __half2float(val); }

    // Basic arithmetic operators
    __host__ __device__ __forceinline__ real_t operator+(const real_t &other) const
    {
#ifdef __CUDA_ARCH__
        return real_t(__hadd(val, other.val));
#else
        return real_t(__half2float(val) + __half2float(other.val));
#endif
    }

    __host__ __device__ __forceinline__ real_t operator-(const real_t &other) const
    {
#ifdef __CUDA_ARCH__
        return real_t(__hsub(val, other.val));
#else
        return real_t(__half2float(val) - __half2float(other.val));
#endif
    }

    __host__ __device__ __forceinline__ real_t operator*(const real_t &other) const
    {
#ifdef __CUDA_ARCH__
        return real_t(__hmul(val, other.val));
#else
        return real_t(__half2float(val) * __half2float(other.val));
#endif
    }

    __host__ __device__ __forceinline__ real_t operator/(const real_t &other) const
    {
#ifdef __CUDA_ARCH__
        return real_t(__hdiv(val, other.val));
#else
        return real_t(__half2float(val) / __half2float(other.val));
#endif
    }

    // Compound assignment operators
    __host__ __device__ __forceinline__ real_t &operator+=(const real_t &other)
    {
#ifdef __CUDA_ARCH__
        val = __hadd(val, other.val);
#else
        val = __float2half(__half2float(val) + __half2float(other.val));
#endif
        return *this;
    }

    __host__ __device__ __forceinline__ real_t &operator-=(const real_t &other)
    {
#ifdef __CUDA_ARCH__
        val = __hsub(val, other.val);
#else
        val = __float2half(__half2float(val) - __half2float(other.val));
#endif
        return *this;
    }

    __host__ __device__ __forceinline__ real_t &operator*=(const real_t &other)
    {
#ifdef __CUDA_ARCH__
        val = __hmul(val, other.val);
#else
        val = __float2half(__half2float(val) * __half2float(other.val));
#endif
        return *this;
    }

    __host__ __device__ __forceinline__ real_t &operator/=(const real_t &other)
    {
#ifdef __CUDA_ARCH__
        val = __hdiv(val, other.val);
#else
        val = __float2half(__half2float(val) / __half2float(other.val));
#endif
        return *this;
    }

    // Comparison operators
    __host__ __device__ __forceinline__ bool operator<(const real_t &other) const
    {
#ifdef __CUDA_ARCH__
        return __hlt(val, other.val);
#else
        return __half2float(val) < __half2float(other.val);
#endif
    }

    __host__ __device__ __forceinline__ bool operator>(const real_t &other) const
    {
#ifdef __CUDA_ARCH__
        return __hgt(val, other.val);
#else
        return __half2float(val) > __half2float(other.val);
#endif
    }

    __host__ __device__ __forceinline__ bool operator<=(const real_t &other) const
    {
#ifdef __CUDA_ARCH__
        return __hle(val, other.val);
#else
        return __half2float(val) <= __half2float(other.val);
#endif
    }

    __host__ __device__ __forceinline__ bool operator>=(const real_t &other) const
    {
#ifdef __CUDA_ARCH__
        return __hge(val, other.val);
#else
        return __half2float(val) >= __half2float(other.val);
#endif
    }

    // math
    __host__ __device__ __forceinline__ static real_t sqrt(const real_t &x)
    {
#ifdef __CUDA_ARCH__
        return real_t(hsqrt(x.val));
#else
        return real_t(sqrtf(__half2float(x.val)));
#endif
    }

    __host__ __device__ __forceinline__ static real_t rsqrt(const real_t &x)
    {
#ifdef __CUDA_ARCH__
        return real_t(hrsqrt(x.val));
#else
        return real_t(1.0f / sqrtf(__half2float(x.val)));
#endif
    }
};

__host__ __device__ __forceinline__ std::ostream& operator<<(std::ostream& os, const real_t& x) {
    os << __half2float(x.val);
    return os;
}

__host__ __device__ __forceinline__ std::istream& operator>>(std::istream& is, real_t& x) {
    float temp;
    is >> temp;
    x = real_t(temp);
    return is;
}

#else
// Else use default float32 data type, no wrapper needed.
typedef float real_t;

#endif

// conversion
__host__ __device__ __forceinline__ real_t to_real(float x) { return real_t(x); }
__host__ __device__ __forceinline__ float from_real(real_t x) { return float(x); }

#endif // PRECISION_TYPES_H
