#include <math.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <THC/THCAtomics.cuh>
#include "utils.h"

#define PRIVATE_CASE_TYPE_AND_VAL(ENUM_TYPE, TYPE, TYPE_NAME, VAL, ...) \
  case ENUM_TYPE: { \
    using TYPE_NAME = TYPE; \
    const int num_threads = VAL; \
    return __VA_ARGS__(); \
  }


#define DISPATCH_INPUT_TYPES(TYPE, TYPE_NAME, SCOPE_NAME, ...) \
  [&] { \
    switch(TYPE) \
    { \
      PRIVATE_CASE_TYPE_AND_VAL(at::ScalarType::Float, float, TYPE_NAME, 1024, __VA_ARGS__) \
      PRIVATE_CASE_TYPE_AND_VAL(at::ScalarType::Double, double, TYPE_NAME, 512, __VA_ARGS__) \
      default: \
        AT_ERROR(#SCOPE_NAME, " not implemented for '", toString(TYPE), "'"); \
    } \
  }()


namespace primitive {

struct my_float3 {
    float x, y, z;
    __device__ __forceinline__ float operator[](int i) const { 
        switch (i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
        };
        return x;
    };
    __device__ __forceinline__ float& operator[](int i) {
        switch (i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
        };
        return x;
    }
};

struct my_double3 {
    double x, y, z;
    __device__ __forceinline__ double operator[](int i) const { 
        switch (i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
        };
        return x;
    };
    __device__ __forceinline__ double& operator[](int i) {
        switch (i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
        };
        return x;
    }
};

__device__ __forceinline__ my_float3 make_my_float3(float x, float y, float z)
{
  my_float3 t; t.x = x; t.y = y; t.z = z; return t;
}

__device__ __forceinline__ my_double3 make_my_double3(double x, double y, double z)
{
  my_double3 t; t.x = x; t.y = y; t.z = z; return t;
}

template<typename T>
struct ScalarTypeToVec3 { using type = void; };
template <> struct ScalarTypeToVec3<float> { using type = my_float3; };
template <> struct ScalarTypeToVec3<double> { using type = my_double3; };

template<typename T>
struct Vec3TypeToScalar { using type = void; };
template <> struct Vec3TypeToScalar<my_float3> { using type = float; };
template <> struct Vec3TypeToScalar<my_double3> { using type = double; };


__device__ __forceinline__ my_float3 make_vector(float x, float y, float z) {
  return make_my_float3(x, y, z);
}

__device__ __forceinline__ my_double3 make_vector(double x, double y, double z) {
  return make_my_double3(x, y, z);
}

template <typename vector_t>
__device__ __forceinline__ typename Vec3TypeToScalar<vector_t>::type dot(vector_t a, vector_t b) {
  return a.x * b.x + a.y * b.y + a.z * b.z ;
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ scalar_t dot2(vector_t v) {
  return dot<scalar_t, vector_t>(v, v);
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t clamp(scalar_t x, scalar_t a, scalar_t b) {
  return max(a, min(b, x));
}

template<typename vector_t>
__device__ __forceinline__ vector_t clamp_vec(vector_t x, vector_t a, vector_t b) {
  return make_vector(clamp(x.x, a.x, b.x), clamp(x.y, a.y, b.y), clamp(x.z, a.z, b.z));
}

template<typename scalar_t>
__device__ __forceinline__ int sign(scalar_t a) {
  if (a <= 0) {return -1;}
  else {return 1;}
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ vector_t operator* (vector_t a, scalar_t b) {
  return make_vector(a.x * b, a.y * b, a.z * b);
}

template<typename vector_t>
__device__ __forceinline__ vector_t operator* (vector_t a, vector_t b) {
  return make_vector(a.x * b.x, a.y * b.y, a.z * b.z);
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ vector_t operator+ (vector_t a, scalar_t b) {
  return make_vector(a.x + b, a.y + b, a.z + b);
}

template<typename vector_t>
__device__ __forceinline__ vector_t operator+ (vector_t a, vector_t b) {
  return make_vector(a.x + b.x, a.y + b.y, a.z + b.z);
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ vector_t operator- (vector_t a, scalar_t b) {
  return make_vector(a.x - b, a.y - b, a.z - b);
}

template<typename vector_t>
__device__ __forceinline__ vector_t operator- (vector_t a, vector_t b) {
  return make_vector(a.x - b.x, a.y - b.y, a.z - b.z);
}

template<typename vector_t>
__device__ __forceinline__ vector_t operator- (vector_t a) {
  return make_vector(-a.x, -a.y, -a.z);
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ vector_t operator/ (vector_t a, scalar_t b) {
  return make_vector(a.x / b, a.y / b, a.z / b);
}

template<typename vector_t>
__device__ __forceinline__ vector_t operator/ (vector_t a, vector_t b) {
  return make_vector(a.x / b.x, a.y / b.y, a.z / b.z);
}

template<typename vector_t>
__device__ __forceinline__ vector_t abs_vec(vector_t a) {
    return make_vector(abs(a.x), abs(a.y), abs(a.z));
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ vector_t max_vec(vector_t a, scalar_t b) {
    return make_vector(max(a.x, b), max(a.y, b), max(a.z, b));
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ vector_t min_vec(vector_t a, scalar_t b) {
    return make_vector(min(a.x, b), min(a.y, b), min(a.z, b));
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ scalar_t min_vec(vector_t a) {
    return min(a.x, min(a.y, a.z));
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ scalar_t max_vec(vector_t a) {
    return max(a.x, max(a.y, a.z));
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t square(scalar_t a) {
    return a * a;
}


template<typename scalar_t, typename vector_t>
__global__ void box_distance_forward_cuda_kernel(
    const vector_t* points,
    const vector_t* box,
    int num_points,
    scalar_t* distances,
    bool* dis_signs,
    vector_t* closest_points) {
    vector_t vbox = *box;
    for (int point_id = threadIdx.x + blockIdx.x * blockDim.x; point_id < num_points; point_id += blockDim.x * gridDim.x) {
        vector_t q = abs_vec(points[point_id]) - vbox;
        vector_t q_clamped = max_vec(q, scalar_t(0));
        distances[point_id] = dot(q_clamped, q_clamped) + square(min(max_vec<scalar_t, vector_t>(q), scalar_t(0)));
        dis_signs[point_id] = (q.x > 0) || (q.y > 0) || (q.z > 0);
        closest_points[point_id] = clamp_vec(points[point_id], - vbox, vbox);
        if (!dis_signs[point_id]) {
            int closest_face = 0;
            if (q[1] > q[closest_face]) closest_face = 1;
            if (q[2] > q[closest_face]) closest_face = 2;
            closest_points[point_id][closest_face] = vbox[closest_face] * sign(points[point_id][closest_face]);
        }
    }
}


template<typename scalar_t, typename vector_t>
__global__ void box_distance_backward_cuda_kernel(
    const scalar_t* grad_dist,
    const vector_t* points,
    const vector_t* clst_points,
    int num_points,
    vector_t* grad_points) {
    for (int point_id = threadIdx.x + blockIdx.x * blockDim.x; point_id < num_points; point_id += blockDim.x * gridDim.x) {
        // scalar_t grad_out = 2. * grad_dist[point_id];
        // vector_t dist_vec = points[point_id] - clst_points[point_id];
        // dist_vec = dist_vec * grad_out;
        // grad_points[point_id] = dist_vec;
        grad_points[point_id] = (points[point_id] - clst_points[point_id]) * (scalar_t(2) * grad_dist[point_id]);
    }
}

void box_distance_forward_cuda_impl(
    at::Tensor points, 
    at::Tensor box, 
    at::Tensor distances, 
    at::Tensor dis_signs, 
    at::Tensor closest_points) {
    const int num_threads = 512;
    const int num_points = points.size(0);
    const int num_blocks = (num_points + num_threads - 1) / num_threads;
    using scalar_t = float;
    using vector_t = ScalarTypeToVec3<scalar_t>::type;
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(points));
    box_distance_forward_cuda_kernel<scalar_t, vector_t><<<num_blocks, num_threads>>>(
        reinterpret_cast<vector_t*>(points.data_ptr<scalar_t>()),
        reinterpret_cast<vector_t*>(box.data_ptr<scalar_t>()),
        points.size(0),
        distances.data_ptr<scalar_t>(),
        dis_signs.data_ptr<bool>(),
        reinterpret_cast<vector_t*>(closest_points.data_ptr<scalar_t>()));
    CUDA_CHECK(cudaGetLastError());
}

void box_distance_backward_cuda_impl(
    at::Tensor grad_distances, 
    at::Tensor points, 
    at::Tensor closest_points, 
    at::Tensor grad_points) {

    const int num_threads = 512;
    const int num_points = points.size(0);
    const int num_blocks = (num_points + num_threads - 1) / num_threads;
    using scalar_t = float;
    using vector_t = ScalarTypeToVec3<scalar_t>::type;
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(points));
    box_distance_backward_cuda_kernel<scalar_t, vector_t><<<num_blocks, num_threads>>>(
        grad_distances.data_ptr<scalar_t>(),
        reinterpret_cast<vector_t*>(points.data_ptr<scalar_t>()),
        reinterpret_cast<vector_t*>(closest_points.data_ptr<scalar_t>()),
        points.size(0),
        reinterpret_cast<vector_t*>(grad_points.data_ptr<scalar_t>()));
    CUDA_CHECK(cudaGetLastError());
}

template<typename scalar_t>
__global__ void transform_points_inverse_forward_cuda_kernel(
    const scalar_t* points,
    const scalar_t* translations,
    const scalar_t* rotations,
    int batch_size, 
    int num_points, 
    scalar_t* points_transformed) {
    __shared__ scalar_t translation[3];
    __shared__ scalar_t rotation[9];
    // store translation and rotation into shared memory
    int batch_id = blockIdx.x;
    if (threadIdx.y < 9) {
        rotation[threadIdx.y] = rotations[batch_id * 9 + threadIdx.y];
    }
    if (threadIdx.y < 3) {
        translation[threadIdx.y] = translations[batch_id * 3 + threadIdx.y];
    }
    __syncthreads();
    // translate and rotate points
    for (int point_id = threadIdx.y + blockIdx.y * blockDim.y; point_id < num_points; point_id += blockDim.y * gridDim.y) {
        int point_idx = batch_id * num_points * 3 + point_id * 3;
        scalar_t x = points[point_idx + 0] - translation[0];
        scalar_t y = points[point_idx + 1] - translation[1];
        scalar_t z = points[point_idx + 2] - translation[2];
        points_transformed[point_idx + 0] = x * rotation[0] + y * rotation[3] + z * rotation[6];
        points_transformed[point_idx + 1] = x * rotation[1] + y * rotation[4] + z * rotation[7];
        points_transformed[point_idx + 2] = x * rotation[2] + y * rotation[5] + z * rotation[8];
    }
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile float sdata[], const unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template<typename scalar_t, unsigned int blockSize>
__global__ void transform_points_inverse_backward_cuda_kernel(
    const scalar_t* grad_points_transformed,
    const scalar_t* points,
    const scalar_t* translations,
    const scalar_t* rotations,
    int batch_size,
    int num_points,
    scalar_t* grad_points,
    scalar_t* grad_translations,
    scalar_t* grad_rotations) {
    extern __shared__ scalar_t sdata[];
    scalar_t *translation = &sdata[blockDim.y];
    scalar_t *rotation = &translation[3];
    // store translation and rotation into shared memory
    int batch_id = blockIdx.x;
    if (threadIdx.y < 9) {
        rotation[threadIdx.y] = rotations[batch_id * 9 + threadIdx.y];
    }
    if (threadIdx.y < 3) {
        translation[threadIdx.y] = translations[batch_id * 3 + threadIdx.y];
    }
    __syncthreads();
    // compute gradients
    scalar_t grad_translations_broadcasted[] = { 0, 0, 0 };
    scalar_t grad_rotations_broadcasted[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int point_id = threadIdx.y + blockIdx.y * blockDim.y; point_id < num_points; point_id += blockDim.y * gridDim.y) {
        int point_idx = batch_id * num_points * 3 + point_id * 3;
        scalar_t dx = grad_points_transformed[point_idx + 0];
        scalar_t dy = grad_points_transformed[point_idx + 1];
        scalar_t dz = grad_points_transformed[point_idx + 2];
        scalar_t x = points[point_idx + 0] - translation[0];
        scalar_t y = points[point_idx + 1] - translation[1];
        scalar_t z = points[point_idx + 2] - translation[2];
        scalar_t gx = dx * rotation[0] + dy * rotation[1] + dz * rotation[2];
        scalar_t gy = dx * rotation[3] + dy * rotation[4] + dz * rotation[5];
        scalar_t gz = dx * rotation[6] + dy * rotation[7] + dz * rotation[8];
        grad_rotations_broadcasted[0] += x * dx;
        grad_rotations_broadcasted[1] += x * dy;
        grad_rotations_broadcasted[2] += x * dz;
        grad_rotations_broadcasted[3] += y * dx;
        grad_rotations_broadcasted[4] += y * dy;
        grad_rotations_broadcasted[5] += y * dz;
        grad_rotations_broadcasted[6] += z * dx;
        grad_rotations_broadcasted[7] += z * dy;
        grad_rotations_broadcasted[8] += z * dz;
        grad_points[point_idx + 0] = gx;
        grad_points[point_idx + 1] = gy;
        grad_points[point_idx + 2] = gz;
        grad_translations_broadcasted[0] -= gx;
        grad_translations_broadcasted[1] -= gy;
        grad_translations_broadcasted[2] -= gz;
    }
    // reduce results
    for (int i = 0; i < 3; i++) {
        sdata[threadIdx.y] = grad_translations_broadcasted[i];
        __syncthreads();
        if (blockSize >= 512) { if (threadIdx.y < 256) { sdata[threadIdx.y] += sdata[threadIdx.y + 256]; } __syncthreads(); }
        if (blockSize >= 256) { if (threadIdx.y < 128) { sdata[threadIdx.y] += sdata[threadIdx.y + 128]; } __syncthreads(); }
        if (blockSize >= 128) { if (threadIdx.y < 64) { sdata[threadIdx.y] += sdata[threadIdx.y + 64]; } __syncthreads(); }
        if (threadIdx.y < 32) warpReduce<blockSize>(sdata, threadIdx.y);
        if (threadIdx.y == 0) grad_translations[batch_id * 3 + i] = sdata[0];
        __syncthreads();
    }
    for (int i = 0; i < 9; i++) {
        sdata[threadIdx.y] = grad_rotations_broadcasted[i];
        __syncthreads();
        if (blockSize >= 512) { if (threadIdx.y < 256) { sdata[threadIdx.y] += sdata[threadIdx.y + 256]; } __syncthreads(); }
        if (blockSize >= 256) { if (threadIdx.y < 128) { sdata[threadIdx.y] += sdata[threadIdx.y + 128]; } __syncthreads(); }
        if (blockSize >= 128) { if (threadIdx.y < 64) { sdata[threadIdx.y] += sdata[threadIdx.y + 64]; } __syncthreads(); }
        if (threadIdx.y < 32) warpReduce<blockSize>(sdata, threadIdx.y);
        if (threadIdx.y == 0) grad_rotations[batch_id * 9 + i] = sdata[0];
        __syncthreads();
    }
}

void transform_points_inverse_forward_cuda_impl(
    at::Tensor points,
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor points_transformed) {
    
    const int batch_size = points.size(0);
    const int num_points = points.size(1);
    dim3 num_threads(1, 512);
    dim3 num_blocks(batch_size, 1);
    using scalar_t = float;
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(points));
    transform_points_inverse_forward_cuda_kernel<scalar_t><<<num_blocks, num_threads>>>(
        points.data_ptr<scalar_t>(),
        translations.data_ptr<scalar_t>(),
        rotations.data_ptr<scalar_t>(),
        batch_size,
        num_points,
        points_transformed.data_ptr<scalar_t>());
    CUDA_CHECK(cudaGetLastError());
}

void transform_points_inverse_backward_cuda_impl(
    at::Tensor grad_points_transformed, 
    at::Tensor points, 
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor grad_points,
    at::Tensor grad_translations,
    at::Tensor grad_rotations) {

    const int batch_size = points.size(0);
    const int num_points = points.size(1);
    dim3 num_threads(1, 512);
    dim3 num_blocks(batch_size, 1);
    using scalar_t = float;
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(points));
    transform_points_inverse_backward_cuda_kernel<scalar_t, 512><<<num_blocks, num_threads, (num_threads.y + 12) * sizeof(scalar_t)>>>(
        grad_points_transformed.data_ptr<scalar_t>(),
        points.data_ptr<scalar_t>(),
        translations.data_ptr<scalar_t>(),
        rotations.data_ptr<scalar_t>(),
        batch_size,
        num_points,
        grad_points.data_ptr<scalar_t>(),
        grad_translations.data_ptr<scalar_t>(),
        grad_rotations.data_ptr<scalar_t>());
    CUDA_CHECK(cudaGetLastError());
}

template<typename scalar_t>
__global__ void fixed_transform_points_inverse_forward_cuda_kernel(
    const scalar_t* points,
    const scalar_t* translations,
    const scalar_t* rotations,
    int num_points, 
    scalar_t* points_transformed) {
    __shared__ scalar_t translation[3];
    __shared__ scalar_t rotation[9];
    // store translation and rotation into shared memory
    if (threadIdx.x < 9) {
        rotation[threadIdx.x] = rotations[threadIdx.x];
    }
    if (threadIdx.x < 3) {
        translation[threadIdx.x] = translations[threadIdx.x];
    }
    __syncthreads();
    // translate and rotate points
    for (int point_id = threadIdx.x + blockIdx.x * blockDim.x; point_id < num_points; point_id += blockDim.x * gridDim.x) {
        int point_idx = point_id * 3;
        scalar_t x = points[point_idx + 0] - translation[0];
        scalar_t y = points[point_idx + 1] - translation[1];
        scalar_t z = points[point_idx + 2] - translation[2];
        points_transformed[point_idx + 0] = x * rotation[0] + y * rotation[3] + z * rotation[6];
        points_transformed[point_idx + 1] = x * rotation[1] + y * rotation[4] + z * rotation[7];
        points_transformed[point_idx + 2] = x * rotation[2] + y * rotation[5] + z * rotation[8];
    }
}

template<typename scalar_t>
__global__ void fixed_transform_points_inverse_backward_cuda_kernel(
    const scalar_t* grad_points_transformed,
    const scalar_t* points,
    const scalar_t* translations,
    const scalar_t* rotations,
    int num_points,
    scalar_t* grad_points) {
    __shared__ scalar_t rotation[9];
    // store translation and rotation into shared memory
    if (threadIdx.x < 9) {
        rotation[threadIdx.x] = rotations[threadIdx.x];
    }
    __syncthreads();
    // compute gradients
    for (int point_id = threadIdx.x + blockIdx.x * blockDim.x; point_id < num_points; point_id += blockDim.x * gridDim.x) {
        int point_idx = point_id * 3;
        scalar_t dx = grad_points_transformed[point_idx + 0];
        scalar_t dy = grad_points_transformed[point_idx + 1];
        scalar_t dz = grad_points_transformed[point_idx + 2];
        grad_points[point_idx + 0] = dx * rotation[0] + dy * rotation[1] + dz * rotation[2];
        grad_points[point_idx + 1] = dx * rotation[3] + dy * rotation[4] + dz * rotation[5];
        grad_points[point_idx + 2] = dx * rotation[6] + dy * rotation[7] + dz * rotation[8];
    }
}

void fixed_transform_points_inverse_forward_cuda_impl(
    at::Tensor points,
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor points_transformed) {

    const int num_points = points.size(0);
    int num_threads = 512;
    int num_blocks = (num_points + num_threads - 1) / num_threads;
    using scalar_t = float;
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(points));
    fixed_transform_points_inverse_forward_cuda_kernel<scalar_t><<<num_blocks, num_threads>>>(
        points.data_ptr<scalar_t>(),
        translations.data_ptr<scalar_t>(),
        rotations.data_ptr<scalar_t>(),
        num_points,
        points_transformed.data_ptr<scalar_t>());
    CUDA_CHECK(cudaGetLastError());
}

void fixed_transform_points_inverse_backward_cuda_impl(
    at::Tensor grad_points_transformed, 
    at::Tensor points, 
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor grad_points) {

    const int num_points = points.size(0);
    int num_threads = 512;
    int num_blocks = (num_points + num_threads - 1) / num_threads;
    using scalar_t = float;
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(points));
    fixed_transform_points_inverse_backward_cuda_kernel<scalar_t><<<num_blocks, num_threads>>>(
        grad_points_transformed.data_ptr<scalar_t>(),
        points.data_ptr<scalar_t>(),
        translations.data_ptr<scalar_t>(),
        rotations.data_ptr<scalar_t>(),
        num_points,
        grad_points.data_ptr<scalar_t>());
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace primitive

#undef PRIVATE_CASE_TYPE_AND_VAL
#undef DISPATCH_INPUT_TYPES
