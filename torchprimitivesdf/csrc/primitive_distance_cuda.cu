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

template<typename T>
struct ScalarTypeToVec3 { using type = void; };
template <> struct ScalarTypeToVec3<float> { using type = float3; };
template <> struct ScalarTypeToVec3<double> { using type = double3; };

template<typename T>
struct Vec3TypeToScalar { using type = void; };
template <> struct Vec3TypeToScalar<float3> { using type = float; };
template <> struct Vec3TypeToScalar<double3> { using type = double; };


__device__ __forceinline__ float3 make_vector(float x, float y, float z) {
  return make_float3(x, y, z);
}

__device__ __forceinline__ double3 make_vector(double x, double y, double z) {
  return make_double3(x, y, z);
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
__device__ __forceinline__ scalar_t min(vector_t a) {
    return std::min(a.x, std::min(a.y, a.z));
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ scalar_t max(vector_t a) {
    return std::max(a.x, std::max(a.y, a.z));
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
    for (int point_id = threadIdx.x + blockIdx.x * blockDim.x; point_id < num_points; point_id += blockDim.x * gridDim.x) {
        vector_t q = abs_vec(points[point_id]) - *box;
        vector_t q_clamped = max_vec(q, scalar_t(0));
        distances[point_id] = dot(q_clamped, q_clamped) + square(std::min(max<scalar_t, vector_t>(q), scalar_t(0)));
        dis_signs[point_id] = (q.x > 0) || (q.y > 0) || (q.z > 0);
        closest_points[point_id] = clamp_vec(points[point_id], - *box, *box);
        if (!dis_signs[point_id]) {
            int closest_face = 0;
            scalar_t* q_pointer = reinterpret_cast<scalar_t*>(&q);
            if (q_pointer[1] > q_pointer[closest_face]) closest_face = 1;
            if (q_pointer[2] > q_pointer[closest_face]) closest_face = 2;
            (reinterpret_cast<scalar_t*>(&closest_points[point_id]))[closest_face] = (reinterpret_cast<scalar_t*>(const_cast<vector_t*>(box)))[closest_face] * sign((reinterpret_cast<scalar_t*>(const_cast<vector_t*>(&points[point_id])))[closest_face]);
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
    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "box_distance_forward_cuda", [&] {
        using vector_t = ScalarTypeToVec3<scalar_t>::type;
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(points));
        auto stream = at::cuda::getCurrentCUDAStream();
        box_distance_forward_cuda_kernel<scalar_t, vector_t><<<num_blocks, num_threads, 0, stream>>>(
            reinterpret_cast<vector_t*>(points.data_ptr<scalar_t>()),
            reinterpret_cast<vector_t*>(box.data_ptr<scalar_t>()),
            points.size(0),
            distances.data_ptr<scalar_t>(),
            dis_signs.data_ptr<bool>(),
            reinterpret_cast<vector_t*>(closest_points.data_ptr<scalar_t>()));
        CUDA_CHECK(cudaGetLastError());
    });
}

void box_distance_backward_cuda_impl(
    at::Tensor grad_distances, 
    at::Tensor points, 
    at::Tensor closest_points, 
    at::Tensor grad_points) {

    DISPATCH_INPUT_TYPES(points.scalar_type(), scalar_t, "box_distance_backward_cuda", [&] {
        const int num_points = points.size(0);
        const int num_blocks = (num_points + num_threads - 1) / num_threads;
        using vector_t = ScalarTypeToVec3<scalar_t>::type;
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(points));
        auto stream = at::cuda::getCurrentCUDAStream();
        box_distance_backward_cuda_kernel<scalar_t, vector_t><<<num_blocks, num_threads, 0, stream>>>(
            grad_distances.data_ptr<scalar_t>(),
            reinterpret_cast<vector_t*>(points.data_ptr<scalar_t>()),
            reinterpret_cast<vector_t*>(closest_points.data_ptr<scalar_t>()),
            points.size(0),
            reinterpret_cast<vector_t*>(grad_points.data_ptr<scalar_t>()));
        CUDA_CHECK(cudaGetLastError());
    });
}

}  // namespace primitive

#undef PRIVATE_CASE_TYPE_AND_VAL
#undef DISPATCH_INPUT_TYPES