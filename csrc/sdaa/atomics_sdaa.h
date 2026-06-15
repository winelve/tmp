#pragma once

// Shared SDAA atomic operations for sparse kernels.
// Uses __builtin_sw_slave_cas_32/64 for CAS-based atomics on arbitrary types.

#include <cstdint>
#include <type_traits>

__device__ inline bool sdaa_cas32(int32_t *a, int32_t e, int32_t d) {
  return __builtin_sw_slave_cas_32(a, e, d);
}

__device__ inline bool sdaa_cas64(int64_t *a, int64_t e, int64_t d) {
  return __builtin_sw_slave_cas_64(a, e, d);
}

template <typename T>
__device__ inline void sdaa_atomic_add(T *addr, T v) {
  constexpr size_t S = sizeof(T);
  if constexpr (S == 4) {
    int32_t *ia = (int32_t *)addr;
    int32_t old = *ia, as;
    do {
      as = old;
      T nv = __builtin_bit_cast(T, as) + v;
      old = *ia;
      if (sdaa_cas32(ia, as, __builtin_bit_cast(int32_t, nv))) break;
    } while (1);
  } else if constexpr (S == 8) {
    int64_t *ia = (int64_t *)addr;
    int64_t old = *ia, as;
    do {
      as = old;
      T nv = __builtin_bit_cast(T, as) + v;
      old = *ia;
      if (sdaa_cas64(ia, as, __builtin_bit_cast(int64_t, nv))) break;
    } while (1);
  }
}

template <typename T>
__device__ inline void sdaa_atomic_mul(T *addr, T v) {
  constexpr size_t S = sizeof(T);
  using I = std::conditional_t<S == 4, int32_t, int64_t>;
  if constexpr (S == 4 || S == 8) {
    I *ia = (I *)addr;
    I old = *ia, as;
    do {
      as = old;
      T nv = __builtin_bit_cast(T, as) * v;
      old = *ia;
      if constexpr (S == 4) {
        if (sdaa_cas32(ia, as, __builtin_bit_cast(int32_t, nv))) break;
      } else {
        if (sdaa_cas64(ia, as, __builtin_bit_cast(int64_t, nv))) break;
      }
    } while (1);
  }
}

template <typename T>
__device__ inline void sdaa_atomic_div(T *addr, T v) {
  constexpr size_t S = sizeof(T);
  using I = std::conditional_t<S == 4, int32_t, int64_t>;
  if constexpr (S == 4 || S == 8) {
    I *ia = (I *)addr;
    I old = *ia, as;
    do {
      as = old;
      T nv = __builtin_bit_cast(T, as) / v;
      old = *ia;
      if constexpr (S == 4) {
        if (sdaa_cas32(ia, as, __builtin_bit_cast(int32_t, nv))) break;
      } else {
        if (sdaa_cas64(ia, as, __builtin_bit_cast(int64_t, nv))) break;
      }
    } while (1);
  }
}

template <typename T>
__device__ inline void sdaa_atomic_min(T *addr, T v) {
  constexpr size_t S = sizeof(T);
  if constexpr (S == 4 || S == 8) {
    using I = std::conditional_t<S == 4, int32_t, int64_t>;
    I *ia = (I *)addr;
    I old = *ia, as;
    do {
      as = old;
      T cur = __builtin_bit_cast(T, as);
      T nv = (v < cur) ? v : cur;
      old = *ia;
      if constexpr (S == 4) {
        if (sdaa_cas32(ia, as, __builtin_bit_cast(int32_t, nv))) break;
      } else {
        if (sdaa_cas64(ia, as, __builtin_bit_cast(int64_t, nv))) break;
      }
    } while (1);
  }
}

template <typename T>
__device__ inline void sdaa_atomic_max(T *addr, T v) {
  constexpr size_t S = sizeof(T);
  if constexpr (S == 4 || S == 8) {
    using I = std::conditional_t<S == 4, int32_t, int64_t>;
    I *ia = (I *)addr;
    I old = *ia, as;
    do {
      as = old;
      T cur = __builtin_bit_cast(T, as);
      T nv = (v > cur) ? v : cur;
      old = *ia;
      if constexpr (S == 4) {
        if (sdaa_cas32(ia, as, __builtin_bit_cast(int32_t, nv))) break;
      } else {
        if (sdaa_cas64(ia, as, __builtin_bit_cast(int64_t, nv))) break;
      }
    } while (1);
  }
}
