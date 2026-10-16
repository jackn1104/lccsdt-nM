#include "localized_cc/cuda_triples.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace localized_cc {
namespace {

constexpr int threads_per_block = 256;
constexpr std::array<std::array<std::size_t, 3>, 6> permutations{{
    {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}},
    {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}}}};
constexpr std::array<double, 6> permutation_sign{{1.0, -1.0, -1.0,
                                                  1.0, 1.0, -1.0}};

void require_cuda(const cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

void require_cublas(const cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) +
                             ": cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
}

std::size_t checked_bytes(const std::size_t count) {
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(double))
    throw std::overflow_error("CUDA triples allocation size overflows");
  return count * sizeof(double);
}

std::size_t checked_add(const std::size_t first, const std::size_t second) {
  if (first > std::numeric_limits<std::size_t>::max() - second)
    throw std::overflow_error("CUDA triples workspace size overflows");
  return first + second;
}

std::size_t checked_product(const std::size_t first,
                            const std::size_t second) {
  if (first != 0 &&
      second > std::numeric_limits<std::size_t>::max() / first)
    throw std::overflow_error("CUDA CCSD tile dimension overflows");
  return first * second;
}

int checked_cublas_dimension(const std::size_t value) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("CUDA triples tile exceeds cuBLAS integer range");
  return static_cast<int>(value);
}

int parse_nonnegative_environment(const char* name) {
  const char* text = std::getenv(name);
  if (text == nullptr || *text == '\0')
    return -1;
  errno = 0;
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value < 0 ||
      value > std::numeric_limits<int>::max())
    throw std::runtime_error(std::string("invalid ") + name + " value");
  return static_cast<int>(value);
}

int select_device(const int requested, const int count) {
  if (count < 1)
    throw std::runtime_error("no visible CUDA device");
  int selected = requested;
  if (selected < 0)
    selected = parse_nonnegative_environment("CCSD_CUDA_DEVICE");
  if (selected < 0) {
    const int local_rank = parse_nonnegative_environment("SLURM_LOCALID");
    selected = local_rank < 0 ? 0 : local_rank % count;
  }
  if (selected >= count)
    throw std::runtime_error("selected CUDA device is not visible");
  return selected;
}

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(const std::size_t count) : count_(count) {
    if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::overflow_error("CUDA buffer size overflows");
    if (count_ != 0)
      require_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                              count_ * sizeof(T)),
                   "cudaMalloc");
  }
  ~DeviceBuffer() {
    if (pointer_ != nullptr)
      cudaFree(pointer_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : pointer_(other.pointer_), count_(other.count_) {
    other.pointer_ = nullptr;
    other.count_ = 0;
  }
  [[nodiscard]] T* get() noexcept { return pointer_; }
  [[nodiscard]] const T* get() const noexcept { return pointer_; }

private:
  T* pointer_{};
  std::size_t count_{};
};

template <typename T> class ReusableDeviceBuffer {
public:
  ReusableDeviceBuffer() = default;
  ~ReusableDeviceBuffer() { reset(); }
  ReusableDeviceBuffer(const ReusableDeviceBuffer&) = delete;
  ReusableDeviceBuffer& operator=(const ReusableDeviceBuffer&) = delete;

  void reserve(const std::size_t count) {
    if (count <= capacity_)
      return;
    reset();
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::overflow_error("reusable CUDA buffer size overflows");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                            count * sizeof(T)),
                 "cudaMalloc reusable TCE workspace");
    capacity_ = count;
  }

  void reset() noexcept {
    if (pointer_ != nullptr)
      cudaFree(pointer_);
    pointer_ = nullptr;
    capacity_ = 0;
  }

  [[nodiscard]] T* get() noexcept { return pointer_; }
  [[nodiscard]] const T* get() const noexcept { return pointer_; }

private:
  T* pointer_{};
  std::size_t capacity_{};
};

class TceCudaWorkspace {
public:
  void select_device(const int device) {
    if (device_ == device)
      return;
    reset();
    device_ = device;
  }

  void reset() noexcept {
    packed.reset();
    raw.reset();
    task_doubles.reset();
    task_singles.reset();
    doubles.reset();
    left.reset();
    partial.reset();
    invalid.reset();
    for (auto& buffer : energies)
      buffer.reset();
  }

  ReusableDeviceBuffer<double> packed;
  ReusableDeviceBuffer<double> raw;
  ReusableDeviceBuffer<double> task_doubles;
  ReusableDeviceBuffer<double> task_singles;
  ReusableDeviceBuffer<double> doubles;
  ReusableDeviceBuffer<double> left;
  ReusableDeviceBuffer<double> partial;
  ReusableDeviceBuffer<int> invalid;
  std::array<ReusableDeviceBuffer<double>, 3> energies;

private:
  int device_{-1};
};

class PersistentCublasHandle {
public:
  PersistentCublasHandle() = default;
  ~PersistentCublasHandle() {
    if (handle_ != nullptr)
      cublasDestroy(handle_);
  }
  PersistentCublasHandle(const PersistentCublasHandle&) = delete;
  PersistentCublasHandle& operator=(const PersistentCublasHandle&) = delete;

  [[nodiscard]] cublasHandle_t get(const int device) {
    if (handle_ != nullptr && device_ != device) {
      require_cublas(cublasDestroy(handle_), "cublasDestroy");
      handle_ = nullptr;
      device_ = -1;
    }
    if (handle_ == nullptr) {
      require_cublas(cublasCreate(&handle_), "cublasCreate");
      require_cublas(cublasSetPointerMode(handle_,
                                          CUBLAS_POINTER_MODE_HOST),
                     "cublasSetPointerMode");
      require_cublas(cublasSetAtomicsMode(handle_,
                                          CUBLAS_ATOMICS_NOT_ALLOWED),
                     "cublasSetAtomicsMode");
      device_ = device;
    }
    return handle_;
  }

private:
  cublasHandle_t handle_{};
  int device_{-1};
};

thread_local PersistentCublasHandle persistent_cublas_handle;
thread_local TceCudaWorkspace persistent_tce_workspace;

__global__ void triples_energy_kernel(
    const double* doubles_moment, const double* singles_moment,
    const double* e0, const double* e1, const double* e2, const double* e3,
    const double* e4, const double* e5, const std::size_t d0,
    const std::size_t d1, const std::size_t d2, const std::size_t d3,
    const std::size_t d4, const std::size_t d5, const std::size_t size,
    const double factor, const double denominator_floor,
    double* partial_bracket, double* partial_parenthetical,
    int* invalid_denominator) {
  extern __shared__ double shared[];
  double* bracket = shared;
  double* parenthetical = shared + blockDim.x;
  const std::size_t offset =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  double bracket_value = 0.0;
  double parenthetical_value = 0.0;
  if (offset < size) {
    std::size_t rest = offset;
    const std::size_t i5 = rest % d5;
    rest /= d5;
    const std::size_t i4 = rest % d4;
    rest /= d4;
    const std::size_t i3 = rest % d3;
    rest /= d3;
    const std::size_t i2 = rest % d2;
    rest /= d2;
    const std::size_t i1 = rest % d1;
    rest /= d1;
    const std::size_t i0 = rest % d0;
    const double denominator =
        e0[i0] + e1[i1] + e2[i2] - e3[i3] - e4[i4] - e5[i5];
    if (!isfinite(denominator) || fabs(denominator) < denominator_floor) {
      atomicExch(invalid_denominator, 1);
    } else {
      const double doubles = doubles_moment[offset];
      const double singles = singles_moment[offset];
      bracket_value = factor * doubles * doubles / denominator;
      parenthetical_value =
          factor * doubles * (doubles + singles) / denominator;
    }
  }
  bracket[threadIdx.x] = bracket_value;
  parenthetical[threadIdx.x] = parenthetical_value;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      bracket[threadIdx.x] += bracket[threadIdx.x + stride];
      parenthetical[threadIdx.x] += parenthetical[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial_bracket[blockIdx.x] = bracket[0];
    partial_parenthetical[blockIdx.x] = parenthetical[0];
  }
}

struct PackedMomentPermutation {
  std::size_t t2_jk_axis_e[3]{};
  std::size_t eri_pair_e[6]{};
  std::size_t t2_m_pair[6]{};
  std::size_t eri_m_axis[3]{};
  std::size_t t1_axis[3]{};
  std::size_t eri_pair_jk[6]{};
  std::size_t fock_axis[3]{};
};

__device__ __constant__ unsigned char moment_permutations[18] = {
    0, 1, 2, 0, 2, 1, 1, 0, 2,
    1, 2, 0, 2, 0, 1, 2, 1, 0};
__device__ __constant__ double moment_permutation_sign[6] = {
    1.0, -1.0, -1.0, 1.0, 1.0, -1.0};

__global__ void triples_moment_kernel(
    const double* packed, const PackedMomentPermutation* metadata,
    const double* virtual_energy0, const double* virtual_energy1,
    const double* virtual_energy2, const std::size_t occupied,
    const std::size_t virtuals, const std::size_t offset0,
    const std::size_t offset1, const std::size_t offset2,
    const std::size_t dimension0, const std::size_t dimension1,
    const std::size_t dimension2, const std::size_t size,
    const double occupied_energy_sum, const double denominator_floor,
    double* doubles_output, double* left_output,
    double* partial_parenthetical, int* invalid_value) {
  extern __shared__ double shared[];
  const std::size_t linear =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  double contribution = 0.0;
  if (linear < size) {
    std::size_t rest = linear;
    const std::size_t x2 = rest % dimension2;
    rest /= dimension2;
    const std::size_t x1 = rest % dimension1;
    rest /= dimension1;
    const std::size_t x0 = rest;
    const std::size_t global[3] = {offset0 + x0, offset1 + x1,
                                   offset2 + x2};
    const std::size_t local[3] = {x0, x1, x2};
    const std::size_t dimensions[3] = {dimension0, dimension1,
                                       dimension2};
    if (global[0] < global[1] && global[1] < global[2]) {
      double doubles = 0.0;
      double singles = 0.0;
      for (int occupied_permutation = 0; occupied_permutation < 6;
           ++occupied_permutation) {
        const PackedMomentPermutation data = metadata[occupied_permutation];
        for (int virtual_permutation = 0; virtual_permutation < 6;
             ++virtual_permutation) {
          const int permutation_base = 3 * virtual_permutation;
          const std::size_t a_axis =
              moment_permutations[permutation_base];
          const std::size_t b_axis =
              moment_permutations[permutation_base + 1];
          const std::size_t c_axis =
              moment_permutations[permutation_base + 2];
          const std::size_t a = local[a_axis];
          const std::size_t b = local[b_axis];
          const std::size_t c = local[c_axis];
          const std::size_t pair = b * dimensions[c_axis] + c;
          double raw = 0.0;
          for (std::size_t e = 0; e < virtuals; ++e)
            raw += packed[data.t2_jk_axis_e[a_axis] + a * virtuals + e] *
                   packed[data.eri_pair_e[virtual_permutation] +
                          pair * virtuals + e];
          for (std::size_t m = 0; m < occupied; ++m)
            raw -= packed[data.t2_m_pair[virtual_permutation] +
                          m * dimensions[b_axis] * dimensions[c_axis] + pair] *
                   packed[data.eri_m_axis[a_axis] +
                          m * dimensions[a_axis] + a];
          const double sign =
              moment_permutation_sign[occupied_permutation] *
              moment_permutation_sign[virtual_permutation];
          doubles += 0.25 * sign * raw;
          singles +=
              0.25 * sign *
              (packed[data.t1_axis[a_axis] + a] *
                   packed[data.eri_pair_jk[virtual_permutation] + pair] +
               packed[data.fock_axis[a_axis] + a] *
                   packed[data.t2_jk_axis_e[b_axis] + b * virtuals +
                          global[c_axis]]);
        }
      }
      const double denominator = occupied_energy_sum - virtual_energy0[x0] -
                                 virtual_energy1[x1] - virtual_energy2[x2];
      if (!isfinite(denominator) || fabs(denominator) < denominator_floor) {
        atomicExch(invalid_value, 1);
      } else {
        doubles_output[linear] = doubles;
        left_output[linear] = doubles + singles;
        contribution = doubles * (doubles + singles) / denominator;
        if (!isfinite(contribution))
          atomicExch(invalid_value, 1);
      }
    }
  }
  shared[threadIdx.x] = contribution;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    partial_parenthetical[blockIdx.x] = shared[0];
}

__global__ void scatter_tce_moment_task_kernel(
    const double* packed, const double* raw,
    const PackedMomentPermutation metadata, const std::size_t task,
    const std::size_t a_axis, const std::size_t b_axis,
    const std::size_t c_axis, const std::size_t virtuals,
    const std::size_t offset0, const std::size_t offset1,
    const std::size_t offset2, const std::size_t dimension0,
    const std::size_t dimension1, const std::size_t dimension2,
    const std::size_t pair_size, const double factor,
    double* task_doubles, double* task_singles) {
  const std::size_t tile_size = dimension0 * dimension1 * dimension2;
  const std::size_t linear =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= tile_size)
    return;
  std::size_t rest = linear;
  const std::size_t local2 = rest % dimension2;
  rest /= dimension2;
  const std::size_t local1 = rest % dimension1;
  const std::size_t local0 = rest / dimension1;
  const std::size_t local[3] = {local0, local1, local2};
  const std::size_t global[3] = {offset0 + local0, offset1 + local1,
                                 offset2 + local2};
  if (!(global[0] < global[1] && global[1] < global[2]))
    return;
  const std::size_t dimensions[3] = {dimension0, dimension1, dimension2};
  const std::size_t a = local[a_axis];
  const std::size_t b = local[b_axis];
  const std::size_t c = local[c_axis];
  const std::size_t pair = b * dimensions[c_axis] + c;
  const std::size_t destination = task * tile_size + linear;
  task_doubles[destination] = factor * raw[a * pair_size + pair];
  task_singles[destination] =
      factor *
      (packed[metadata.t1_axis[a_axis] + a] *
           packed[metadata.eri_pair_jk[task % 6] + pair] +
       packed[metadata.fock_axis[a_axis] + a] *
           packed[metadata.t2_jk_axis_e[b_axis] +
                  b * virtuals + global[c_axis]]);
}

__global__ void reduce_tce_moment_tasks_kernel(
    const double* task_doubles, const double* task_singles,
    const double* virtual_energy0, const double* virtual_energy1,
    const double* virtual_energy2, const std::size_t offset0,
    const std::size_t offset1, const std::size_t offset2,
    const std::size_t dimension0, const std::size_t dimension1,
    const std::size_t dimension2, const double occupied_energy_sum,
    const double denominator_floor, double* doubles_output,
    double* left_output, double* partial_parenthetical,
    int* invalid_value) {
  extern __shared__ double shared[];
  constexpr std::size_t task_count = 36;
  const std::size_t tile_size = dimension0 * dimension1 * dimension2;
  const std::size_t linear =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  double contribution = 0.0;
  if (linear < tile_size) {
    double doubles = 0.0;
    double singles = 0.0;
    for (std::size_t task = 0; task < task_count; ++task) {
      doubles += task_doubles[task * tile_size + linear];
      singles += task_singles[task * tile_size + linear];
    }
    doubles_output[linear] = doubles;
    left_output[linear] = doubles + singles;

    std::size_t rest = linear;
    const std::size_t local2 = rest % dimension2;
    rest /= dimension2;
    const std::size_t local1 = rest % dimension1;
    const std::size_t local0 = rest / dimension1;
    if (offset0 + local0 < offset1 + local1 &&
        offset1 + local1 < offset2 + local2) {
      const double denominator =
          occupied_energy_sum - virtual_energy0[local0] -
          virtual_energy1[local1] - virtual_energy2[local2];
      if (!isfinite(denominator) ||
          fabs(denominator) < denominator_floor) {
        atomicExch(invalid_value, 1);
      } else {
        contribution = doubles * (doubles + singles) / denominator;
        if (!isfinite(contribution))
          atomicExch(invalid_value, 1);
      }
    }
  }
  shared[threadIdx.x] = contribution;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    partial_parenthetical[blockIdx.x] = shared[0];
}

__global__ void ccsd_wabef_kernel(
    const double* tau, const double* wabef, const std::size_t occupied_pairs,
    const std::size_t contraction_size, const std::size_t output_count,
    double* contractions) {
  extern __shared__ double shared[];
  const std::size_t output = static_cast<std::size_t>(blockIdx.x);
  double value = 0.0;
  if (output < output_count) {
    const std::size_t task = output / occupied_pairs;
    const std::size_t pair = output % occupied_pairs;
    for (std::size_t element = threadIdx.x; element < contraction_size;
         element += blockDim.x)
      value += tau[pair * contraction_size + element] *
               wabef[task * contraction_size + element];
  }
  shared[threadIdx.x] = value;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0 && output < output_count)
    contractions[output] = shared[0];
}

std::size_t append_values(std::vector<double>& destination,
                          const std::vector<double>& source) {
  const auto offset = destination.size();
  if (source.size() > destination.max_size() - destination.size())
    throw std::overflow_error("CUDA triples packed tile size overflows");
  destination.insert(destination.end(), source.begin(), source.end());
  return offset;
}

std::size_t evaluated_virtual_triples(const TriplesMomentTile& tile) {
  std::size_t count = 0;
  for (std::size_t x0 = 0; x0 < tile.virtual_dimensions[0]; ++x0)
    for (std::size_t x1 = 0; x1 < tile.virtual_dimensions[1]; ++x1)
      for (std::size_t x2 = 0; x2 < tile.virtual_dimensions[2]; ++x2)
        if (tile.virtual_offsets[0] + x0 < tile.virtual_offsets[1] + x1 &&
            tile.virtual_offsets[1] + x1 < tile.virtual_offsets[2] + x2)
          ++count;
  return count;
}

} // namespace

CudaTriplesRuntimeInfo cuda_triples_runtime_info() {
  CudaTriplesRuntimeInfo info;
  info.compiled = true;
  int count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&count);
  if (count_status == cudaErrorNoDevice || count_status == cudaErrorInsufficientDriver) {
    cudaGetLastError();
    return info;
  }
  require_cuda(count_status, "cudaGetDeviceCount");
  info.visible_devices = count;
  if (count == 0)
    return info;
  info.selected_device = select_device(-1, count);
  require_cuda(cudaSetDevice(info.selected_device), "cudaSetDevice");
  cudaDeviceProp properties{};
  require_cuda(cudaGetDeviceProperties(&properties, info.selected_device),
               "cudaGetDeviceProperties");
  info.device_name = properties.name;
  require_cuda(cudaMemGetInfo(&info.free_bytes, &info.total_bytes),
               "cudaMemGetInfo");
  info.available = true;
  return info;
}

TriplesEnergyPair reduce_triples_energy_cuda(const TriplesEnergyTile& tile,
                                             const int device) {
  validate_triples_energy_tile(tile);
  int count = 0;
  require_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  const int selected = select_device(device, count);
  require_cuda(cudaSetDevice(selected), "cudaSetDevice");
  const std::size_t size = triples_energy_tile_size(tile);
  const std::size_t blocks =
      (size + static_cast<std::size_t>(threads_per_block) - 1) /
      static_cast<std::size_t>(threads_per_block);
  if (blocks > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()))
    throw std::overflow_error("CUDA triples grid dimension overflows");

  std::size_t required_bytes = checked_add(
      checked_add(checked_bytes(size), checked_bytes(size)),
      checked_add(checked_bytes(blocks), checked_bytes(blocks)));
  required_bytes = checked_add(required_bytes, sizeof(int));
  for (const auto& energies : tile.orbital_energies)
    required_bytes =
        checked_add(required_bytes, checked_bytes(energies.size()));
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  require_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  if (required_bytes > free_bytes - free_bytes / 10)
    throw std::runtime_error(
        "CUDA triples tile exceeds 90 percent of currently free device memory");

  DeviceBuffer<double> doubles_device(size);
  DeviceBuffer<double> singles_device(size);
  std::array<DeviceBuffer<double>, 6> energy_device{
      DeviceBuffer<double>(tile.orbital_energies[0].size()),
      DeviceBuffer<double>(tile.orbital_energies[1].size()),
      DeviceBuffer<double>(tile.orbital_energies[2].size()),
      DeviceBuffer<double>(tile.orbital_energies[3].size()),
      DeviceBuffer<double>(tile.orbital_energies[4].size()),
      DeviceBuffer<double>(tile.orbital_energies[5].size())};
  DeviceBuffer<double> bracket_device(blocks);
  DeviceBuffer<double> parenthetical_device(blocks);
  DeviceBuffer<int> invalid_device(1);
  require_cuda(cudaMemcpy(doubles_device.get(), tile.doubles_moment.data(),
                          checked_bytes(size), cudaMemcpyHostToDevice),
               "cudaMemcpy doubles moment");
  require_cuda(cudaMemcpy(singles_device.get(), tile.singles_moment.data(),
                          checked_bytes(size), cudaMemcpyHostToDevice),
               "cudaMemcpy singles moment");
  for (std::size_t axis = 0; axis < energy_device.size(); ++axis)
    require_cuda(
        cudaMemcpy(energy_device[axis].get(),
                   tile.orbital_energies[axis].data(),
                   checked_bytes(tile.orbital_energies[axis].size()),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy orbital energies");
  require_cuda(cudaMemset(invalid_device.get(), 0, sizeof(int)),
               "cudaMemset denominator flag");

  const std::size_t shared_bytes =
      2 * static_cast<std::size_t>(threads_per_block) * sizeof(double);
  triples_energy_kernel<<<static_cast<unsigned int>(blocks),
                          threads_per_block, shared_bytes>>>(
      doubles_device.get(), singles_device.get(), energy_device[0].get(),
      energy_device[1].get(), energy_device[2].get(), energy_device[3].get(),
      energy_device[4].get(), energy_device[5].get(), tile.dimensions[0],
      tile.dimensions[1], tile.dimensions[2], tile.dimensions[3],
      tile.dimensions[4], tile.dimensions[5], size, tile.symmetry_factor,
      tile.denominator_floor, bracket_device.get(),
      parenthetical_device.get(), invalid_device.get());
  require_cuda(cudaGetLastError(), "triples_energy_kernel launch");
  require_cuda(cudaDeviceSynchronize(), "triples_energy_kernel execution");

  int invalid = 0;
  require_cuda(cudaMemcpy(&invalid, invalid_device.get(), sizeof(int),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy denominator flag");
  if (invalid != 0)
    throw std::runtime_error("small or invalid triples tile denominator");
  std::vector<double> bracket_partial(blocks);
  std::vector<double> parenthetical_partial(blocks);
  require_cuda(cudaMemcpy(bracket_partial.data(), bracket_device.get(),
                          checked_bytes(blocks), cudaMemcpyDeviceToHost),
               "cudaMemcpy CCSD[T] partials");
  require_cuda(cudaMemcpy(parenthetical_partial.data(),
                          parenthetical_device.get(), checked_bytes(blocks),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy CCSD(T) partials");

  TriplesEnergyPair result;
  for (std::size_t block = 0; block < blocks; ++block) {
    result.bracket += bracket_partial[block];
    result.parenthetical += parenthetical_partial[block];
  }
  if (!std::isfinite(result.bracket) ||
      !std::isfinite(result.parenthetical))
    throw std::runtime_error("CUDA triples tile reduction is non-finite");
  result.workspace_bytes = required_bytes;
  result.device = selected;
  result.used_cuda = true;
  return result;
}

TriplesMomentResult contract_triples_moment_cuda(const TriplesMomentTile& tile,
                                                  const int device) {
  validate_triples_moment_tile(tile);
  int count = 0;
  require_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  const int selected = select_device(device, count);
  require_cuda(cudaSetDevice(selected), "cudaSetDevice");

  std::vector<double> packed;
  std::array<PackedMomentPermutation, 6> metadata{};
  for (std::size_t occupied_permutation = 0;
       occupied_permutation < metadata.size(); ++occupied_permutation) {
    const auto& source = tile.occupied_permutations[occupied_permutation];
    auto& target = metadata[occupied_permutation];
    for (std::size_t axis = 0; axis < 3; ++axis) {
      target.t2_jk_axis_e[axis] =
          append_values(packed, source.t2_jk_axis_e[axis]);
      target.eri_m_axis[axis] =
          append_values(packed, source.eri_m_axis[axis]);
      target.t1_axis[axis] = append_values(packed, source.t1_axis[axis]);
      target.fock_axis[axis] = append_values(packed, source.fock_axis[axis]);
    }
    for (std::size_t virtual_permutation = 0; virtual_permutation < 6;
         ++virtual_permutation) {
      target.eri_pair_e[virtual_permutation] =
          append_values(packed, source.eri_pair_e[virtual_permutation]);
      target.t2_m_pair[virtual_permutation] =
          append_values(packed, source.t2_m_pair[virtual_permutation]);
      target.eri_pair_jk[virtual_permutation] =
          append_values(packed, source.eri_pair_jk[virtual_permutation]);
    }
  }

  std::size_t tile_size = 1;
  for (const auto dimension : tile.virtual_dimensions) {
    if (tile_size > std::numeric_limits<std::size_t>::max() / dimension)
      throw std::overflow_error("CUDA triples moment grid size overflows");
    tile_size *= dimension;
  }
  const std::size_t blocks =
      (tile_size + static_cast<std::size_t>(threads_per_block) - 1) /
      static_cast<std::size_t>(threads_per_block);
  if (blocks > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()))
    throw std::overflow_error("CUDA triples moment grid dimension overflows");

  std::size_t required_bytes = checked_bytes(packed.size());
  required_bytes = checked_add(
      required_bytes, metadata.size() * sizeof(PackedMomentPermutation));
  for (const auto& energies : tile.virtual_energies)
    required_bytes = checked_add(required_bytes, checked_bytes(energies.size()));
  required_bytes = checked_add(required_bytes, checked_bytes(blocks));
  required_bytes = checked_add(required_bytes, checked_bytes(tile_size));
  required_bytes = checked_add(required_bytes, checked_bytes(tile_size));
  required_bytes = checked_add(required_bytes, sizeof(int));
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  require_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  if (required_bytes > free_bytes - free_bytes / 10)
    throw std::runtime_error(
        "CUDA triples moment tile exceeds 90 percent of free device memory");

  DeviceBuffer<double> packed_device(packed.size());
  DeviceBuffer<PackedMomentPermutation> metadata_device(metadata.size());
  std::array<DeviceBuffer<double>, 3> energy_device{
      DeviceBuffer<double>(tile.virtual_energies[0].size()),
      DeviceBuffer<double>(tile.virtual_energies[1].size()),
      DeviceBuffer<double>(tile.virtual_energies[2].size())};
  DeviceBuffer<double> partial_device(blocks);
  DeviceBuffer<double> doubles_device(tile_size);
  DeviceBuffer<double> left_device(tile_size);
  DeviceBuffer<int> invalid_device(1);
  require_cuda(cudaMemcpy(packed_device.get(), packed.data(),
                          checked_bytes(packed.size()), cudaMemcpyHostToDevice),
               "cudaMemcpy packed triples moment tile");
  require_cuda(cudaMemcpy(metadata_device.get(), metadata.data(),
                          metadata.size() * sizeof(PackedMomentPermutation),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy triples moment metadata");
  for (std::size_t axis = 0; axis < 3; ++axis)
    require_cuda(cudaMemcpy(energy_device[axis].get(),
                            tile.virtual_energies[axis].data(),
                            checked_bytes(tile.virtual_energies[axis].size()),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy triples moment virtual energies");
  require_cuda(cudaMemset(invalid_device.get(), 0, sizeof(int)),
               "cudaMemset triples moment invalid flag");
  require_cuda(cudaMemset(doubles_device.get(), 0, checked_bytes(tile_size)),
               "cudaMemset triples doubles moment output");
  require_cuda(cudaMemset(left_device.get(), 0, checked_bytes(tile_size)),
               "cudaMemset triples left moment output");

  const double occupied_energy_sum = tile.occupied_energies[0] +
                                     tile.occupied_energies[1] +
                                     tile.occupied_energies[2];
  triples_moment_kernel<<<static_cast<unsigned int>(blocks), threads_per_block,
                          static_cast<std::size_t>(threads_per_block) *
                              sizeof(double)>>>(
      packed_device.get(), metadata_device.get(), energy_device[0].get(),
      energy_device[1].get(), energy_device[2].get(), tile.occupied_count,
      tile.virtual_count, tile.virtual_offsets[0], tile.virtual_offsets[1],
      tile.virtual_offsets[2], tile.virtual_dimensions[0],
      tile.virtual_dimensions[1], tile.virtual_dimensions[2], tile_size,
      occupied_energy_sum, tile.denominator_floor, doubles_device.get(),
      left_device.get(), partial_device.get(), invalid_device.get());
  require_cuda(cudaGetLastError(), "triples_moment_kernel launch");
  require_cuda(cudaDeviceSynchronize(), "triples_moment_kernel execution");

  int invalid = 0;
  require_cuda(cudaMemcpy(&invalid, invalid_device.get(), sizeof(int),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy triples moment invalid flag");
  if (invalid != 0)
    throw std::runtime_error(
        "small denominator or non-finite CUDA triples moment contribution");
  std::vector<double> partial(blocks);
  require_cuda(cudaMemcpy(partial.data(), partial_device.get(),
                          checked_bytes(blocks), cudaMemcpyDeviceToHost),
               "cudaMemcpy CCSD(T) moment partials");
  TriplesMomentResult result;
  result.doubles_moment.resize(tile_size);
  result.left_moment.resize(tile_size);
  require_cuda(cudaMemcpy(result.doubles_moment.data(), doubles_device.get(),
                          checked_bytes(tile_size), cudaMemcpyDeviceToHost),
               "cudaMemcpy triples doubles moments");
  require_cuda(cudaMemcpy(result.left_moment.data(), left_device.get(),
                          checked_bytes(tile_size), cudaMemcpyDeviceToHost),
               "cudaMemcpy triples left moments");
  for (const double value : partial)
    result.parenthetical += value;
  if (!std::isfinite(result.parenthetical))
    throw std::runtime_error("CUDA triples moment reduction is non-finite");
  result.workspace_bytes = required_bytes;
  result.evaluated_virtual_triples = evaluated_virtual_triples(tile);
  result.device = selected;
  result.used_cuda = true;
  return result;
}

TriplesMomentResult contract_triples_moment_tce_cuda(
    const TriplesMomentTile& tile, const int device) {
  validate_triples_moment_tile(tile);
  int count = 0;
  require_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  const int selected = select_device(device, count);
  require_cuda(cudaSetDevice(selected), "cudaSetDevice");

  std::vector<double> packed;
  std::array<PackedMomentPermutation, 6> metadata{};
  for (std::size_t occupied_permutation = 0;
       occupied_permutation < metadata.size(); ++occupied_permutation) {
    const auto& source = tile.occupied_permutations[occupied_permutation];
    auto& target = metadata[occupied_permutation];
    for (std::size_t axis = 0; axis < 3; ++axis) {
      target.t2_jk_axis_e[axis] =
          append_values(packed, source.t2_jk_axis_e[axis]);
      target.eri_m_axis[axis] =
          append_values(packed, source.eri_m_axis[axis]);
      target.t1_axis[axis] = append_values(packed, source.t1_axis[axis]);
      target.fock_axis[axis] = append_values(packed, source.fock_axis[axis]);
    }
    for (std::size_t virtual_permutation = 0;
         virtual_permutation < permutations.size(); ++virtual_permutation) {
      target.eri_pair_e[virtual_permutation] =
          append_values(packed, source.eri_pair_e[virtual_permutation]);
      target.t2_m_pair[virtual_permutation] =
          append_values(packed, source.t2_m_pair[virtual_permutation]);
      target.eri_pair_jk[virtual_permutation] =
          append_values(packed, source.eri_pair_jk[virtual_permutation]);
    }
  }

  const std::size_t tile_size = checked_product(
      checked_product(tile.virtual_dimensions[0],
                      tile.virtual_dimensions[1]),
      tile.virtual_dimensions[2]);
  constexpr std::size_t task_count = 36;
  const std::size_t task_elements = checked_product(task_count, tile_size);
  const std::size_t blocks =
      (tile_size + static_cast<std::size_t>(threads_per_block) - 1) /
      static_cast<std::size_t>(threads_per_block);
  if (blocks >
      static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()))
    throw std::overflow_error("TCE CUDA triples grid dimension overflows");

  std::size_t required_bytes = checked_bytes(packed.size());
  required_bytes = checked_add(required_bytes, checked_bytes(tile_size));
  required_bytes = checked_add(required_bytes, checked_bytes(task_elements));
  required_bytes = checked_add(required_bytes, checked_bytes(task_elements));
  required_bytes = checked_add(required_bytes, checked_bytes(tile_size));
  required_bytes = checked_add(required_bytes, checked_bytes(tile_size));
  required_bytes = checked_add(required_bytes, checked_bytes(blocks));
  required_bytes = checked_add(required_bytes, sizeof(int));
  for (const auto& energies : tile.virtual_energies)
    required_bytes = checked_add(required_bytes,
                                 checked_bytes(energies.size()));
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  require_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  if (required_bytes > free_bytes - free_bytes / 10)
    throw std::runtime_error(
        "TCE CUDA triples tile exceeds 90 percent of free device memory");

  auto& workspace = persistent_tce_workspace;
  workspace.select_device(selected);
  workspace.packed.reserve(packed.size());
  workspace.raw.reserve(tile_size);
  workspace.task_doubles.reserve(task_elements);
  workspace.task_singles.reserve(task_elements);
  workspace.doubles.reserve(tile_size);
  workspace.left.reserve(tile_size);
  workspace.partial.reserve(blocks);
  workspace.invalid.reserve(1);
  for (std::size_t axis = 0; axis < 3; ++axis)
    workspace.energies[axis].reserve(tile.virtual_energies[axis].size());

  require_cuda(cudaMemcpy(workspace.packed.get(), packed.data(),
                          checked_bytes(packed.size()),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy TCE triples packed tile");
  require_cuda(cudaMemset(workspace.task_doubles.get(), 0,
                          checked_bytes(task_elements)),
               "cudaMemset TCE doubles tasks");
  require_cuda(cudaMemset(workspace.task_singles.get(), 0,
                          checked_bytes(task_elements)),
               "cudaMemset TCE singles tasks");
  require_cuda(cudaMemset(workspace.invalid.get(), 0, sizeof(int)),
               "cudaMemset TCE triples invalid flag");
  for (std::size_t axis = 0; axis < 3; ++axis)
    require_cuda(cudaMemcpy(workspace.energies[axis].get(),
                            tile.virtual_energies[axis].data(),
                            checked_bytes(tile.virtual_energies[axis].size()),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy TCE triples virtual energies");

  const auto handle = persistent_cublas_handle.get(selected);
  constexpr double one = 1.0;
  constexpr double minus_one = -1.0;
  constexpr double zero = 0.0;
  for (std::size_t task = 0; task < task_count; ++task) {
    const auto occupied_permutation = task / permutations.size();
    const auto virtual_permutation = task % permutations.size();
    const auto& data = metadata[occupied_permutation];
    const auto& permutation = permutations[virtual_permutation];
    const auto a_axis = static_cast<std::size_t>(permutation[0]);
    const auto b_axis = static_cast<std::size_t>(permutation[1]);
    const auto c_axis = static_cast<std::size_t>(permutation[2]);
    const auto external = tile.virtual_dimensions[a_axis];
    const auto pair_size = checked_product(tile.virtual_dimensions[b_axis],
                                           tile.virtual_dimensions[c_axis]);
    const int pairs_blas = checked_cublas_dimension(pair_size);
    const int external_blas = checked_cublas_dimension(external);
    const int virtuals_blas =
        checked_cublas_dimension(tile.virtual_count);
    const int occupied_blas =
        checked_cublas_dimension(tile.occupied_count);

    require_cublas(
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, pairs_blas,
                    external_blas, virtuals_blas, &one,
                    workspace.packed.get() +
                        data.eri_pair_e[virtual_permutation],
                    virtuals_blas,
                    workspace.packed.get() + data.t2_jk_axis_e[a_axis],
                    virtuals_blas, &zero, workspace.raw.get(), pairs_blas),
        "cublasDgemm TCE triples direct moment");
    require_cublas(
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, pairs_blas,
                    external_blas, occupied_blas, &minus_one,
                    workspace.packed.get() +
                        data.t2_m_pair[virtual_permutation],
                    pairs_blas,
                    workspace.packed.get() + data.eri_m_axis[a_axis],
                    external_blas, &one, workspace.raw.get(), pairs_blas),
        "cublasDgemm TCE triples exchange moment");

    const double factor =
        0.25 * permutation_sign[occupied_permutation] *
        permutation_sign[virtual_permutation];
    scatter_tce_moment_task_kernel<<<
        static_cast<unsigned int>(blocks), threads_per_block>>>(
        workspace.packed.get(), workspace.raw.get(), data, task, a_axis,
        b_axis, c_axis, tile.virtual_count, tile.virtual_offsets[0],
        tile.virtual_offsets[1], tile.virtual_offsets[2],
        tile.virtual_dimensions[0], tile.virtual_dimensions[1],
        tile.virtual_dimensions[2], pair_size, factor,
        workspace.task_doubles.get(), workspace.task_singles.get());
    require_cuda(cudaGetLastError(),
                 "scatter_tce_moment_task_kernel launch");
  }

  const double occupied_energy_sum = tile.occupied_energies[0] +
                                     tile.occupied_energies[1] +
                                     tile.occupied_energies[2];
  reduce_tce_moment_tasks_kernel<<<
      static_cast<unsigned int>(blocks), threads_per_block,
      static_cast<std::size_t>(threads_per_block) * sizeof(double)>>>(
      workspace.task_doubles.get(), workspace.task_singles.get(),
      workspace.energies[0].get(), workspace.energies[1].get(),
      workspace.energies[2].get(), tile.virtual_offsets[0],
      tile.virtual_offsets[1], tile.virtual_offsets[2],
      tile.virtual_dimensions[0], tile.virtual_dimensions[1],
      tile.virtual_dimensions[2], occupied_energy_sum,
      tile.denominator_floor, workspace.doubles.get(), workspace.left.get(),
      workspace.partial.get(), workspace.invalid.get());
  require_cuda(cudaGetLastError(),
               "reduce_tce_moment_tasks_kernel launch");
  require_cuda(cudaDeviceSynchronize(),
               "TCE CUDA triples moment execution");

  int invalid = 0;
  require_cuda(cudaMemcpy(&invalid, workspace.invalid.get(), sizeof(int),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy TCE triples invalid flag");
  if (invalid != 0)
    throw std::runtime_error(
        "small denominator or non-finite TCE CUDA triples contribution");

  TriplesMomentResult result;
  result.doubles_moment.resize(tile_size);
  result.left_moment.resize(tile_size);
  std::vector<double> partial(blocks);
  require_cuda(cudaMemcpy(result.doubles_moment.data(),
                          workspace.doubles.get(),
                          checked_bytes(tile_size), cudaMemcpyDeviceToHost),
               "cudaMemcpy TCE triples doubles moments");
  require_cuda(cudaMemcpy(result.left_moment.data(), workspace.left.get(),
                          checked_bytes(tile_size), cudaMemcpyDeviceToHost),
               "cudaMemcpy TCE triples left moments");
  require_cuda(cudaMemcpy(partial.data(), workspace.partial.get(),
                          checked_bytes(blocks), cudaMemcpyDeviceToHost),
               "cudaMemcpy TCE CCSD(T) partials");
  for (const double value : partial)
    result.parenthetical += value;
  if (!std::isfinite(result.parenthetical))
    throw std::runtime_error("TCE CUDA triples reduction is non-finite");
  result.workspace_bytes = required_bytes;
  result.evaluated_virtual_triples = evaluated_virtual_triples(tile);
  result.device = selected;
  result.used_cuda = true;
  return result;
}

CcsdWabefResult contract_ccsd_wabef_cuda(const CcsdWabefTile& tile,
                                          const int device) {
  validate_ccsd_wabef_tile(tile);
  int count = 0;
  require_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  const int selected = select_device(device, count);
  require_cuda(cudaSetDevice(selected), "cudaSetDevice");
  const auto output_count =
      checked_product(tile.virtual_pair_tasks, tile.occupied_pairs);
  if (output_count >
      static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()))
    throw std::overflow_error("CUDA CCSD W_abef grid dimension overflows");
  std::size_t required_bytes = checked_bytes(tile.tau.size());
  required_bytes = checked_add(required_bytes, checked_bytes(tile.wabef.size()));
  required_bytes = checked_add(required_bytes, checked_bytes(output_count));
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  require_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  if (required_bytes > free_bytes - free_bytes / 10)
    throw std::runtime_error(
        "CUDA CCSD W_abef tile exceeds 90 percent of free device memory");

  DeviceBuffer<double> tau_device(tile.tau.size());
  DeviceBuffer<double> wabef_device(tile.wabef.size());
  DeviceBuffer<double> result_device(output_count);
  require_cuda(cudaMemcpy(tau_device.get(), tile.tau.data(),
                          checked_bytes(tile.tau.size()),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy CCSD tau tile");
  require_cuda(cudaMemcpy(wabef_device.get(), tile.wabef.data(),
                          checked_bytes(tile.wabef.size()),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy CCSD W_abef tile");
  ccsd_wabef_kernel<<<static_cast<unsigned int>(output_count),
                      threads_per_block,
                      static_cast<std::size_t>(threads_per_block) *
                          sizeof(double)>>>(
      tau_device.get(), wabef_device.get(), tile.occupied_pairs,
      tile.virtual_pair_elements, output_count, result_device.get());
  require_cuda(cudaGetLastError(), "ccsd_wabef_kernel launch");
  require_cuda(cudaDeviceSynchronize(), "ccsd_wabef_kernel execution");

  CcsdWabefResult result;
  result.contractions.resize(output_count);
  require_cuda(cudaMemcpy(result.contractions.data(), result_device.get(),
                          checked_bytes(output_count), cudaMemcpyDeviceToHost),
               "cudaMemcpy CCSD W_abef contractions");
  for (const auto value : result.contractions)
    if (!std::isfinite(value))
      throw std::runtime_error(
          "CUDA CCSD W_abef contraction produced a non-finite value");
  result.workspace_bytes = required_bytes;
  result.device = selected;
  result.used_cuda = true;
  return result;
}

} // namespace localized_cc
