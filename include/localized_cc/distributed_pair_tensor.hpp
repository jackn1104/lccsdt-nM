#pragma once

#include "localized_cc/tensor.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace localized_cc {

// NWChem's internal-transformation CCSD path uses the composite
// occupied-virtual indices (ia,jb), with v-by-v chunks, for g_nt2/g_nht2.
// This class owns the backend-independent index and memory arithmetic for that
// layout.
class OccupiedPairLayout {
public:
  OccupiedPairLayout(std::size_t occupied, std::size_t virtuals);

  [[nodiscard]] std::size_t occupied() const noexcept { return occupied_; }
  [[nodiscard]] std::size_t virtuals() const noexcept { return virtuals_; }
  [[nodiscard]] std::size_t composite_extent() const noexcept;
  [[nodiscard]] std::size_t pair_count() const noexcept;
  [[nodiscard]] std::size_t element_count() const noexcept;
  [[nodiscard]] std::size_t global_bytes() const noexcept;
  [[nodiscard]] std::size_t composite_index(std::size_t occupied_index,
                                            std::size_t virtual_index) const;
  [[nodiscard]] std::array<std::size_t, 2>
  pair_from_flat(std::size_t pair) const;
  [[nodiscard]] bool pair_owned(std::size_t pair, std::size_t rank,
                                std::size_t ranks) const;
  [[nodiscard]] std::size_t owned_pair_count(std::size_t rank,
                                             std::size_t ranks) const;

private:
  std::size_t occupied_{};
  std::size_t virtuals_{};
};

struct DistributedTensorMemory {
  std::size_t global_bytes{};
  std::size_t local_bytes{};
  std::size_t replicated_bytes{};
  std::array<int, 2> local_low{{0, 0}};
  std::array<int, 2> local_high{{-1, -1}};
};

[[nodiscard]] bool global_arrays_backend_available() noexcept;
[[nodiscard]] bool global_arrays_runtime_active() noexcept;

// Must be constructed after parallel::Runtime and destroyed before it.
// It is a no-op when Global Arrays was not enabled at configure time.
class GlobalArraysRuntime {
public:
  GlobalArraysRuntime();
  ~GlobalArraysRuntime();

  GlobalArraysRuntime(const GlobalArraysRuntime&) = delete;
  GlobalArraysRuntime& operator=(const GlobalArraysRuntime&) = delete;

  [[nodiscard]] bool active() const noexcept;

private:
  [[maybe_unused]] bool owns_runtime_{};
};

// Collective construction/destruction; get/put operations are one-sided.
// The global matrix is T(ia,jb), chunked v-by-v so a block corresponds to one
// occupied pair (i,j).  Replication is an explicit checkpoint/reference
// boundary and is not intended for production residual contractions.
class GlobalArraysPairTensor {
public:
  GlobalArraysPairTensor(const OccupiedPairLayout& layout, std::string name);
  ~GlobalArraysPairTensor();

  GlobalArraysPairTensor(const GlobalArraysPairTensor&) = delete;
  GlobalArraysPairTensor& operator=(const GlobalArraysPairTensor&) = delete;
  GlobalArraysPairTensor(GlobalArraysPairTensor&&) = delete;
  GlobalArraysPairTensor& operator=(GlobalArraysPairTensor&&) = delete;

  [[nodiscard]] const OccupiedPairLayout& layout() const noexcept;
  [[nodiscard]] DistributedTensorMemory memory() const;
  [[nodiscard]] bool computation_pair_owned(std::size_t i,
                                             std::size_t j) const;

  void zero();
  void synchronize() const;
  void put_pair(std::size_t i, std::size_t j,
                const std::vector<double>& values);
  void accumulate_pair(std::size_t i, std::size_t j,
                       const std::vector<double>& values);
  [[nodiscard]] std::vector<double> get_pair(std::size_t i,
                                             std::size_t j) const;
  [[nodiscard]] double dot(const GlobalArraysPairTensor& other) const;
  void copy_from(const GlobalArraysPairTensor& other);
  void load_replicated(const Tensor4& values);
  [[nodiscard]] Tensor4 replicate() const;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

struct DistributedPairDiisMemory {
  std::size_t history_entries{};
  std::size_t doubles_history_global_bytes{};
  std::size_t doubles_history_local_bytes{};
  std::size_t doubles_history_replicated_bytes_per_rank{};
  std::size_t singles_history_bytes_per_rank{};
};

// Phase 14 DIIS store. T1 history remains replicated because it is O(ov); T2
// vector and error histories are stored as GA (ia,jb) arrays. The distributed
// overload writes its extrapolated T2 directly to another GA pair tensor.
class GlobalArraysPairDiis {
public:
  GlobalArraysPairDiis(std::size_t capacity,
                       const OccupiedPairLayout& layout,
                       std::string name_prefix);
  ~GlobalArraysPairDiis();

  GlobalArraysPairDiis(const GlobalArraysPairDiis&) = delete;
  GlobalArraysPairDiis& operator=(const GlobalArraysPairDiis&) = delete;
  GlobalArraysPairDiis(GlobalArraysPairDiis&&) = delete;
  GlobalArraysPairDiis& operator=(GlobalArraysPairDiis&&) = delete;

  [[nodiscard]] std::pair<Matrix, Tensor4>
  extrapolate(const Matrix& singles_vector, const Tensor4& doubles_vector,
              const Matrix& singles_error, const Tensor4& doubles_error);
  [[nodiscard]] Matrix extrapolate_distributed(
      const Matrix& singles_vector,
      const GlobalArraysPairTensor& doubles_vector,
      const Matrix& singles_error,
      const GlobalArraysPairTensor& doubles_error,
      GlobalArraysPairTensor& extrapolated_doubles);
  [[nodiscard]] DistributedPairDiisMemory memory() const;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

} // namespace localized_cc
