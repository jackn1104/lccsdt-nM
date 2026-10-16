#pragma once

#include "localized_cc/distributed_pair_tensor.hpp"
#include "localized_cc/tensor.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace localized_cc {

// Global Arrays releases use a fixed-size internal array-name buffer whose
// usable length is not exposed consistently by the public API. Keep selected
// block identifiers within this deliberately conservative portable limit so
// NGA_Create fails through a C++ exception before GA can abort MPI.
inline constexpr std::size_t global_arrays_block_name_maximum = 15;

void validate_global_arrays_block_name(std::string_view name);

// Rectangular Global Arrays storage for a deterministic list of variable-size
// monomer blocks. Rows are blocks and the unused tail of a short row remains
// exactly zero. The scientific element count excludes padding; the allocated
// count reports it explicitly so an unfavorable block partition is visible.
class SelectedBlockLayout {
public:
  explicit SelectedBlockLayout(std::vector<std::size_t> block_elements);

  [[nodiscard]] std::size_t block_count() const noexcept;
  [[nodiscard]] std::size_t block_size(std::size_t block) const;
  [[nodiscard]] std::size_t maximum_block_size() const noexcept;
  [[nodiscard]] std::size_t scientific_element_count() const noexcept;
  [[nodiscard]] std::size_t allocated_element_count() const noexcept;
  [[nodiscard]] bool block_owned(std::size_t block, std::size_t rank,
                                 std::size_t ranks) const;

private:
  std::vector<std::size_t> block_elements_;
  std::size_t maximum_block_size_{};
  std::size_t scientific_element_count_{};
  std::size_t allocated_element_count_{};
};

struct DistributedBlockMemory {
  std::size_t scientific_global_bytes{};
  std::size_t allocated_global_bytes{};
  std::size_t local_allocated_bytes{};
  std::size_t replicated_scientific_bytes{};
  double padding_factor{};
};

// Collective construction/destruction. All scientific values have one global
// owner in GA. Whole-block get/put calls are the only public data operations;
// scalar GA traffic is intentionally impossible at this boundary.
class GlobalArraysBlockTensor {
public:
  GlobalArraysBlockTensor(const SelectedBlockLayout& layout,
                          std::string name);
  ~GlobalArraysBlockTensor();

  GlobalArraysBlockTensor(const GlobalArraysBlockTensor&) = delete;
  GlobalArraysBlockTensor& operator=(const GlobalArraysBlockTensor&) = delete;
  GlobalArraysBlockTensor(GlobalArraysBlockTensor&&) = delete;
  GlobalArraysBlockTensor& operator=(GlobalArraysBlockTensor&&) = delete;

  [[nodiscard]] const SelectedBlockLayout& layout() const noexcept;
  [[nodiscard]] DistributedBlockMemory memory() const;
  [[nodiscard]] bool computation_block_owned(std::size_t block) const;

  void zero();
  void synchronize() const;
  void put_block(std::size_t block, const std::vector<double>& values);
  void accumulate_block(std::size_t block,
                        const std::vector<double>& values);
  [[nodiscard]] std::vector<double> get_block(std::size_t block) const;
  [[nodiscard]] double dot(const GlobalArraysBlockTensor& other) const;
  void copy_from(const GlobalArraysBlockTensor& other);
  void load_replicated(const std::vector<std::vector<double>>& blocks);
  [[nodiscard]] std::vector<std::vector<double>> replicate() const;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

struct DistributedBlockDiisMemory {
  std::size_t history_entries{};
  std::size_t doubles_history_scientific_global_bytes{};
  std::size_t doubles_history_allocated_global_bytes{};
  std::size_t doubles_history_local_allocated_bytes{};
  std::size_t doubles_history_replicated_scientific_bytes_per_rank{};
  std::size_t singles_history_bytes_per_rank{};
};

class GlobalArraysBlockDiis {
public:
  GlobalArraysBlockDiis(std::size_t capacity,
                        const SelectedBlockLayout& layout,
                        std::string name_prefix);
  ~GlobalArraysBlockDiis();

  GlobalArraysBlockDiis(const GlobalArraysBlockDiis&) = delete;
  GlobalArraysBlockDiis& operator=(const GlobalArraysBlockDiis&) = delete;
  GlobalArraysBlockDiis(GlobalArraysBlockDiis&&) = delete;
  GlobalArraysBlockDiis& operator=(GlobalArraysBlockDiis&&) = delete;

  [[nodiscard]] Matrix extrapolate_distributed(
      const Matrix& singles_vector,
      const GlobalArraysBlockTensor& doubles_vector,
      const Matrix& singles_error,
      const GlobalArraysBlockTensor& doubles_error,
      GlobalArraysBlockTensor& extrapolated_doubles);
  [[nodiscard]] DistributedBlockDiisMemory memory() const;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

} // namespace localized_cc
