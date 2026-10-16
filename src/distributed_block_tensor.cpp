#include "localized_cc/distributed_block_tensor.hpp"

#include "localized_cc/diis_solver.hpp"
#include "localized_cc/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>
#include <utility>

#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
#include <ga.h>
#include <gacommon.h>
#endif

namespace localized_cc {
namespace {

std::size_t checked_product(const std::size_t left,
                            const std::size_t right,
                            const char* message) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(message);
  return left * right;
}

std::size_t checked_sum(const std::size_t left, const std::size_t right,
                        const char* message) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error(message);
  return left + right;
}

#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
int checked_ga_index(const std::size_t value, const char* message) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error(message);
  return static_cast<int>(value);
}
#endif

bool same_layout(const SelectedBlockLayout& left,
                 const SelectedBlockLayout& right) {
  if (left.block_count() != right.block_count() ||
      left.maximum_block_size() != right.maximum_block_size() ||
      left.scientific_element_count() != right.scientific_element_count())
    return false;
  for (std::size_t block = 0; block < left.block_count(); ++block)
    if (left.block_size(block) != right.block_size(block))
      return false;
  return true;
}

} // namespace

void validate_global_arrays_block_name(const std::string_view name) {
  if (name.empty())
    throw std::invalid_argument("Global Arrays selected tensor name is empty");
  if (name.size() > global_arrays_block_name_maximum)
    throw std::invalid_argument(
        "Global Arrays selected tensor name exceeds the portable " +
        std::to_string(global_arrays_block_name_maximum) +
        "-character limit");
}

SelectedBlockLayout::SelectedBlockLayout(
    std::vector<std::size_t> block_elements)
    : block_elements_(std::move(block_elements)) {
  if (block_elements_.empty())
    throw std::invalid_argument("selected block layout is empty");
  for (const auto elements : block_elements_) {
    if (elements == 0)
      throw std::invalid_argument(
          "selected block layout contains an empty block");
    maximum_block_size_ = std::max(maximum_block_size_, elements);
    scientific_element_count_ = checked_sum(
        scientific_element_count_, elements,
        "selected block scientific element count overflows");
  }
  allocated_element_count_ = checked_product(
      block_count(), maximum_block_size_,
      "selected block allocated element count overflows");
}

std::size_t SelectedBlockLayout::block_count() const noexcept {
  return block_elements_.size();
}

std::size_t SelectedBlockLayout::block_size(const std::size_t block) const {
  return block_elements_.at(block);
}

std::size_t SelectedBlockLayout::maximum_block_size() const noexcept {
  return maximum_block_size_;
}

std::size_t SelectedBlockLayout::scientific_element_count() const noexcept {
  return scientific_element_count_;
}

std::size_t SelectedBlockLayout::allocated_element_count() const noexcept {
  return allocated_element_count_;
}

bool SelectedBlockLayout::block_owned(const std::size_t block,
                                      const std::size_t rank,
                                      const std::size_t ranks) const {
  if (ranks == 0 || rank >= ranks)
    throw std::invalid_argument("invalid selected-block process grid");
  if (block >= block_count())
    throw std::out_of_range("selected block is out of range");
  return block % ranks == rank;
}

struct GlobalArraysBlockTensor::Implementation {
  explicit Implementation(SelectedBlockLayout input_layout)
      : layout(std::move(input_layout)) {}

  SelectedBlockLayout layout;
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  int handle{};
#endif
};

GlobalArraysBlockTensor::GlobalArraysBlockTensor(
    const SelectedBlockLayout& layout, std::string name)
    : implementation_(std::make_unique<Implementation>(layout)) {
  validate_global_arrays_block_name(name);
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  if (GA_Initialized() == 0)
    throw std::runtime_error(
        "Global Arrays runtime must precede selected block storage");
  int dimensions[2]{
      checked_ga_index(layout.block_count(),
                       "Global Arrays selected block count exceeds int"),
      checked_ga_index(layout.maximum_block_size(),
                       "Global Arrays selected block size exceeds int")};
  int chunk[2]{1, dimensions[1]};
  std::vector<char> mutable_name(name.begin(), name.end());
  mutable_name.push_back('\0');
  implementation_->handle =
      NGA_Create(C_DBL, 2, dimensions, mutable_name.data(), chunk);
  if (implementation_->handle == 0)
    throw std::runtime_error(
        "Global Arrays selected block allocation failed");
  GA_Zero(implementation_->handle);
  GA_Sync();
#else
  (void)name;
  throw std::runtime_error(
      "Global Arrays support was not enabled when this target was configured");
#endif
}

GlobalArraysBlockTensor::~GlobalArraysBlockTensor() {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  if (implementation_ && implementation_->handle != 0 &&
      GA_Initialized() != 0) {
    GA_Sync();
    GA_Destroy(implementation_->handle);
  }
#endif
}

const SelectedBlockLayout&
GlobalArraysBlockTensor::layout() const noexcept {
  return implementation_->layout;
}

DistributedBlockMemory GlobalArraysBlockTensor::memory() const {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  const auto communicator = parallel::world();
  if (communicator.rank >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("Global Arrays rank exceeds int");
  int low[2]{0, 0};
  int high[2]{-1, -1};
  NGA_Distribution(implementation_->handle,
                   static_cast<int>(communicator.rank), low, high);
  std::size_t local_elements = 0;
  if (high[0] >= low[0] && high[1] >= low[1]) {
    local_elements = checked_product(
        static_cast<std::size_t>(high[0] - low[0] + 1),
        static_cast<std::size_t>(high[1] - low[1] + 1),
        "Global Arrays selected local patch overflows");
  }
  const auto scientific_bytes = checked_product(
      layout().scientific_element_count(), sizeof(double),
      "selected scientific byte count overflows");
  const auto allocated_bytes = checked_product(
      layout().allocated_element_count(), sizeof(double),
      "selected allocated byte count overflows");
  return {scientific_bytes,
          allocated_bytes,
          checked_product(local_elements, sizeof(double),
                          "selected local byte count overflows"),
          scientific_bytes,
          static_cast<double>(layout().allocated_element_count()) /
              static_cast<double>(layout().scientific_element_count())};
#else
  throw std::runtime_error("Global Arrays backend is unavailable");
#endif
}

bool GlobalArraysBlockTensor::computation_block_owned(
    const std::size_t block) const {
  const auto communicator = parallel::world();
  return layout().block_owned(block, communicator.rank, communicator.size);
}

void GlobalArraysBlockTensor::zero() {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  GA_Zero(implementation_->handle);
#else
  throw std::runtime_error("Global Arrays backend is unavailable");
#endif
}

void GlobalArraysBlockTensor::synchronize() const {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  GA_Sync();
#else
  throw std::runtime_error("Global Arrays backend is unavailable");
#endif
}

void GlobalArraysBlockTensor::put_block(
    const std::size_t block, const std::vector<double>& values) {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  if (values.size() != layout().block_size(block))
    throw std::invalid_argument("selected block has wrong dimensions");
  int low[2]{checked_ga_index(block, "selected block index exceeds int"), 0};
  int high[2]{low[0],
              checked_ga_index(values.size() - 1,
                               "selected block extent exceeds int")};
  int leading_dimension = checked_ga_index(
      values.size(), "selected block leading dimension exceeds int");
  NGA_Put(implementation_->handle, low, high,
          const_cast<double*>(values.data()), &leading_dimension);
#else
  (void)block;
  (void)values;
  throw std::runtime_error("Global Arrays backend is unavailable");
#endif
}

void GlobalArraysBlockTensor::accumulate_block(
    const std::size_t block, const std::vector<double>& values) {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  if (values.size() != layout().block_size(block))
    throw std::invalid_argument("selected block has wrong dimensions");
  int low[2]{checked_ga_index(block, "selected block index exceeds int"), 0};
  int high[2]{low[0],
              checked_ga_index(values.size() - 1,
                               "selected block extent exceeds int")};
  int leading_dimension = checked_ga_index(
      values.size(), "selected block leading dimension exceeds int");
  double alpha = 1.0;
  NGA_Acc(implementation_->handle, low, high,
          const_cast<double*>(values.data()), &leading_dimension, &alpha);
#else
  (void)block;
  (void)values;
  throw std::runtime_error("Global Arrays backend is unavailable");
#endif
}

std::vector<double> GlobalArraysBlockTensor::get_block(
    const std::size_t block) const {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  std::vector<double> values(layout().block_size(block));
  int low[2]{checked_ga_index(block, "selected block index exceeds int"), 0};
  int high[2]{low[0],
              checked_ga_index(values.size() - 1,
                               "selected block extent exceeds int")};
  int leading_dimension = checked_ga_index(
      values.size(), "selected block leading dimension exceeds int");
  NGA_Get(implementation_->handle, low, high, values.data(),
          &leading_dimension);
  return values;
#else
  (void)block;
  throw std::runtime_error("Global Arrays backend is unavailable");
#endif
}

double GlobalArraysBlockTensor::dot(
    const GlobalArraysBlockTensor& other) const {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  if (!same_layout(layout(), other.layout()))
    throw std::invalid_argument("selected block dot-product layout mismatch");
  return GA_Ddot(implementation_->handle, other.implementation_->handle);
#else
  (void)other;
  throw std::runtime_error("Global Arrays backend is unavailable");
#endif
}

void GlobalArraysBlockTensor::copy_from(
    const GlobalArraysBlockTensor& other) {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  if (!same_layout(layout(), other.layout()))
    throw std::invalid_argument("selected block copy layout mismatch");
  GA_Copy(other.implementation_->handle, implementation_->handle);
  GA_Sync();
#else
  (void)other;
  throw std::runtime_error("Global Arrays backend is unavailable");
#endif
}

void GlobalArraysBlockTensor::load_replicated(
    const std::vector<std::vector<double>>& blocks) {
  if (blocks.size() != layout().block_count())
    throw std::invalid_argument("selected replicated block count mismatch");
  const auto communicator = parallel::world();
  for (std::size_t block = communicator.rank; block < blocks.size();
       block += communicator.size) {
    if (blocks[block].size() != layout().block_size(block))
      throw std::invalid_argument("selected replicated block size mismatch");
    for (const auto value : blocks[block])
      if (!std::isfinite(value))
        throw std::invalid_argument(
            "selected replicated blocks contain non-finite values");
    put_block(block, blocks[block]);
  }
  synchronize();
}

std::vector<std::vector<double>>
GlobalArraysBlockTensor::replicate() const {
  std::vector<std::vector<double>> result;
  result.reserve(layout().block_count());
  for (std::size_t block = 0; block < layout().block_count(); ++block)
    result.push_back(get_block(block));
  return result;
}

struct GlobalArraysBlockDiis::Implementation {
  struct Entry {
    Matrix singles_vector;
    Matrix singles_error;
    std::unique_ptr<GlobalArraysBlockTensor> doubles_vector;
    std::unique_ptr<GlobalArraysBlockTensor> doubles_error;
  };

  Implementation(const std::size_t input_capacity,
                 SelectedBlockLayout input_layout,
                 std::string input_name_prefix)
      : capacity(input_capacity), layout(std::move(input_layout)),
        name_prefix(std::move(input_name_prefix)) {}

  std::size_t capacity{};
  SelectedBlockLayout layout;
  std::string name_prefix;
  std::size_t next_identifier{};
  std::deque<std::unique_ptr<Entry>> entries;
};

GlobalArraysBlockDiis::GlobalArraysBlockDiis(
    const std::size_t capacity, const SelectedBlockLayout& layout,
    std::string name_prefix)
    : implementation_(std::make_unique<Implementation>(
          capacity, layout, std::move(name_prefix))) {
  if (capacity < 2)
    throw std::invalid_argument("DIIS capacity must be at least two");
  if (!global_arrays_runtime_active())
    throw std::runtime_error(
        "Global Arrays runtime must precede selected block DIIS");
  if (implementation_->name_prefix.empty())
    throw std::invalid_argument(
        "selected block DIIS name prefix is empty");
}

GlobalArraysBlockDiis::~GlobalArraysBlockDiis() = default;

Matrix GlobalArraysBlockDiis::extrapolate_distributed(
    const Matrix& singles_vector,
    const GlobalArraysBlockTensor& doubles_vector,
    const Matrix& singles_error,
    const GlobalArraysBlockTensor& doubles_error,
    GlobalArraysBlockTensor& extrapolated_doubles) {
  const auto& layout = implementation_->layout;
  if (singles_vector.dimensions() != singles_error.dimensions() ||
      !same_layout(layout, doubles_vector.layout()) ||
      !same_layout(layout, doubles_error.layout()) ||
      !same_layout(layout, extrapolated_doubles.layout()))
    throw std::invalid_argument(
        "selected block DIIS amplitude dimensions mismatch");
  if (implementation_->entries.size() == implementation_->capacity)
    implementation_->entries.pop_front();
  const auto identifier = implementation_->next_identifier++;
  auto entry = std::make_unique<Implementation::Entry>();
  entry->singles_vector = singles_vector;
  entry->singles_error = singles_error;
  entry->doubles_vector = std::make_unique<GlobalArraysBlockTensor>(
      layout, implementation_->name_prefix + "v" +
                  std::to_string(identifier));
  entry->doubles_error = std::make_unique<GlobalArraysBlockTensor>(
      layout, implementation_->name_prefix + "e" +
                  std::to_string(identifier));
  entry->doubles_vector->copy_from(doubles_vector);
  entry->doubles_error->copy_from(doubles_error);
  implementation_->entries.push_back(std::move(entry));
  const auto history_size = implementation_->entries.size();
  if (history_size < 2) {
    extrapolated_doubles.copy_from(doubles_vector);
    return singles_vector;
  }
  std::vector<std::vector<double>> products(
      history_size, std::vector<double>(history_size));
  for (std::size_t i = 0; i < history_size; ++i)
    for (std::size_t j = 0; j < history_size; ++j) {
      const auto& left = *implementation_->entries[i];
      const auto& right = *implementation_->entries[j];
      products[i][j] = dot(left.singles_error.data(),
                           right.singles_error.data()) +
                       left.doubles_error->dot(*right.doubles_error);
    }
  const auto solution = solve_diis_coefficients(products);
  if (!solution) {
    extrapolated_doubles.copy_from(doubles_vector);
    return singles_vector;
  }
  for (std::size_t discarded = 0; discarded < solution->first_history;
       ++discarded)
    implementation_->entries.pop_front();
  const auto retained_history_size = implementation_->entries.size();
  Matrix extrapolated_singles(singles_vector.dimensions());
  for (std::size_t history = 0; history < retained_history_size; ++history)
    for (std::size_t index = 0; index < extrapolated_singles.size(); ++index)
      extrapolated_singles.data()[index] +=
          solution->coefficients[history] *
          implementation_->entries[history]->singles_vector.data()[index];
  extrapolated_doubles.zero();
  const auto communicator = parallel::world();
  for (std::size_t block = communicator.rank; block < layout.block_count();
       block += communicator.size) {
    std::vector<double> values(layout.block_size(block), 0.0);
    for (std::size_t history = 0; history < retained_history_size; ++history) {
      const auto history_block =
          implementation_->entries[history]->doubles_vector->get_block(block);
      for (std::size_t index = 0; index < values.size(); ++index)
        values[index] +=
            solution->coefficients[history] * history_block[index];
    }
    extrapolated_doubles.put_block(block, values);
  }
  extrapolated_doubles.synchronize();
  return extrapolated_singles;
}

DistributedBlockDiisMemory GlobalArraysBlockDiis::memory() const {
  DistributedBlockDiisMemory result;
  result.history_entries = implementation_->entries.size();
  for (const auto& entry : implementation_->entries) {
    const auto vector_memory = entry->doubles_vector->memory();
    const auto error_memory = entry->doubles_error->memory();
    result.doubles_history_scientific_global_bytes = checked_sum(
        result.doubles_history_scientific_global_bytes,
        checked_sum(vector_memory.scientific_global_bytes,
                    error_memory.scientific_global_bytes,
                    "selected DIIS scientific bytes overflow"),
        "selected DIIS scientific bytes overflow");
    result.doubles_history_allocated_global_bytes = checked_sum(
        result.doubles_history_allocated_global_bytes,
        checked_sum(vector_memory.allocated_global_bytes,
                    error_memory.allocated_global_bytes,
                    "selected DIIS allocated bytes overflow"),
        "selected DIIS allocated bytes overflow");
    result.doubles_history_local_allocated_bytes = checked_sum(
        result.doubles_history_local_allocated_bytes,
        checked_sum(vector_memory.local_allocated_bytes,
                    error_memory.local_allocated_bytes,
                    "selected DIIS local bytes overflow"),
        "selected DIIS local bytes overflow");
    result.singles_history_bytes_per_rank = checked_sum(
        result.singles_history_bytes_per_rank,
        checked_product(
            checked_sum(entry->singles_vector.size(),
                        entry->singles_error.size(),
                        "selected DIIS singles elements overflow"),
            sizeof(double), "selected DIIS singles bytes overflow"),
        "selected DIIS singles bytes overflow");
  }
  result.doubles_history_replicated_scientific_bytes_per_rank =
      result.doubles_history_scientific_global_bytes;
  return result;
}

} // namespace localized_cc
