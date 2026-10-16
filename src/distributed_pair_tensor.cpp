#include "localized_cc/distributed_pair_tensor.hpp"

#include "localized_cc/diis_solver.hpp"
#include "localized_cc/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
#include <ga.h>
#include <gacommon.h>
#include <mpi.h>
#endif

namespace localized_cc {
namespace {

std::size_t checked_product(const std::size_t left, const std::size_t right,
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

} // namespace

OccupiedPairLayout::OccupiedPairLayout(const std::size_t occupied,
                                       const std::size_t virtuals)
    : occupied_(occupied), virtuals_(virtuals) {
  if (occupied == 0 || virtuals == 0)
    throw std::invalid_argument(
        "occupied-pair layout dimensions must be positive");
  const auto extent = checked_product(
      occupied, virtuals, "occupied-pair composite extent overflows");
  const auto elements = checked_product(
      extent, extent, "occupied-pair element count overflows");
  (void)checked_product(elements, sizeof(double),
                        "occupied-pair byte count overflows");
}

std::size_t OccupiedPairLayout::composite_extent() const noexcept {
  return occupied_ * virtuals_;
}

std::size_t OccupiedPairLayout::pair_count() const noexcept {
  return occupied_ * occupied_;
}

std::size_t OccupiedPairLayout::element_count() const noexcept {
  return composite_extent() * composite_extent();
}

std::size_t OccupiedPairLayout::global_bytes() const noexcept {
  return element_count() * sizeof(double);
}

std::size_t
OccupiedPairLayout::composite_index(const std::size_t occupied_index,
                                    const std::size_t virtual_index) const {
  if (occupied_index >= occupied_ || virtual_index >= virtuals_)
    throw std::out_of_range("occupied-pair composite index is out of range");
  return occupied_index * virtuals_ + virtual_index;
}

std::array<std::size_t, 2>
OccupiedPairLayout::pair_from_flat(const std::size_t pair) const {
  if (pair >= pair_count())
    throw std::out_of_range("occupied-pair task is out of range");
  return {pair / occupied_, pair % occupied_};
}

bool OccupiedPairLayout::pair_owned(const std::size_t pair,
                                    const std::size_t rank,
                                    const std::size_t ranks) const {
  if (ranks == 0 || rank >= ranks)
    throw std::invalid_argument("invalid occupied-pair process grid");
  if (pair >= pair_count())
    throw std::out_of_range("occupied-pair task is out of range");
  return pair % ranks == rank;
}

std::size_t
OccupiedPairLayout::owned_pair_count(const std::size_t rank,
                                     const std::size_t ranks) const {
  if (ranks == 0 || rank >= ranks)
    throw std::invalid_argument("invalid occupied-pair process grid");
  if (rank >= pair_count())
    return 0;
  return 1 + (pair_count() - 1 - rank) / ranks;
}

bool global_arrays_backend_available() noexcept {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  return true;
#else
  return false;
#endif
}

bool global_arrays_runtime_active() noexcept {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  return GA_Initialized() != 0;
#else
  return false;
#endif
}

GlobalArraysRuntime::GlobalArraysRuntime() {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  if (GA_Initialized() == 0) {
    if (GA_Initialize_comm(MPI_COMM_WORLD) == 0)
      throw std::runtime_error(
          "Global Arrays initialization on MPI_COMM_WORLD failed");
    owns_runtime_ = true;
  }
#endif
}

GlobalArraysRuntime::~GlobalArraysRuntime() {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  if (owns_runtime_ && GA_Initialized() != 0)
    GA_Terminate();
#endif
}

bool GlobalArraysRuntime::active() const noexcept {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  return GA_Initialized() != 0;
#else
  return false;
#endif
}

struct GlobalArraysPairTensor::Implementation {
  explicit Implementation(OccupiedPairLayout input_layout)
      : layout(std::move(input_layout)) {}

  OccupiedPairLayout layout;
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  int handle{};
#endif
};

GlobalArraysPairTensor::GlobalArraysPairTensor(
    const OccupiedPairLayout& layout, std::string name)
    : implementation_(std::make_unique<Implementation>(layout)) {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  if (GA_Initialized() == 0)
    throw std::runtime_error(
        "Global Arrays runtime must precede distributed tensor creation");
  const auto extent = checked_ga_index(layout.composite_extent(),
                                       "Global Arrays extent exceeds int");
  const auto chunk_size =
      checked_ga_index(layout.virtuals(), "Global Arrays chunk exceeds int");
  int dimensions[2]{extent, extent};
  int chunk[2]{chunk_size, chunk_size};
  if (name.empty())
    throw std::invalid_argument("Global Arrays tensor name is empty");
  std::vector<char> mutable_name(name.begin(), name.end());
  mutable_name.push_back('\0');
  implementation_->handle =
      NGA_Create(C_DBL, 2, dimensions, mutable_name.data(), chunk);
  if (implementation_->handle == 0)
    throw std::runtime_error("Global Arrays occupied-pair allocation failed");
  GA_Zero(implementation_->handle);
  GA_Sync();
#else
  (void)name;
  throw std::runtime_error(
      "Global Arrays support was not enabled when this target was configured");
#endif
}

GlobalArraysPairTensor::~GlobalArraysPairTensor() {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  if (implementation_ && implementation_->handle != 0 &&
      GA_Initialized() != 0) {
    GA_Sync();
    GA_Destroy(implementation_->handle);
  }
#endif
}

const OccupiedPairLayout& GlobalArraysPairTensor::layout() const noexcept {
  return implementation_->layout;
}

DistributedTensorMemory GlobalArraysPairTensor::memory() const {
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
    const auto rows = static_cast<std::size_t>(high[0] - low[0] + 1);
    const auto columns = static_cast<std::size_t>(high[1] - low[1] + 1);
    local_elements = checked_product(rows, columns,
                                     "Global Arrays local patch overflows");
  }
  return {layout().global_bytes(),
          checked_product(local_elements, sizeof(double),
                          "Global Arrays local byte count overflows"),
          layout().global_bytes(),
          {low[0], low[1]},
          {high[0], high[1]}};
#else
  throw std::runtime_error("Global Arrays backend is unavailable");
#endif
}

bool GlobalArraysPairTensor::computation_pair_owned(const std::size_t i,
                                                    const std::size_t j) const {
  if (i >= layout().occupied() || j >= layout().occupied())
    throw std::out_of_range("occupied pair is out of range");
  const auto communicator = parallel::world();
  return layout().pair_owned(i * layout().occupied() + j, communicator.rank,
                             communicator.size);
}

void GlobalArraysPairTensor::zero() {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  GA_Zero(implementation_->handle);
#else
  throw std::runtime_error("Global Arrays backend is unavailable");
#endif
}

void GlobalArraysPairTensor::synchronize() const {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  GA_Sync();
#else
  throw std::runtime_error("Global Arrays backend is unavailable");
#endif
}

void GlobalArraysPairTensor::put_pair(const std::size_t i,
                                      const std::size_t j,
                                      const std::vector<double>& values) {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  const auto virtuals = layout().virtuals();
  if (values.size() != checked_product(virtuals, virtuals,
                                       "occupied-pair block overflows"))
    throw std::invalid_argument("occupied-pair block has wrong dimensions");
  const auto first_row = layout().composite_index(i, 0);
  const auto first_column = layout().composite_index(j, 0);
  int low[2]{checked_ga_index(first_row, "Global Arrays index exceeds int"),
             checked_ga_index(first_column, "Global Arrays index exceeds int")};
  int high[2]{checked_ga_index(first_row + virtuals - 1,
                               "Global Arrays index exceeds int"),
              checked_ga_index(first_column + virtuals - 1,
                               "Global Arrays index exceeds int")};
  int leading_dimension =
      checked_ga_index(virtuals, "Global Arrays leading dimension exceeds int");
  NGA_Put(implementation_->handle, low, high,
          const_cast<double*>(values.data()), &leading_dimension);
#else
  (void)i;
  (void)j;
  (void)values;
  throw std::runtime_error("Global Arrays backend is unavailable");
#endif
}

void GlobalArraysPairTensor::accumulate_pair(
    const std::size_t i, const std::size_t j,
    const std::vector<double>& values) {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  const auto virtuals = layout().virtuals();
  if (values.size() != checked_product(virtuals, virtuals,
                                       "occupied-pair block overflows"))
    throw std::invalid_argument("occupied-pair block has wrong dimensions");
  const auto first_row = layout().composite_index(i, 0);
  const auto first_column = layout().composite_index(j, 0);
  int low[2]{checked_ga_index(first_row, "Global Arrays index exceeds int"),
             checked_ga_index(first_column, "Global Arrays index exceeds int")};
  int high[2]{checked_ga_index(first_row + virtuals - 1,
                               "Global Arrays index exceeds int"),
              checked_ga_index(first_column + virtuals - 1,
                               "Global Arrays index exceeds int")};
  int leading_dimension =
      checked_ga_index(virtuals, "Global Arrays leading dimension exceeds int");
  double alpha = 1.0;
  NGA_Acc(implementation_->handle, low, high,
          const_cast<double*>(values.data()), &leading_dimension, &alpha);
#else
  (void)i;
  (void)j;
  (void)values;
  throw std::runtime_error("Global Arrays backend is unavailable");
#endif
}

std::vector<double>
GlobalArraysPairTensor::get_pair(const std::size_t i,
                                 const std::size_t j) const {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  const auto virtuals = layout().virtuals();
  std::vector<double> values(
      checked_product(virtuals, virtuals, "occupied-pair block overflows"));
  const auto first_row = layout().composite_index(i, 0);
  const auto first_column = layout().composite_index(j, 0);
  int low[2]{checked_ga_index(first_row, "Global Arrays index exceeds int"),
             checked_ga_index(first_column, "Global Arrays index exceeds int")};
  int high[2]{checked_ga_index(first_row + virtuals - 1,
                               "Global Arrays index exceeds int"),
              checked_ga_index(first_column + virtuals - 1,
                               "Global Arrays index exceeds int")};
  int leading_dimension =
      checked_ga_index(virtuals, "Global Arrays leading dimension exceeds int");
  NGA_Get(implementation_->handle, low, high, values.data(),
          &leading_dimension);
  return values;
#else
  (void)i;
  (void)j;
  throw std::runtime_error("Global Arrays backend is unavailable");
#endif
}

double GlobalArraysPairTensor::dot(
    const GlobalArraysPairTensor& other) const {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  if (layout().occupied() != other.layout().occupied() ||
      layout().virtuals() != other.layout().virtuals())
    throw std::invalid_argument(
        "Global Arrays dot product layout mismatch");
  return GA_Ddot(implementation_->handle, other.implementation_->handle);
#else
  (void)other;
  throw std::runtime_error("Global Arrays backend is unavailable");
#endif
}

void GlobalArraysPairTensor::copy_from(
    const GlobalArraysPairTensor& other) {
#if defined(LOCALIZED_CC_HAVE_GLOBAL_ARRAYS)
  if (layout().occupied() != other.layout().occupied() ||
      layout().virtuals() != other.layout().virtuals())
    throw std::invalid_argument("Global Arrays copy layout mismatch");
  GA_Copy(other.implementation_->handle, implementation_->handle);
  GA_Sync();
#else
  (void)other;
  throw std::runtime_error("Global Arrays backend is unavailable");
#endif
}

void GlobalArraysPairTensor::load_replicated(const Tensor4& values) {
  const auto expected = std::array<std::size_t, 4>{
      layout().occupied(), layout().occupied(), layout().virtuals(),
      layout().virtuals()};
  if (values.dimensions() != expected)
    throw std::invalid_argument(
        "replicated T2 dimensions do not match occupied-pair layout");
  const auto communicator = parallel::world();
  const auto virtuals = layout().virtuals();
  std::vector<double> block(virtuals * virtuals);
  for (std::size_t pair = communicator.rank; pair < layout().pair_count();
       pair += communicator.size) {
    const auto [i, j] = layout().pair_from_flat(pair);
    for (std::size_t a = 0; a < virtuals; ++a)
      for (std::size_t b = 0; b < virtuals; ++b)
        block[a * virtuals + b] = values(i, j, a, b);
    put_pair(i, j, block);
  }
  synchronize();
}

Tensor4 GlobalArraysPairTensor::replicate() const {
  Tensor4 values({layout().occupied(), layout().occupied(),
                  layout().virtuals(), layout().virtuals()});
  for (std::size_t i = 0; i < layout().occupied(); ++i)
    for (std::size_t j = 0; j < layout().occupied(); ++j) {
      const auto block = get_pair(i, j);
      for (std::size_t a = 0; a < layout().virtuals(); ++a)
        for (std::size_t b = 0; b < layout().virtuals(); ++b)
          values(i, j, a, b) = block[a * layout().virtuals() + b];
    }
  return values;
}

struct GlobalArraysPairDiis::Implementation {
  struct Entry {
    Matrix singles_vector;
    Matrix singles_error;
    std::unique_ptr<GlobalArraysPairTensor> doubles_vector;
    std::unique_ptr<GlobalArraysPairTensor> doubles_error;
  };

  Implementation(const std::size_t input_capacity,
                 OccupiedPairLayout input_layout,
                 std::string input_name_prefix)
      : capacity(input_capacity), layout(std::move(input_layout)),
        name_prefix(std::move(input_name_prefix)) {}

  std::size_t capacity{};
  OccupiedPairLayout layout;
  std::string name_prefix;
  std::size_t next_identifier{};
  std::size_t transient_identifier{};
  std::deque<std::unique_ptr<Entry>> entries;
};

GlobalArraysPairDiis::GlobalArraysPairDiis(
    const std::size_t capacity, const OccupiedPairLayout& layout,
    std::string name_prefix)
    : implementation_(std::make_unique<Implementation>(
          capacity, layout, std::move(name_prefix))) {
  if (capacity < 2)
    throw std::invalid_argument("DIIS capacity must be at least two");
  if (!global_arrays_runtime_active())
    throw std::runtime_error(
        "Global Arrays runtime must precede distributed DIIS construction");
  if (implementation_->name_prefix.empty())
    throw std::invalid_argument("distributed DIIS name prefix is empty");
}

GlobalArraysPairDiis::~GlobalArraysPairDiis() = default;

std::pair<Matrix, Tensor4> GlobalArraysPairDiis::extrapolate(
    const Matrix& singles_vector, const Tensor4& doubles_vector,
    const Matrix& singles_error, const Tensor4& doubles_error) {
  const auto& layout = implementation_->layout;
  const auto expected_singles =
      std::array<std::size_t, 2>{layout.occupied(), layout.virtuals()};
  const auto expected_doubles = std::array<std::size_t, 4>{
      layout.occupied(), layout.occupied(), layout.virtuals(),
      layout.virtuals()};
  if (singles_vector.dimensions() != expected_singles ||
      singles_error.dimensions() != expected_singles ||
      doubles_vector.dimensions() != expected_doubles ||
      doubles_error.dimensions() != expected_doubles)
    throw std::invalid_argument("distributed DIIS amplitude dimensions mismatch");

  const auto identifier = implementation_->transient_identifier++;
  GlobalArraysPairTensor distributed_vector(
      layout, implementation_->name_prefix + "_input_vector_" +
                  std::to_string(identifier));
  GlobalArraysPairTensor distributed_error(
      layout, implementation_->name_prefix + "_input_error_" +
                  std::to_string(identifier));
  GlobalArraysPairTensor distributed_output(
      layout, implementation_->name_prefix + "_output_" +
                  std::to_string(identifier));
  distributed_vector.load_replicated(doubles_vector);
  distributed_error.load_replicated(doubles_error);
  auto extrapolated_singles = extrapolate_distributed(
      singles_vector, distributed_vector, singles_error, distributed_error,
      distributed_output);
  return {std::move(extrapolated_singles), distributed_output.replicate()};
}

Matrix GlobalArraysPairDiis::extrapolate_distributed(
    const Matrix& singles_vector,
    const GlobalArraysPairTensor& doubles_vector,
    const Matrix& singles_error,
    const GlobalArraysPairTensor& doubles_error,
    GlobalArraysPairTensor& extrapolated_doubles) {
  const auto& layout = implementation_->layout;
  const auto expected_singles =
      std::array<std::size_t, 2>{layout.occupied(), layout.virtuals()};
  const auto matching_layout = [&](const GlobalArraysPairTensor& tensor) {
    return tensor.layout().occupied() == layout.occupied() &&
           tensor.layout().virtuals() == layout.virtuals();
  };
  if (singles_vector.dimensions() != expected_singles ||
      singles_error.dimensions() != expected_singles ||
      !matching_layout(doubles_vector) || !matching_layout(doubles_error) ||
      !matching_layout(extrapolated_doubles))
    throw std::invalid_argument("distributed DIIS amplitude dimensions mismatch");

  // Destroy the oldest collective arrays before allocating replacements, so
  // the peak history never exceeds the configured capacity.
  if (implementation_->entries.size() == implementation_->capacity)
    implementation_->entries.pop_front();
  const auto identifier = implementation_->next_identifier++;
  auto entry = std::make_unique<Implementation::Entry>();
  entry->singles_vector = singles_vector;
  entry->singles_error = singles_error;
  entry->doubles_vector = std::make_unique<GlobalArraysPairTensor>(
      layout, implementation_->name_prefix + "_vector_" +
                  std::to_string(identifier));
  entry->doubles_error = std::make_unique<GlobalArraysPairTensor>(
      layout, implementation_->name_prefix + "_error_" +
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

  Matrix extrapolated_singles(expected_singles);
  for (std::size_t history = 0; history < retained_history_size; ++history)
    for (std::size_t index = 0; index < extrapolated_singles.size(); ++index)
      extrapolated_singles.data()[index] +=
          solution->coefficients[history] *
          implementation_->entries[history]->singles_vector.data()[index];

  extrapolated_doubles.zero();
  const auto communicator = parallel::world();
  const auto virtuals = layout.virtuals();
  for (std::size_t pair = communicator.rank; pair < layout.pair_count();
       pair += communicator.size) {
    const auto [i, j] = layout.pair_from_flat(pair);
    std::vector<double> block(virtuals * virtuals);
    for (std::size_t history = 0; history < retained_history_size; ++history) {
      const auto history_block =
          implementation_->entries[history]->doubles_vector->get_pair(i, j);
      for (std::size_t index = 0; index < block.size(); ++index)
        block[index] +=
            solution->coefficients[history] * history_block[index];
    }
    extrapolated_doubles.put_pair(i, j, block);
  }
  extrapolated_doubles.synchronize();
  return extrapolated_singles;
}

DistributedPairDiisMemory GlobalArraysPairDiis::memory() const {
  DistributedPairDiisMemory result;
  result.history_entries = implementation_->entries.size();
  for (const auto& entry : implementation_->entries) {
    const auto vector_memory = entry->doubles_vector->memory();
    const auto error_memory = entry->doubles_error->memory();
    result.doubles_history_global_bytes = checked_sum(
        result.doubles_history_global_bytes,
        checked_sum(vector_memory.global_bytes, error_memory.global_bytes,
                    "distributed DIIS global byte count overflows"),
        "distributed DIIS global byte count overflows");
    result.doubles_history_local_bytes = checked_sum(
        result.doubles_history_local_bytes,
        checked_sum(vector_memory.local_bytes, error_memory.local_bytes,
                    "distributed DIIS local byte count overflows"),
        "distributed DIIS local byte count overflows");
    result.singles_history_bytes_per_rank = checked_sum(
        result.singles_history_bytes_per_rank,
        checked_product(
            checked_sum(entry->singles_vector.size(),
                        entry->singles_error.size(),
                        "distributed DIIS singles element count overflows"),
            sizeof(double),
            "distributed DIIS singles byte count overflows"),
        "distributed DIIS singles byte count overflows");
  }
  result.doubles_history_replicated_bytes_per_rank =
      result.doubles_history_global_bytes;
  return result;
}

} // namespace localized_cc
