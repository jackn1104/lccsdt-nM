#include "localized_cc/canonical_triples.hpp"
#include "localized_cc/cuda_triples.hpp"
#include "localized_cc/parallel.hpp"
#include "localized_cc/selected_ccsd_storage.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace localized_cc {
namespace {

constexpr std::array<std::array<std::size_t, 3>, 6> permutations{{
    {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}},
    {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}}}};
constexpr std::array<double, 6> permutation_sign{{1.0, -1.0, -1.0,
                                                  1.0, 1.0, -1.0}};

struct Triple {
  std::size_t first{};
  std::size_t second{};
  std::size_t third{};
};

std::size_t checked_cube(const std::size_t value) {
  if (value == 0 || value > std::numeric_limits<std::size_t>::max() / value)
    throw std::overflow_error("canonical triples dimension overflow");
  const auto square = value * value;
  if (square > std::numeric_limits<std::size_t>::max() / value)
    throw std::overflow_error("canonical triples dimension overflow");
  return square * value;
}

std::size_t checked_product(const std::size_t left,
                            const std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error("canonical triples packed tile size overflow");
  return left * right;
}

std::vector<Triple> unique_triples(const std::size_t extent) {
  std::vector<Triple> result;
  for (std::size_t first = 0; first < extent; ++first)
    for (std::size_t second = first + 1; second < extent; ++second)
      for (std::size_t third = second + 1; third < extent; ++third)
        result.push_back({first, second, third});
  return result;
}

std::size_t flat3(const std::size_t first, const std::size_t second,
                  const std::size_t third, const std::size_t extent) {
  return (first * extent + second) * extent + third;
}

template <typename Eri>
bool eri_values_are_finite(const Eri& eri) {
  if constexpr (std::is_same_v<std::remove_cvref_t<Eri>, Tensor4>) {
    for (const auto value : eri.data())
      if (!std::isfinite(value))
        return false;
    return true;
  } else {
    return eri.all_finite();
  }
}

template <typename T2>
bool t2_values_are_finite(const T2& t2) {
  if constexpr (std::is_same_v<std::remove_cvref_t<T2>, Tensor4>) {
    for (const auto value : t2.data())
      if (!std::isfinite(value))
        return false;
    return true;
  } else {
    return t2.all_finite();
  }
}

template <typename Eri, typename T2>
void validate_inputs(const Matrix& fock, const Eri& eri, const Matrix& t1,
                     const T2& t2,
                     const CanonicalTriplesOptions& options) {
  const auto occupied = t1.dimensions()[0];
  const auto virtuals = t1.dimensions()[1];
  const auto orbitals = occupied + virtuals;
  if (occupied < 3 || virtuals < 3)
    throw std::invalid_argument(
        "canonical triples requires at least three occupied and virtual spin orbitals");
  if (fock.dimensions() != std::array<std::size_t, 2>{orbitals, orbitals} ||
      eri.dimensions() !=
          std::array<std::size_t, 4>{orbitals, orbitals, orbitals, orbitals} ||
      t2.dimensions() !=
          std::array<std::size_t, 4>{occupied, occupied, virtuals, virtuals})
    throw std::invalid_argument("inconsistent canonical triples dimensions");
  if (!(options.fock_offdiagonal_tolerance >= 0.0) ||
      !std::isfinite(options.fock_offdiagonal_tolerance) ||
      !(options.denominator_floor > 0.0) ||
      !std::isfinite(options.denominator_floor) ||
      options.occupied_task_grain == 0 || options.virtual_tile_size == 0)
    throw std::invalid_argument("invalid canonical triples options");
  if (options.backend != CanonicalTriplesBackend::reference_cpu &&
      options.compute_bracket)
    throw std::invalid_argument(
        "tiled triples backends support parenthetical CCSD(T) only");
  for (const auto value : fock.data())
    if (!std::isfinite(value))
      throw std::invalid_argument("non-finite canonical triples Fock matrix");
  if (!eri_values_are_finite(eri))
    throw std::invalid_argument("non-finite canonical triples ERIs");
  for (const auto value : t1.data())
    if (!std::isfinite(value))
      throw std::invalid_argument("non-finite canonical triples T1 amplitudes");
  if (!t2_values_are_finite(t2))
    throw std::invalid_argument("non-finite canonical triples T2 amplitudes");
}

double maximum_offdiagonal(const Matrix& fock) {
  const auto orbitals = fock.dimensions()[0];
  double maximum = 0.0;
  for (std::size_t row = 0; row < orbitals; ++row)
    for (std::size_t column = 0; column < orbitals; ++column)
      if (row != column)
        maximum = std::max(maximum, std::abs(fock(row, column)));
  return maximum;
}

struct Worker {
  Worker(const std::size_t virtuals, const std::size_t virtual_triples)
      : raw(checked_cube(virtuals)), doubles_moment(virtual_triples),
        singles_moment(virtual_triples) {}

  std::vector<double> raw;
  std::vector<double> doubles_moment;
  std::vector<double> singles_moment;
};

template <typename Eri, typename T2>
std::pair<double, double> evaluate_occupied_triple(
    const Triple occupied_set, const std::vector<Triple>& virtual_sets,
    const Matrix& fock, const Eri& eri, const Matrix& t1,
    const T2& t2, Worker& worker, const bool compute_bracket) {
  const auto occupied = t1.dimensions()[0];
  const auto virtuals = t1.dimensions()[1];
  const std::array<std::size_t, 3> occupied_indices{
      occupied_set.first, occupied_set.second, occupied_set.third};
  std::fill(worker.doubles_moment.begin(), worker.doubles_moment.end(), 0.0);
  std::fill(worker.singles_moment.begin(), worker.singles_moment.end(), 0.0);

  // NWChem TCE forms separate doubles and singles triples numerators for each
  // six-index task. Here each unique occupied triple is one deterministic
  // task; the complete P(ijk)P(abc) antisymmetrizer is accumulated explicitly.
  for (std::size_t occupied_permutation = 0;
       occupied_permutation < permutations.size(); ++occupied_permutation) {
    const auto& po = permutations[occupied_permutation];
    const auto i = occupied_indices[po[0]];
    const auto j = occupied_indices[po[1]];
    const auto k = occupied_indices[po[2]];

    for (std::size_t a = 0; a < virtuals; ++a)
      for (std::size_t b = 0; b < virtuals; ++b)
        for (std::size_t c = 0; c < virtuals; ++c) {
          double value = 0.0;
          for (std::size_t e = 0; e < virtuals; ++e)
            value += t2(j, k, a, e) *
                     eri(occupied + b, occupied + c, occupied + e, i);
          for (std::size_t m = 0; m < occupied; ++m)
            value -= t2(i, m, b, c) * eri(m, occupied + a, j, k);
          worker.raw[flat3(a, b, c, virtuals)] = value;
        }

    for (std::size_t virtual_index = 0;
         virtual_index < virtual_sets.size(); ++virtual_index) {
      const auto virtual_set = virtual_sets[virtual_index];
      const std::array<std::size_t, 3> virtual_indices{
          virtual_set.first, virtual_set.second, virtual_set.third};
      for (std::size_t virtual_permutation = 0;
           virtual_permutation < permutations.size(); ++virtual_permutation) {
        const auto& pv = permutations[virtual_permutation];
        const auto a = virtual_indices[pv[0]];
        const auto b = virtual_indices[pv[1]];
        const auto c = virtual_indices[pv[2]];
        const auto sign = permutation_sign[occupied_permutation] *
                          permutation_sign[virtual_permutation];
        worker.doubles_moment[virtual_index] +=
            0.25 * sign * worker.raw[flat3(a, b, c, virtuals)];
        worker.singles_moment[virtual_index] +=
            0.25 * sign *
            (t1(i, a) * eri(occupied + b, occupied + c, j, k) +
             fock(occupied + a, i) * t2(j, k, b, c));
      }
    }
  }

  double bracket = 0.0;
  double parenthetical = 0.0;
  for (std::size_t virtual_index = 0; virtual_index < virtual_sets.size();
       ++virtual_index) {
    const auto virtual_set = virtual_sets[virtual_index];
    const double denominator =
        fock(occupied_set.first, occupied_set.first) +
        fock(occupied_set.second, occupied_set.second) +
        fock(occupied_set.third, occupied_set.third) -
        fock(occupied + virtual_set.first, occupied + virtual_set.first) -
        fock(occupied + virtual_set.second, occupied + virtual_set.second) -
        fock(occupied + virtual_set.third, occupied + virtual_set.third);
    const auto doubles = worker.doubles_moment[virtual_index];
    const auto singles = worker.singles_moment[virtual_index];
    if (compute_bracket)
      bracket += doubles * doubles / denominator;
    parenthetical += doubles * (doubles + singles) / denominator;
  }
  return {bracket, parenthetical};
}

template <typename Eri, typename T2>
TriplesMomentTile pack_moment_tile(
    const Triple occupied_set, const std::array<std::size_t, 3> offsets,
    const std::array<std::size_t, 3> dimensions, const Matrix& fock,
    const Eri& eri, const Matrix& t1, const T2& t2,
    const double denominator_floor, TriplesMomentTile tile = {},
    const bool shared_pair_packing = false,
    const std::size_t virtual_stride = 1) {
  const auto occupied = t1.dimensions()[0];
  const auto virtuals = t1.dimensions()[1];
  const std::array<std::size_t, 3> occupied_indices{
      occupied_set.first, occupied_set.second, occupied_set.third};
  tile.occupied_count = occupied;
  tile.virtual_count = virtuals;
  tile.virtual_offsets = offsets;
  tile.virtual_dimensions = dimensions;
  tile.virtual_stride = virtual_stride;
  tile.exact_spin_channels = virtual_stride == 2;
  tile.antisymmetric_virtual_integrals = shared_pair_packing && virtual_stride == 2;
  for(std::size_t axis=0;axis<3;++axis)
    tile.occupied_spins[axis]=occupied_indices[axis]%2;
  tile.denominator_floor = denominator_floor;
  tile.pair_operand_sources = shared_pair_packing
      ? std::array<std::size_t, 6>{0, 0, 2, 2, 4, 4}
      : std::array<std::size_t, 6>{0, 1, 2, 3, 4, 5};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    tile.occupied_energies[axis] =
        fock(occupied_indices[axis], occupied_indices[axis]);
    tile.virtual_energies[axis].resize(dimensions[axis]);
    for (std::size_t local = 0; local < dimensions[axis]; ++local) {
      const auto global = offsets[axis] + virtual_stride * local;
      tile.virtual_energies[axis][local] =
          fock(occupied + global, occupied + global);
    }
  }

  for (std::size_t occupied_permutation = 0;
       occupied_permutation < permutations.size(); ++occupied_permutation) {
    const auto& po = permutations[occupied_permutation];
    const auto i = occupied_indices[po[0]];
    const auto j = occupied_indices[po[1]];
    const auto k = occupied_indices[po[2]];
    auto& data = tile.occupied_permutations[occupied_permutation];
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const auto dimension = dimensions[axis];
      data.t2_jk_axis_e[axis].resize(
          checked_product(dimension, virtuals));
      data.eri_m_axis[axis].resize(checked_product(occupied, dimension));
      data.t1_axis[axis].resize(dimension);
      data.fock_axis[axis].resize(dimension);
      for (std::size_t local = 0; local < dimension; ++local) {
        const auto a = offsets[axis] + virtual_stride * local;
        data.t1_axis[axis][local] = t1(i, a);
        data.fock_axis[axis][local] = fock(occupied + a, i);
        for (std::size_t e = 0; e < virtuals; ++e)
          data.t2_jk_axis_e[axis][local * virtuals + e] = t2(j, k, a, e);
        for (std::size_t m = 0; m < occupied; ++m)
          data.eri_m_axis[axis][m * dimension + local] =
              (tile.exact_spin_channels && m%2+a%2!=j%2+k%2) ? 0.0 :
              eri(m, occupied + a, j, k);
      }
    }

    for (std::size_t virtual_permutation = 0;
         virtual_permutation < permutations.size(); ++virtual_permutation) {
      const auto& pv = permutations[virtual_permutation];
      const auto b_axis = pv[1];
      const auto c_axis = pv[2];
      const auto b_dimension = dimensions[b_axis];
      const auto c_dimension = dimensions[c_axis];
      const auto pair_size = checked_product(b_dimension, c_dimension);
      const bool owns_pairs = tile.pair_operand_sources[occupied_permutation] ==
                              occupied_permutation;
      data.eri_pair_e[virtual_permutation].resize(
          owns_pairs ? checked_product(pair_size, virtuals) : 0);
      data.t2_m_pair[virtual_permutation].resize(
          owns_pairs ? checked_product(occupied, pair_size) : 0);
      data.eri_pair_jk[virtual_permutation].resize(pair_size);
      // Allocate outside OpenMP; the shared path packs all permutations in
      // one coarse parallel region below, instead of 36 fork/join regions.
      if (shared_pair_packing) continue;
      // Operands are immutable; each worker owns a distinct packed pair.
      // Preserve fail-closed ERI/T2 exceptions across the OpenMP boundary.
      std::exception_ptr packing_failure;
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static) if(pair_size >= 256)
#endif
      for (std::size_t b_local = 0; b_local < b_dimension; ++b_local)
        for (std::size_t c_local = 0; c_local < c_dimension; ++c_local) {
          try {
          const auto b = offsets[b_axis] + virtual_stride * b_local;
          const auto c = offsets[c_axis] + virtual_stride * c_local;
          const auto pair = b_local * c_dimension + c_local;
          data.eri_pair_jk[virtual_permutation][pair] =
              eri(occupied + b, occupied + c, j, k);
          for (std::size_t e = 0; e < virtuals; ++e)
            data.eri_pair_e[virtual_permutation][pair * virtuals + e] =
                (tile.exact_spin_channels && b%2+c%2!=e%2+i%2) ? 0.0 :
                eri(occupied + b, occupied + c, occupied + e, i);
          for (std::size_t m = 0; m < occupied; ++m)
            data.t2_m_pair[virtual_permutation][m * pair_size + pair] =
                t2(i, m, b, c);
          } catch (...) {
#if defined(_OPENMP)
#pragma omp critical(triples_packing_failure)
#endif
            { if (!packing_failure) packing_failure = std::current_exception(); }
          }
        }
      if (packing_failure) std::rethrow_exception(packing_failure);
    }
  }
  if (shared_pair_packing) {
    // Consecutive occupied permutations have identical i but swapped j,k.
    // G(bc,ei) and T2(im,bc) therefore coincide literally. Do not share
    // G(bc,jk), G(ma,jk), or T2(jk,ae), and do not omit any contraction.
    const auto maximum_dimension = *std::max_element(dimensions.begin(), dimensions.end());
    std::exception_ptr packing_failure;
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static, 1) if(maximum_dimension >= 8)
#endif
    for (std::size_t task = 0; task < 36; ++task)
      for (std::size_t b_local = 0; b_local < maximum_dimension; ++b_local) {
        try {
          const auto op = task / permutations.size();
          const auto vp = task % permutations.size();
          const auto& po = permutations[op];
          const auto& pv = permutations[vp];
          const auto i = occupied_indices[po[0]];
          const auto j = occupied_indices[po[1]];
          const auto k = occupied_indices[po[2]];
          const auto b_axis = pv[1], c_axis = pv[2];
          if (b_local >= dimensions[b_axis]) continue;
          const auto pair_size = checked_product(dimensions[b_axis], dimensions[c_axis]);
          const auto b = offsets[b_axis] + virtual_stride * b_local;
          auto& data = tile.occupied_permutations[op];
          const bool owns_pairs = tile.pair_operand_sources[op] == op;
          for (std::size_t c_local = 0; c_local < dimensions[c_axis]; ++c_local) {
            const auto c = offsets[c_axis] + virtual_stride * c_local;
            const auto pair = b_local * dimensions[c_axis] + c_local;
            data.eri_pair_jk[vp][pair] = eri(occupied + b, occupied + c, j, k);
            if (!owns_pairs) continue;
            for (std::size_t e = 0; e < virtuals; ++e)
              data.eri_pair_e[vp][pair * virtuals + e] =
                  (tile.exact_spin_channels && b%2+c%2!=e%2+i%2) ? 0.0 :
                  eri(occupied + b, occupied + c, occupied + e, i);
            for (std::size_t m = 0; m < occupied; ++m)
              data.t2_m_pair[vp][m * pair_size + pair] = t2(i, m, b, c);
          }
        } catch (...) {
#if defined(_OPENMP)
#pragma omp critical(triples_packing_failure)
#endif
          { if (!packing_failure) packing_failure = std::current_exception(); }
        }
      }
    if (packing_failure) std::rethrow_exception(packing_failure);
  }
  return tile;
}

struct TiledTriplesEnergy {
  double parenthetical{};
  std::size_t workspace_bytes{};
};

TriplesMomentResult contract_moment_tile(
    const TriplesMomentTile& tile, const CanonicalTriplesBackend backend,
    const int cuda_device) {
  if (backend == CanonicalTriplesBackend::tce_cuda)
    return contract_triples_moment_tce_cuda(tile, cuda_device);
  if (backend == CanonicalTriplesBackend::cuda)
    return contract_triples_moment_cuda(tile, cuda_device);
  return contract_triples_moment_cpu(tile);
}

template <typename Eri, typename T2>
TiledTriplesEnergy evaluate_occupied_triple_tiled(
    const Triple occupied_set, const std::size_t expected_virtual_triples,
    const Matrix& fock, const Eri& eri, const Matrix& t1, const T2& t2,
    const CanonicalTriplesOptions& options) {
  const auto virtuals = t1.dimensions()[1];
  const auto tile_size = std::min(options.virtual_tile_size, virtuals);
  TiledTriplesEnergy energy;
  std::size_t evaluated = 0;
  for (std::size_t offset0 = 0; offset0 < virtuals; offset0 += tile_size)
    for (std::size_t offset1 = offset0; offset1 < virtuals;
         offset1 += tile_size)
      for (std::size_t offset2 = offset1; offset2 < virtuals;
           offset2 += tile_size) {
        const std::array<std::size_t, 3> offsets{offset0, offset1, offset2};
        const std::array<std::size_t, 3> dimensions{
            std::min(tile_size, virtuals - offset0),
            std::min(tile_size, virtuals - offset1),
            std::min(tile_size, virtuals - offset2)};
        const auto tile = pack_moment_tile(occupied_set, offsets, dimensions,
                                           fock, eri, t1, t2,
                                           options.denominator_floor);
        const auto contribution = contract_moment_tile(
            tile, options.backend, options.cuda_device);
        energy.parenthetical += contribution.parenthetical;
        energy.workspace_bytes =
            std::max(energy.workspace_bytes, contribution.workspace_bytes);
        evaluated += contribution.evaluated_virtual_triples;
      }
  if (evaluated != expected_virtual_triples)
    throw std::runtime_error(
        "tiled triples traversal did not cover each virtual triple once");
  return energy;
}

void validate_denominators(const Matrix& fock,
                           const std::vector<Triple>& occupied_sets,
                           const std::vector<Triple>& virtual_sets,
                           const std::size_t occupied,
                           const double denominator_floor) {
  for (const auto occupied_set : occupied_sets)
    for (const auto virtual_set : virtual_sets) {
      const double denominator =
          fock(occupied_set.first, occupied_set.first) +
          fock(occupied_set.second, occupied_set.second) +
          fock(occupied_set.third, occupied_set.third) -
          fock(occupied + virtual_set.first, occupied + virtual_set.first) -
          fock(occupied + virtual_set.second, occupied + virtual_set.second) -
          fock(occupied + virtual_set.third, occupied + virtual_set.third);
      if (std::abs(denominator) < denominator_floor)
        throw std::runtime_error("small canonical triples denominator");
    }
}

} // namespace

template <typename Eri, typename T2>
CanonicalTriplesResult solve_canonical_triples_streamed_impl(
    const Matrix& fock, const Eri& eri, const Matrix& t1,
    const T2& t2, const CanonicalTriplesOptions& options) {
  validate_inputs(fock, eri, t1, t2, options);
  const auto occupied = t1.dimensions()[0];
  const auto virtuals = t1.dimensions()[1];
  const auto occupied_sets = unique_triples(occupied);
  const auto virtual_sets = unique_triples(virtuals);

  if (canonical_triples_backend_uses_cuda(options.backend)) {
    const auto runtime = cuda_triples_runtime_info();
    if (!runtime.compiled || !runtime.available)
      throw std::runtime_error(
          "CUDA triples backend requested without a visible CUDA device");
  }

  CanonicalTriplesResult result;
  result.backend = options.backend;
  result.maximum_fock_offdiagonal = maximum_offdiagonal(fock);
  if (result.maximum_fock_offdiagonal > options.fock_offdiagonal_tolerance)
    throw std::invalid_argument(
        "streamed canonical triples requires a diagonal Fock matrix");
  validate_denominators(fock, occupied_sets, virtual_sets, occupied,
                        options.denominator_floor);
  result.occupied_triples = occupied_sets.size();
  result.virtual_triples = virtual_sets.size();
  result.tasks = (occupied_sets.size() + options.occupied_task_grain - 1) /
                 options.occupied_task_grain;
  const bool restarting = !options.completed_tasks.empty() ||
                          !options.initial_bracket_task_energies.empty() ||
                          !options.initial_parenthetical_task_energies.empty();
  if (restarting &&
      (options.completed_tasks.size() != result.tasks ||
       options.initial_bracket_task_energies.size() != result.tasks ||
       options.initial_parenthetical_task_energies.size() != result.tasks))
    throw std::invalid_argument(
        "canonical triples restart task dimensions mismatch");
  result.bracket_task_energies = restarting
      ? options.initial_bracket_task_energies
      : std::vector<double>(result.tasks, 0.0);
  result.parenthetical_task_energies = restarting
      ? options.initial_parenthetical_task_energies
      : std::vector<double>(result.tasks, 0.0);
  if (!options.compute_bracket)
    std::fill(result.bracket_task_energies.begin(),
              result.bracket_task_energies.end(), 0.0);
  std::vector<std::uint8_t> completed = restarting
      ? options.completed_tasks
      : std::vector<std::uint8_t>(result.tasks, 0U);
  for (std::size_t task = 0; task < result.tasks; ++task) {
    if (completed[task] > 1U)
      throw std::invalid_argument("invalid canonical triples completion flag");
    if (completed[task] == 0U) {
      result.bracket_task_energies[task] = 0.0;
      result.parenthetical_task_energies[task] = 0.0;
    }
  }
  if (options.backend == CanonicalTriplesBackend::reference_cpu) {
    const long double workspace_elements =
        static_cast<long double>(checked_cube(virtuals)) +
        2.0L * static_cast<long double>(virtual_sets.size());
    result.workspace_per_worker_mib =
        static_cast<double>(workspace_elements * sizeof(double) / 1048576.0L);
  }
  const auto communicator = parallel::world();

  const auto wave_size = options.checkpoint_interval_tasks == 0
      ? std::max<std::size_t>(1, result.tasks)
      : options.checkpoint_interval_tasks;
  for (std::size_t wave_begin = 0; wave_begin < result.tasks;
       wave_begin += wave_size) {
    const auto wave_end = std::min(result.tasks, wave_begin + wave_size);
    bool wave_has_work = false;
    for (std::size_t task = wave_begin; task < wave_end; ++task)
      wave_has_work = wave_has_work || completed[task] == 0U;
    if (!wave_has_work)
      continue;
    std::vector<double> bracket_delta(result.tasks, 0.0);
    std::vector<double> parenthetical_delta(result.tasks, 0.0);
    std::vector<double> workspace_delta(result.tasks, 0.0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)                                \
    if (!canonical_triples_backend_uses_cuda(options.backend))
#endif
    for (std::ptrdiff_t raw_task = static_cast<std::ptrdiff_t>(wave_begin);
         raw_task < static_cast<std::ptrdiff_t>(wave_end); ++raw_task) {
      const auto task = static_cast<std::size_t>(raw_task);
      if (completed[task] != 0U || task % communicator.size != communicator.rank)
        continue;
      double bracket = 0.0;
      double parenthetical = 0.0;
      std::size_t maximum_workspace_bytes = 0;
      const auto begin = task * options.occupied_task_grain;
      const auto end = std::min(begin + options.occupied_task_grain,
                                occupied_sets.size());
      for (std::size_t occupied_index = begin; occupied_index < end;
           ++occupied_index) {
        if (options.backend == CanonicalTriplesBackend::reference_cpu) {
          Worker worker(virtuals, virtual_sets.size());
          const auto energy = evaluate_occupied_triple(
              occupied_sets[occupied_index], virtual_sets, fock, eri, t1, t2,
              worker, options.compute_bracket);
          bracket += energy.first;
          parenthetical += energy.second;
        } else {
          const auto energy = evaluate_occupied_triple_tiled(
              occupied_sets[occupied_index], virtual_sets.size(), fock, eri,
              t1, t2, options);
          parenthetical += energy.parenthetical;
          maximum_workspace_bytes =
              std::max(maximum_workspace_bytes, energy.workspace_bytes);
        }
      }
      bracket_delta[task] = bracket;
      parenthetical_delta[task] = parenthetical;
      workspace_delta[task] = static_cast<double>(maximum_workspace_bytes);
    }
    parallel::sum_in_place(bracket_delta);
    parallel::sum_in_place(parenthetical_delta);
    parallel::sum_in_place(workspace_delta);
    for (const auto bytes : workspace_delta)
      result.tile_workspace_per_task_mib =
          std::max(result.tile_workspace_per_task_mib, bytes / 1048576.0);
    for (std::size_t task = wave_begin; task < wave_end; ++task) {
      if (completed[task] == 0U) {
        result.bracket_task_energies[task] = bracket_delta[task];
        result.parenthetical_task_energies[task] =
            parenthetical_delta[task];
        completed[task] = 1U;
      }
    }
    if (options.observer)
      options.observer(CanonicalTriplesProgress{
          result.bracket_task_energies,
          result.parenthetical_task_energies, completed});
  }

  // Fixed task-order summation keeps the result deterministic even though the
  // tasks may be scheduled dynamically across OpenMP workers.
  for (std::size_t task = 0; task < result.tasks; ++task) {
    result.bracket_energy += result.bracket_task_energies[task];
    result.parenthetical_energy +=
        result.parenthetical_task_energies[task];
  }
  return result;
}

CanonicalTriplesResult solve_canonical_triples_streamed(
    const Matrix& fock, const Tensor4& eri, const Matrix& t1,
    const Tensor4& t2, const CanonicalTriplesOptions& options) {
  return solve_canonical_triples_streamed_impl(fock, eri, t1, t2, options);
}

CanonicalTriplesResult solve_canonical_triples_streamed(
    const SpatialRhfHamiltonian& hamiltonian, const Matrix& t1,
    const Tensor4& t2, const CanonicalTriplesOptions& options) {
  const auto spin_fock =
      expand_spatial_one_body_to_spin_orbital(hamiltonian.fock);
  const SpinOrbitalEriView spin_eri(hamiltonian);
  return solve_canonical_triples_streamed_impl(spin_fock, spin_eri, t1, t2,
                                                options);
}

template <typename Eri, typename T2>
void stream_localized_triples_moments_impl(
    const Matrix& fock, const Eri& eri, const Matrix& t1,
    const T2& t2,
    const LocalizedTriplesMomentStreamOptions& options) {
  if (!options.observer)
    throw std::invalid_argument(
        "localized triples moment streaming requires an observer");
  if (options.shared_pair_packing &&
      options.backend != CanonicalTriplesBackend::tiled_cpu)
    throw std::invalid_argument("shared triples pair packing requires tiled CPU moments");
  if (options.spin_aware_construction &&
      (options.backend != CanonicalTriplesBackend::tiled_cpu ||
       std::is_same_v<std::remove_cvref_t<Eri>, Tensor4>))
    throw std::invalid_argument("spin-aware moments require the CPU spatial-Hamiltonian path");
  if (options.backend == CanonicalTriplesBackend::reference_cpu)
    throw std::invalid_argument(
        "localized triples moment streaming requires a bounded tiled "
        "backend");
  CanonicalTriplesOptions validation;
  validation.backend = options.backend;
  validation.compute_bracket = false;
  validation.virtual_tile_size = options.virtual_tile_size;
  validation.denominator_floor = options.denominator_floor;
  validation.cuda_device = options.cuda_device;
  validation.fock_offdiagonal_tolerance =
      std::numeric_limits<double>::max();
  validate_inputs(fock, eri, t1, t2, validation);
  if (canonical_triples_backend_uses_cuda(options.backend)) {
    const auto runtime = cuda_triples_runtime_info();
    if (!runtime.compiled || !runtime.available)
      throw std::runtime_error(
          "CUDA localized triples moments requested without a visible "
          "CUDA device");
  }

  const auto occupied = t1.dimensions()[0];
  const auto virtuals = t1.dimensions()[1];
  const auto occupied_sets = unique_triples(occupied);
  const auto tile_size = std::min(options.virtual_tile_size, virtuals);
  const auto communicator = parallel::world();
  std::vector<std::size_t> selected_rows;
  for (std::size_t task = communicator.rank; task < occupied_sets.size();
       task += communicator.size) {
    const auto triple = occupied_sets[task];
    if (!options.occupied_representative ||
        options.occupied_representative({triple.first, triple.second, triple.third}))
      selected_rows.push_back(task);
  }
  const auto local_rows = selected_rows.size();
  std::size_t completed_rows = 0;
  double packing_seconds = 0.0, contraction_seconds = 0.0;
  TriplesMomentTile packed_tile;
  for (const auto occupied_task : selected_rows) {
    const auto occupied_set = occupied_sets[occupied_task];
    const auto occupied_beta_count = occupied_set.first % 2 +
        occupied_set.second % 2 + occupied_set.third % 2;
    for (std::size_t offset0 = 0; offset0 < virtuals;
         offset0 += tile_size)
      for (std::size_t offset1 = offset0; offset1 < virtuals;
           offset1 += tile_size)
        for (std::size_t offset2 = offset1; offset2 < virtuals;
             offset2 += tile_size) {
          const std::array<std::size_t, 3> offsets{offset0, offset1,
                                                   offset2};
          if (options.retain_block &&
              !options.retain_block({occupied_set.first, occupied_set.second,
                                     occupied_set.third}, offsets))
            continue;
          const std::array<std::size_t, 3> dimensions{
              std::min(tile_size, virtuals - offset0),
              std::min(tile_size, virtuals - offset1),
              std::min(tile_size, virtuals - offset2)};
          const auto emit_tile = [&](const std::array<std::size_t, 3>& tile_offsets,
                                     const std::array<std::size_t, 3>& tile_dimensions,
                                     const std::size_t stride) {
          const auto packing_start = std::chrono::steady_clock::now();
          packed_tile = pack_moment_tile(
              occupied_set, tile_offsets, tile_dimensions, fock, eri, t1, t2,
              options.denominator_floor, std::move(packed_tile),
              options.shared_pair_packing, stride);
          const auto contraction_start = std::chrono::steady_clock::now();
          packing_seconds += std::chrono::duration<double>(
              contraction_start - packing_start).count();
          auto moments = contract_moment_tile(
              packed_tile, options.backend, options.cuda_device);
          contraction_seconds += std::chrono::duration<double>(
              std::chrono::steady_clock::now() - contraction_start).count();
          options.observer(LocalizedTriplesMomentBlock{
              .occupied_indices = {occupied_set.first, occupied_set.second,
                                   occupied_set.third},
              .virtual_offsets = tile_offsets,
              .virtual_dimensions = tile_dimensions,
              .doubles_moment = std::move(moments.doubles_moment),
              .left_moment = std::move(moments.left_moment),
              .used_cuda = moments.used_cuda,
              .device = moments.device,
              .virtual_stride = stride});
          };
          if (!options.spin_aware_construction) {
            emit_tile(offsets, dimensions, 1);
            continue;
          }
          // Split each original range into disjoint even/odd subranges. Keep
          // the original retain_block decision above (screening budgets and
          // block identities do not change). No orbital or internal sum is cut.
          for (std::size_t mask = 0; mask < 8; ++mask) {
            const std::array<std::size_t, 3> parity{
                mask & 1U, (mask >> 1U) & 1U, (mask >> 2U) & 1U};
            if (parity[0] + parity[1] + parity[2] != occupied_beta_count)
              continue;
            std::array<std::size_t, 3> starts{}, counts{}, lasts{};
            bool empty = false;
            for (std::size_t axis = 0; axis < 3; ++axis) {
              const auto shift = (parity[axis] + 2 - offsets[axis] % 2) % 2;
              if (shift >= dimensions[axis]) { empty = true; break; }
              starts[axis] = offsets[axis] + shift;
              counts[axis] = 1 + (dimensions[axis] - 1 - shift) / 2;
              lasts[axis] = starts[axis] + 2 * (counts[axis] - 1);
            }
            if (empty) continue;
            // Greedy first increasing triple detects subtiles with no a<b<c
            // before packing, including equal parent tiles and ragged tails.
            auto previous = starts[0];
            for (std::size_t axis = 1; axis < 3; ++axis) {
              if (previous >= lasts[axis]) { empty = true; break; }
              auto first = starts[axis];
              if (first <= previous)
                first += 2 * ((previous - first) / 2 + 1);
              if (first > lasts[axis]) { empty = true; break; }
              previous = first;
            }
            if (!empty) emit_tile(starts, counts, 2);
          }
        }
    ++completed_rows;
    if(options.completed_row_observer)
      options.completed_row_observer({occupied_set.first,occupied_set.second,occupied_set.third});
    if (communicator.rank == 0 && options.row_observer &&
        (completed_rows == 1 || completed_rows % 16 == 0 || completed_rows == local_rows))
      options.row_observer(completed_rows, local_rows);
  }
  if (local_rows == 0 && communicator.rank == 0 && options.row_observer)
    options.row_observer(0, 0);
  if (options.timing_observer)
    options.timing_observer(packing_seconds, contraction_seconds);
}

void stream_localized_triples_moments(
    const Matrix& fock, const Tensor4& eri, const Matrix& t1,
    const Tensor4& t2,
    const LocalizedTriplesMomentStreamOptions& options) {
  stream_localized_triples_moments_impl(fock, eri, t1, t2, options);
}

void stream_localized_triples_moments(
    const SpatialRhfHamiltonian& hamiltonian, const Matrix& t1,
    const Tensor4& t2,
    const LocalizedTriplesMomentStreamOptions& options) {
  const auto spin_fock =
      expand_spatial_one_body_to_spin_orbital(hamiltonian.fock);
  const SpinOrbitalEriView spin_eri(hamiltonian);
  stream_localized_triples_moments_impl(spin_fock, spin_eri, t1, t2,
                                        options);
}

void stream_localized_triples_moments(
    const SpatialRhfHamiltonian& hamiltonian, const Matrix& t1,
    const SelectedDoublesView& t2,
    const LocalizedTriplesMomentStreamOptions& options) {
  const auto spin_fock =
      expand_spatial_one_body_to_spin_orbital(hamiltonian.fock);
  const SpinOrbitalEriView spin_eri(hamiltonian);
  stream_localized_triples_moments_impl(spin_fock, spin_eri, t1, t2,
                                        options);
}

} // namespace localized_cc
