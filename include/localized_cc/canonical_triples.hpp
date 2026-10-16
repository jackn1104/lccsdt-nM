#pragma once

#include "localized_cc/frozen_core.hpp"
#include "localized_cc/tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace localized_cc {

class SelectedDoublesView;

// Canonical, frozen-core conventional (T) evaluation with explicit
// spin-orbital contractions and occupied/virtual antisymmetrization.
// Development references are recorded in notices/algorithm-references.md.
// Unlike the localized noncanonical solver, this path requires diagonal
// occupied-occupied and virtual-virtual Fock blocks and never materializes a
// complete o^3 v^3 tensor.
struct CanonicalTriplesProgress {
  const std::vector<double>& bracket_task_energies;
  const std::vector<double>& parenthetical_task_energies;
  const std::vector<std::uint8_t>& completed_tasks;
};
using CanonicalTriplesObserver =
    std::function<void(const CanonicalTriplesProgress&)>;

enum class CanonicalTriplesBackend {
  reference_cpu,
  tiled_cpu,
  cuda,
  // Bounded tensor contractions: construct the same exact
  // D and D+S tiles as the CPU oracle, but evaluate the two dense moment
  // contractions with binary64 cuBLAS before an on-device permutation and
  // energy reduction. The legacy fused CUDA kernel remains available as
  // `cuda` for controlled comparisons.
  tce_cuda,
};

[[nodiscard]] constexpr bool
canonical_triples_backend_uses_cuda(const CanonicalTriplesBackend backend) {
  return backend == CanonicalTriplesBackend::cuda ||
         backend == CanonicalTriplesBackend::tce_cuda;
}

struct CanonicalTriplesOptions {
  double fock_offdiagonal_tolerance{1.0e-10};
  double denominator_floor{1.0e-10};
  std::size_t occupied_task_grain{1};
  CanonicalTriplesBackend backend{CanonicalTriplesBackend::reference_cpu};
  // The Phase 15B tiled paths compute the parenthetical CCSD(T) correction
  // only. The reference CPU path can retain CCSD[T] for regression tests.
  bool compute_bracket{true};
  std::size_t virtual_tile_size{64};
  int cuda_device{-1};
  std::vector<double> initial_bracket_task_energies;
  std::vector<double> initial_parenthetical_task_energies;
  std::vector<std::uint8_t> completed_tasks;
  // Zero means one final checkpoint callback. Otherwise completed tasks are
  // committed in deterministic waves of this size.
  std::size_t checkpoint_interval_tasks{};
  CanonicalTriplesObserver observer;
};

struct CanonicalTriplesResult {
  double bracket_energy{};       // CCSD[T]
  double parenthetical_energy{}; // CCSD(T)
  double maximum_fock_offdiagonal{};
  std::size_t occupied_triples{};
  std::size_t virtual_triples{};
  std::size_t tasks{};
  double workspace_per_worker_mib{};
  double tile_workspace_per_task_mib{};
  CanonicalTriplesBackend backend{CanonicalTriplesBackend::reference_cpu};
  std::vector<double> bracket_task_energies;
  std::vector<double> parenthetical_task_energies;
};

struct LocalizedTriplesMomentBlock {
  std::array<std::size_t, 3> occupied_indices{};
  std::array<std::size_t, 3> virtual_offsets{};
  std::array<std::size_t, 3> virtual_dimensions{};
  // Dense tile storage with the final virtual index contiguous. Only global
  // a<b<c entries are populated. left_moment is D+S.
  std::vector<double> doubles_moment;
  std::vector<double> left_moment;
  bool used_cuda{};
  int device{-1};
  // global orbital = virtual_offsets[axis] + virtual_stride * local_index.
  std::size_t virtual_stride{1};
};
using LocalizedTriplesMomentObserver =
    std::function<void(LocalizedTriplesMomentBlock&&)>;

struct LocalizedTriplesMomentStreamOptions {
  CanonicalTriplesBackend backend{CanonicalTriplesBackend::tiled_cpu};
  std::size_t virtual_tile_size{64};
  double denominator_floor{1.0e-10};
  int cuda_device{-1};
  bool shared_pair_packing{false};
  // CPU spatial-Hamiltonian path only: split external tiles by spin before
  // packing. Retain matching occupied/virtual spin counts, all internal sums.
  bool spin_aware_construction{false};
  // Rank-local pack/contract time (seconds), excluding observer/MPI work.
  // Called once per stream, outside OpenMP, on every participating rank.
  std::function<void(double, double)> timing_observer;
  LocalizedTriplesMomentObserver observer;
  // Called outside OpenMP regions, on rank zero; counts are rank-local.
  std::function<void(std::size_t, std::size_t)> row_observer;
  // Called on every owning rank after all virtual tiles of an occupied row.
  std::function<void(const std::array<std::size_t,3>&)> completed_row_observer;
  // Optional exact symmetry representative predicate. The caller must
  // reconstruct omitted symmetry-equivalent rows before energy evaluation.
  std::function<bool(const std::array<std::size_t, 3>&)> occupied_representative;
  // Reject before packing operands or constructing either triples moment.
  // The caller owns the propagated-energy error budget and zero fill.
  std::function<bool(const std::array<std::size_t, 3>&,
                     const std::array<std::size_t, 3>&)> retain_block;
};

CanonicalTriplesResult solve_canonical_triples_streamed(
    const Matrix& fock, const Tensor4& eri, const Matrix& t1,
    const Tensor4& t2, const CanonicalTriplesOptions& options = {});
CanonicalTriplesResult solve_canonical_triples_streamed(
    const SpatialRhfHamiltonian& hamiltonian, const Matrix& t1,
    const Tensor4& t2, const CanonicalTriplesOptions& options = {});

// Build exact localized external-index D and D+S moment tiles without
// applying diagonal denominators. Occupied triples are distributed across
// MPI ranks and every virtual tile is bounded. The observer must consume or
// move each block before returning.
void stream_localized_triples_moments(
    const Matrix& fock, const Tensor4& eri, const Matrix& t1,
    const Tensor4& t2,
    const LocalizedTriplesMomentStreamOptions& options);
void stream_localized_triples_moments(
    const SpatialRhfHamiltonian& hamiltonian, const Matrix& t1,
    const Tensor4& t2,
    const LocalizedTriplesMomentStreamOptions& options);
void stream_localized_triples_moments(
    const SpatialRhfHamiltonian& hamiltonian, const Matrix& t1,
    const SelectedDoublesView& t2,
    const LocalizedTriplesMomentStreamOptions& options);

} // namespace localized_cc
