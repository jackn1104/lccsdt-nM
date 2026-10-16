#pragma once

#include "localized_cc/canonical_triples.hpp"
#include "localized_cc/frozen_core.hpp"
#include "localized_cc/tensor.hpp"

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace localized_cc {

class SelectedDoublesView;

enum class LaplaceTriplesQuadratureScheme {
  minimax,
  scaled_gauss_laguerre,
};

[[nodiscard]] const char* laplace_triples_quadrature_scheme_name(
    LaplaceTriplesQuadratureScheme scheme);

// Phase 14E numerical oracle for the orbital-invariant Laplace resolvent used
// by localized CCSD(T). This dense implementation is intentionally guarded:
// it validates the equations and quadrature before the same contractions are
// admitted to the distributed block-sparse production path.
struct LaplaceTriplesOptions {
  std::size_t quadrature_points{7};
  LaplaceTriplesQuadratureScheme quadrature_scheme{
      LaplaceTriplesQuadratureScheme::minimax};
  double requested_energy_error_hartree{1.0e-9};
  double denominator_floor{1.0e-10};
  std::size_t validation_grid_size{4800};
  std::size_t maximum_additional_memory_bytes{
      std::numeric_limits<std::size_t>::max()};
};

struct LaplaceTriplesResult {
  double parenthetical_energy{};
  double exact_resolvent_energy{};
  double quadrature_energy_error{};
  double certified_energy_error_bound{};
  double maximum_reciprocal_error{};
  double denominator_minimum{};
  double denominator_maximum{};
  double localized_fock_offdiagonal{};
  std::size_t quadrature_points{};
  double peak_additional_mib{};
  std::vector<double> nodes;
  std::vector<double> weights;
};

struct StreamedLaplaceTriplesPoint {
  std::size_t point{};
  std::size_t points{};
  double node{};
  double weight{};
  double seconds{};
  std::vector<double> many_body_energies;
};
using StreamedLaplaceTriplesObserver =
    std::function<void(const StreamedLaplaceTriplesPoint&)>;

struct StreamedLaplaceTriplesProgress {
  const std::vector<double>& point_many_body_energies;
  const std::vector<unsigned char>& completed_points;
};
using StreamedLaplaceTriplesCheckpointObserver =
    std::function<void(const StreamedLaplaceTriplesProgress&)>;

struct StreamedLaplaceTriplesOptions {
  // Opt-in preconstruction screen, in Eh. Zero preserves the old path.
  // Bounds the error relative to the UNSCREENED SAME quadrature in total
  // and in each external-label nM class. Independent of quadrature error.
  double block_screening_budget_hartree{};
  std::function<void(const std::string&)> screening_observer;
  // Rank-zero phase messages, including work before the first point.
  std::function<void(const std::string&)> progress_observer;
  // Maximum source occupied rows per MPI propagation panel.
  std::size_t propagation_row_batch{128};
  // Disk-only: prepare one source panel per rank concurrently, then broadcast
  // panels in deterministic wave/root order. No MPI inside OpenMP workers.
  bool disk_parallel_panels{false};
  // Exact low-memory schedule: one spin sector through all points, and
  // contract bounded response tiles directly into M1--M6. No sector omitted.
  // Certified point checkpoints require all sectors and the bound to pass.
  // Optional partial sector records below are explicitly uncertified.
  bool sector_streaming{false};
  // Exact closed-shell spin reversal: evaluate sectors 0/1 with multiplicity 2.
  // Requires a full dense spin tensor with exactly spin-flip invariant entries.
  bool two_spin_sectors{false};
  // Optional durable partial-sector progress, distinct from certified points.
  // Identity must bind the saved amplitudes, Hamiltonian and geometry.
  std::string sector_checkpoint_path;
  std::string moment_checkpoint_directory;
  std::string sector_checkpoint_identity;
  std::function<void(std::size_t,std::size_t)> sector_checkpoint_observer;
  // Opt-in Linux rank-private scratch. Exact, sector-streamed, non-symmetry
  // path only. Files are temporary, not restart checkpoints. Explicit per-rank
  // disk budget required; real disk space is reserved before moment work.
  std::string scratch_directory;
  std::size_t maximum_scratch_bytes{};
  bool shared_pair_packing{false};
  // Opt-in exact external-spin tile selection; no numerical screening.
  bool spin_aware_construction{false};
  // Verified PM orbits for virtual propagation and occupied output rows.
  // Every internal source row remains in the occupied-space contraction.
  bool symmetry_propagation{false};
  std::shared_ptr<const struct PmSymmetry> pm_symmetry;
  std::size_t quadrature_points{7};
  LaplaceTriplesQuadratureScheme quadrature_scheme{
      LaplaceTriplesQuadratureScheme::minimax};
  double requested_energy_error_hartree{1.0e-9};
  double denominator_floor{1.0e-10};
  std::size_t validation_grid_size{4800};
  std::size_t virtual_tile_size{64};
  CanonicalTriplesBackend moment_backend{CanonicalTriplesBackend::cuda};
  int cuda_device{-1};
  std::size_t maximum_additional_memory_bytes{
      std::numeric_limits<std::size_t>::max()};
  // Restart data are stored point-major with six M1--M6 energies per point.
  // A completed point is never evaluated twice.
  std::vector<double> initial_point_many_body_energies;
  std::vector<unsigned char> completed_points;
  StreamedLaplaceTriplesObserver observer;
  StreamedLaplaceTriplesCheckpointObserver checkpoint_observer;
};

struct StreamedLaplaceTriplesResult {
  double block_screening_bound_hartree{};
  double block_screening_seconds{};
  std::size_t block_screening_candidates{};
  std::size_t block_screening_skipped{};
  std::size_t block_screening_plan_bytes{};
  double parenthetical_energy{};
  std::vector<double> many_body_energies;
  std::vector<std::size_t> many_body_terms;
  double partition_error{};
  double certified_quadrature_error_bound{};
  double maximum_reciprocal_error{};
  double denominator_minimum{};
  double denominator_maximum{};
  double localized_fock_offdiagonal{};
  double moment_construction_seconds{};
  // Rank-local subphase diagnostics, not additive MPI critical-path times.
  double moment_packing_seconds{};
  double moment_contraction_seconds{};
  double propagator_seconds{};
  // Rank-local nM accumulation only; included in propagator_seconds.
  double partition_accumulation_seconds{};
  double communication_seconds{};
  double disk_panel_read_seconds{}; // rank-local, not summed across ranks
  double disk_panel_transform_seconds{};
  std::size_t disk_panel_waves{};
  std::size_t symmetry_unique_moment_rows{};
  std::size_t quadrature_points{};
  std::size_t occupied_triples{};
  std::size_t spin_allowed_moment_elements{};
  std::size_t maximum_local_moment_bytes{};
  std::size_t planned_peak_additional_bytes{};
  std::size_t planned_scratch_bytes{};
  std::size_t maximum_broadcast_panel_bytes{};
  std::size_t restarted_points{};
  std::vector<double> point_many_body_energies;
  std::vector<unsigned char> completed_points;
  std::vector<double> nodes;
  std::vector<double> weights;
};

// The occupied and virtual Fock blocks are never replaced by their diagonals.
// Their matrix-exponential action is evaluated through eigendecompositions,
// which is mathematically equivalent to applying exp(+t F_oo) and
// exp(-t F_vv) in the original localized representation. The dense six-index
// moments make this function an oracle only, not a production implementation.
LaplaceTriplesResult solve_localized_triples_laplace_dense_oracle(
    const SpatialRhfHamiltonian& localized_hamiltonian,
    const Matrix& localized_t1, const Tensor4& localized_t2,
    const LaplaceTriplesOptions& options = {});

// Distributed W2-capable implementation. Only unique spin triples with equal
// occupied/virtual alpha counts are stored. Moment rows remain owned by one
// MPI rank; bounded source panels are broadcast during the occupied Fock
// propagation, so no complete moment tensor is replicated.
StreamedLaplaceTriplesResult solve_localized_triples_laplace_streamed(
    const SpatialRhfHamiltonian& localized_hamiltonian,
    const Matrix& localized_t1, const Tensor4& localized_t2,
    const std::vector<int>& occupied_monomers,
    const std::vector<int>& virtual_monomers,
    const StreamedLaplaceTriplesOptions& options = {});
StreamedLaplaceTriplesResult solve_localized_triples_laplace_streamed(
    const SpatialRhfHamiltonian& localized_hamiltonian,
    const Matrix& localized_t1, const SelectedDoublesView& localized_t2,
    const std::vector<int>& occupied_monomers,
    const std::vector<int>& virtual_monomers,
    const StreamedLaplaceTriplesOptions& options = {});

} // namespace localized_cc
