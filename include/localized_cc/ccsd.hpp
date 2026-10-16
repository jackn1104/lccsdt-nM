#pragma once

#include "localized_cc/diis_error.hpp"
#include "localized_cc/frozen_core.hpp"
#include "localized_cc/spatial_ccsd.hpp"
#include "localized_cc/tensor.hpp"
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace localized_cc {

class SelectedIntegralStore;

// Dense fail-closed oracle for a selected localized CCSD manifold. The
// sparse executor must reproduce this projected model before it may replace
// dense storage. Entries are row-major binary masks over T1(i,a) and
// T2(i,j,a,b), respectively.
struct CcsdProjectionMask {
  std::vector<unsigned char> singles;
  std::vector<unsigned char> doubles;
  std::size_t selected_singles{};
  std::size_t selected_doubles{};
};

// Exact closed-shell spin manifold for interleaved alpha/beta spin orbitals.
// This is structural pruning, not an energy approximation: T1 conserves the
// electron spin and T2 conserves the number of alpha and beta electrons.
[[nodiscard]] CcsdProjectionMask make_closed_shell_spin_projection_mask(
    std::size_t spatial_occupied, std::size_t spatial_virtuals);

// Logical intersection of two masks over the same spin-orbital dimensions.
// The returned counts are recomputed from the intersected binary arrays.
[[nodiscard]] CcsdProjectionMask intersect_ccsd_projection_masks(
    const CcsdProjectionMask& first, const CcsdProjectionMask& second);

struct CcsdIteration { std::size_t iteration{}; double energy{}, delta_energy{}, r1_norm{}, r2_norm{}; };

enum class CcsdAmplitudeBackend {
  automatic,
  replicated,
  global_arrays,
};

// Convergence may be measured either with the historical unnormalized
// Euclidean residual norm or with the RMS residual over the explicitly stored
// amplitude elements. Choosing the norm changes the convergence measure,
// not the residual equations. See notices/algorithm-references.md.
enum class CcsdResidualMetric { l2, rms };

enum class CcsdWabefBackend { cpu, tiled_cpu, cuda };

// Controls only how exact, iteration-invariant ERI slices used by the tiled
// W_abef executor are supplied. Both paths evaluate the same dense CCSD
// equations. The cached path is opt-in; benchmark it for the target system.
enum class CcsdWabefPackingBackend { direct, cached_invariants };

[[nodiscard]] std::string_view
ccsd_amplitude_backend_name(CcsdAmplitudeBackend backend);
[[nodiscard]] std::string_view
ccsd_wabef_backend_name(CcsdWabefBackend backend);
[[nodiscard]] std::string_view
ccsd_wabef_packing_backend_name(CcsdWabefPackingBackend backend);
[[nodiscard]] std::string_view
ccsd_residual_metric_name(CcsdResidualMetric metric);
[[nodiscard]] double ccsd_residual_measure(double l2_norm,
                                           std::size_t element_count,
                                           CcsdResidualMetric metric);

struct CcsdIterationTiming {
  double intermediates_seconds{};
  std::size_t intermediate_mixed_pair_tasks{};
  std::size_t intermediate_blas_contractions{};
  double singles_seconds{};
  double doubles_seconds{};
  std::size_t doubles_occupied_pair_tasks{};
  std::size_t doubles_blas_contractions{};
  double wabef_seconds{};
  double wabef_build_seconds{};
  double wabef_contraction_seconds{};
  double wabef_accumulation_seconds{};
  std::size_t wabef_external_tiles{};
  std::size_t wabef_internal_panels{};
  std::size_t wabef_output_tasks{};
  std::size_t wabef_blas_contractions{};
  std::size_t wabef_factorized_cached_evaluations{};
  double residual_seconds{};
  double diis_seconds{};
  std::size_t diis_error_dot_products{};
  std::size_t diis_history_vector_accumulations{};
  double energy_seconds{};
  double iteration_seconds{};
};

struct CcsdProgress {
  std::size_t completed_iterations{};
  double correlation_energy{};
  double r1_norm{};
  double r2_norm{};
  bool converged{};
  const Matrix& t1;
  const Tensor4& t2;
  const std::vector<CcsdIteration>& history;
  CcsdAmplitudeBackend amplitude_backend{CcsdAmplitudeBackend::replicated};
  CcsdWabefBackend wabef_backend{CcsdWabefBackend::cpu};
  CcsdIterationTiming timing;
};
using CcsdObserver = std::function<void(const CcsdProgress&)>;

struct CcsdIterationReport {
  std::size_t iteration{};
  double correlation_energy{};
  double delta_energy{};
  double r1_norm{};
  double r2_norm{};
  double r1_convergence_measure{};
  double r2_convergence_measure{};
  bool converged{};
  CcsdAmplitudeBackend amplitude_backend{CcsdAmplitudeBackend::replicated};
  CcsdWabefBackend wabef_backend{CcsdWabefBackend::cpu};
  CcsdIterationTiming timing;
};
using CcsdIterationObserver =
    std::function<void(const CcsdIterationReport&)>;

struct SpatialRccsdProgress {
  std::size_t completed_iterations{};
  double correlation_energy{};
  double r1_norm{};
  double r2_norm{};
  bool converged{};
  const SpatialRccsdAmplitudes& amplitudes;
  const std::vector<CcsdIteration>& history;
};
using SpatialRccsdObserver =
    std::function<void(const SpatialRccsdProgress&)>;

struct CcsdOptions {
  double energy_tolerance{1.0e-10};
  double residual_tolerance{1.0e-8};
  CcsdResidualMetric residual_metric{CcsdResidualMetric::l2};
  double denominator_floor{1.0e-10};
  std::size_t max_iterations{100};
  std::size_t diis_start{2};
  std::size_t diis_size{8};
  // Opt-in spatial solver acceleration only; raw residual stopping tests
  // and the CCSD equations are independent of this error-vector metric.
  SpatialDiisMetric spatial_diis_metric{SpatialDiisMetric::raw};
  // Exact replicated-DIIS execution default. The incremental path stores the
  // residual Gram matrix and computes only the newest row/column before
  // calling the unchanged Pulay coefficient solver. Set false to retain the
  // recomputed molecular oracle.
  bool use_incremental_diis_gram{true};
  // The replicated solver normally evaluates the identical DIIS problem on
  // every MPI rank. This opt-in owner-computes path evaluates it on rank zero
  // and broadcasts only the extrapolated amplitude vector. It changes
  // execution ownership, not the Pulay equations or amplitude space.
  bool use_root_replicated_diis{false};
  // The replicated solver historically recomputed the complete CCSD energy
  // on every MPI rank. This opt-in owner-computes path distributes the exact
  // scalar energy terms across MPI ranks and rank-local OpenMP workers, then
  // reduces one scalar. It changes only execution ownership and accumulation
  // order.
  bool use_distributed_replicated_energy{false};
  double damping{1.0};
  // A restart begins from both amplitudes or neither. DIIS vectors are
  // deliberately rebuilt because they are an accelerator, not solution
  // state. max_iterations is the total iteration limit including the saved
  // iterations.
  std::optional<Matrix> initial_t1;
  std::optional<Tensor4> initial_t2;
  std::size_t completed_iterations{};
  std::vector<CcsdIteration> initial_history;
  // This is an equation oracle, not the production sparse backend. When
  // present, amplitudes and residuals outside the mask are fixed to zero at
  // initialization, every fixed-point update, and every DIIS extrapolation.
  // Global Arrays execution is rejected until its block-sparse counterpart
  // passes equivalence tests.
  std::shared_ptr<const CcsdProjectionMask> projection_mask;
  std::size_t checkpoint_interval{1};
  // For the GA amplitude solver, keep the large T2 vector and error portions
  // of DIIS history distributed. The replicated amplitude solver always uses
  // local DIIS so small systems do not pay GA communication overhead.
  bool use_global_arrays_diis{true};
  // Automatic mode uses the replicated exact solver when the complete spin
  // T2 tensor is small enough that fine-grained GA traffic would dominate.
  // Larger tensors retain authoritative GA pair ownership. The threshold is
  // the bytes in one T2 tensor, not the solver's total working set. A zero
  // threshold therefore forces Global Arrays whenever it is available.
  CcsdAmplitudeBackend amplitude_backend{CcsdAmplitudeBackend::automatic};
  std::size_t replicated_t2_max_bytes{64U * 1024U * 1024U};
  // Maximum rank-local W_abef tile payload used by the distributed solver.
  // The implementation fetches each tau occupied-pair block only twice per
  // tile, independent of the number of virtual pairs in the tile.
  std::size_t wabef_tile_max_bytes{64U * 1024U * 1024U};
  CcsdWabefBackend wabef_backend{CcsdWabefBackend::cpu};
  CcsdWabefPackingBackend wabef_packing_backend{
      CcsdWabefPackingBackend::direct};
  // Exact external virtual-pair antisymmetry: compute a<b once and restore
  // the (b,a) contribution with opposite sign. This option is restricted
  // to unprojected, replicated, cached tiled-CPU execution.
  bool use_wabef_pair_symmetry{false};
  // Verified real signed PM orbital group. Full-space spatial solver only;
  // reuses external W_abef tasks, without restricting internal sums.
  std::shared_ptr<const struct PmSymmetry> pm_symmetry;
  // Opt-in external-index orbits in native spatial R1/R2 and intermediates.
  // Internal contractions and the full amplitude manifold are unchanged.
  bool use_pm_residual_orbits{false};
  // Per-rank cap for the exact bare-VVVV, OVVV, and OOVV slices cached by the
  // tiled W_abef packing backend. The cached backend fails before allocating
  // if the complete required invariant set does not fit this bound.
  std::size_t wabef_invariant_cache_max_bytes{512U * 1024U * 1024U};
  // Opt-in owner-computes BLAS executor for the dominant mixed W_mbej
  // intermediate. The scalar implementation remains the equation oracle.
  bool use_tiled_intermediates{false};
  // Opt-in occupied-pair/BLAS executor for the non-W_abef doubles terms.
  // The scalar path is the default equation reference for validation.
  bool use_tiled_doubles{false};
  // Exact non-VVVV spatial panels used only by SelectedIntegralStore in the
  // tiled intermediate and doubles phases. A zero value or an insufficient
  // cap disables the optimization and retains fail-closed scalar access.
  std::size_t selected_eri_panel_cache_max_bytes{512U * 1024U * 1024U};
  // Spatial reference solver only: replace the spin-expanded singles phase
  // with the exact owner-computes contraction over spatial T1/T2. Direct R2
  // remains a separate opt-in equation-oracle flag below.
  bool use_direct_spatial_singles{false};
  // Small-system spatial equation oracle: evaluate every independent R2
  // element from spatial amplitudes with cyclic MPI ownership and no
  // spin-expanded amplitudes or four-index spin intermediates. This path is
  // a reference kernel with on-demand intermediate evaluation, not a tiled
  // production kernel.
  bool use_direct_spatial_doubles{false};
  // Opt-in exact spatial pair executor: cache SU(2) intermediate channels and
  // use bounded DGEMM panels, without expanding iterative T1/T2 into spin
  // storage. Requires direct spatial singles/doubles and residual validation
  // against the reference executor before production use.
  bool use_tiled_spatial_pairs{false};
  // Opt-in non-PM-orbit kernel: analytically spin-summed non-pair R2 and
  // DGEMM mixed-pair intermediates. Same full equations; requires molecular
  // residual validation before production use.
  bool use_compact_spatial_doubles{false};
  // Compare first and converged molecular residuals with the old native path
  // at identical amplitudes; excludes oracle time from kernel subphase timers.
  bool audit_compact_spatial_doubles{false};
  // Lightweight rank-safe reporting on every iteration. Unlike observer,
  // this callback never forces a distributed T2 replication.
  CcsdIterationObserver iteration_observer;
  // Closed-shell spatial checkpoint boundary. The iterative state remains in
  // pair-symmetric spatial storage; expansion is a caller-side I/O choice.
  SpatialRccsdObserver spatial_observer;
  CcsdObserver observer;
};
struct CcsdDiisMemory {
  bool global_arrays{};
  std::size_t history_entries{};
  std::size_t doubles_history_global_bytes{};
  std::size_t doubles_history_local_bytes{};
  std::size_t doubles_history_replicated_bytes_per_rank{};
  std::size_t singles_history_bytes_per_rank{};
};
struct CcsdDistributedMemory {
  bool global_arrays{};
  std::size_t solver_pair_arrays{};
  std::size_t solver_pair_global_bytes{};
  std::size_t solver_pair_local_bytes{};
  std::size_t solver_pair_replicated_bytes_per_rank{};
  // Tau, tilde-tau, and W_mbej use the same occupied-pair/virtual-pair GA
  // layout as T2. No T2-sized scientific intermediate is replicated.
  std::size_t intermediate_pair_arrays{};
  std::size_t intermediate_pair_global_bytes{};
  std::size_t intermediate_pair_local_bytes{};
  std::size_t intermediate_pair_replicated_bytes_per_rank{};
  // W_mnij uses external occupied pair (i,j) with an occupied (m,n) block.
  std::size_t occupied_intermediate_pair_arrays{};
  std::size_t occupied_intermediate_pair_global_bytes{};
  std::size_t occupied_intermediate_pair_local_bytes{};
  std::size_t occupied_intermediate_pair_replicated_bytes_per_rank{};
  // Maximum explicitly retained occupied-pair cache in the distributed fixed
  // point (payload only). GA calls occur on the calling thread before OpenMP
  // contractions.
  std::size_t bounded_pair_cache_bytes_per_rank{};
  // Remaining replicated small intermediates: F_ae, F_mi, and F_me.
  std::size_t replicated_small_intermediate_bytes_per_rank{};
  // Rank-local streamed W_abef contributions, distributed by external (a,b).
  std::size_t streamed_wabef_accumulator_bytes_per_rank{};
};
struct CcsdResult {
  double correlation_energy{};
  Matrix t1;
  Tensor4 t2;
  double r1_norm{}, r2_norm{};
  std::vector<CcsdIteration> history;
  bool converged{};
  CcsdAmplitudeBackend amplitude_backend{CcsdAmplitudeBackend::replicated};
  CcsdWabefBackend wabef_backend{CcsdWabefBackend::cpu};
  CcsdWabefPackingBackend wabef_packing_backend{
      CcsdWabefPackingBackend::direct};
  std::size_t wabef_invariant_cache_bytes{};
  double wabef_invariant_cache_build_seconds{};
  std::size_t spatial_residual_workspace_bytes{};
  double spatial_residual_workspace_build_seconds{};
  bool selected_eri_panel_cache_enabled{};
  std::size_t selected_eri_panel_cache_required_bytes{};
  std::size_t selected_eri_panel_cache_bytes{};
  std::size_t selected_eri_panel_cache_populated_elements{};
  double selected_eri_panel_cache_build_seconds{};
  CcsdIterationTiming cumulative_timing;
  CcsdDiisMemory diis_memory;
  CcsdDistributedMemory distributed_memory;
};

// One exact evaluation of the spin-orbital CCSD residual at caller-supplied
// amplitudes. This equation-level validation entry point performs no damping,
// DIIS, convergence test, or amplitude update. It is used to compare optimized
// contraction executors against the retained scalar dense oracle.
struct CcsdResidualEvaluation {
  Matrix r1;
  Tensor4 r2;
  double r1_norm{};
  double r2_norm{};
  CcsdIterationTiming timing;
};

// Exact closed-shell spatial residual extracted from the retained
// spin-orbital equation oracle. Optimized spatial contractions must reproduce
// every element within the validation tolerance before replacing this reference.
struct SpatialRccsdResidualEvaluation {
  Matrix r1;
  Tensor4 r2;
  double r1_norm{};
  double r2_norm{};
  CcsdIterationTiming timing;
};

// Exact owner-computes spatial-singles phase. The implementation contracts
// directly from the spatial Hamiltonian and pair-symmetric amplitudes. It
// does not allocate spin-expanded T1 or T2. Small Fae/Fmi/Fme intermediates
// are assigned cyclically to MPI ranks and reduced once before the independent
// alpha singles residual elements are assigned and reduced in the same way.
struct SpatialRccsdSinglesEvaluation {
  Matrix r1;
  double r1_norm{};
  CcsdIterationTiming timing;
  std::size_t global_intermediate_tasks{};
  std::size_t local_intermediate_tasks{};
  std::size_t global_output_tasks{};
  std::size_t local_output_tasks{};
  std::size_t replicated_intermediate_bytes_per_rank{};
  std::size_t avoided_spin_amplitude_bytes_per_rank{};
};

struct SpatialRccsdDoublesEvaluation {
  Tensor4 r2;
  double r2_norm{};
  CcsdIterationTiming timing;
  std::size_t global_output_tasks{};
  std::size_t local_output_tasks{};
  std::size_t replicated_small_intermediate_bytes_per_rank{};
  std::size_t avoided_spin_amplitude_bytes_per_rank{};
  std::size_t avoided_spin_four_index_intermediate_bytes_per_rank{};
};

struct SpatialRccsdResult {
  double correlation_energy{};
  SpatialRccsdAmplitudes amplitudes;
  double r1_norm{};
  double r2_norm{};
  std::vector<CcsdIteration> history;
  bool converged{};
  // Exact spin-orbital residual-oracle workspace retained for the lifetime of
  // the spatial solve. This includes the expanded Fock matrix and fixed
  // denominators, but not the iteration-dependent expanded amplitudes.
  std::size_t residual_workspace_bytes{};
  double residual_workspace_build_seconds{};
  std::size_t wabef_invariant_cache_bytes{};
  double wabef_invariant_cache_build_seconds{};
  CcsdIterationTiming cumulative_timing;
};

[[nodiscard]] SpatialRccsdResidualEvaluation evaluate_spatial_rccsd_residual(
    const SpatialRhfHamiltonian& hamiltonian,
    const SpatialRccsdAmplitudes& amplitudes,
    const CcsdOptions& options = {});

[[nodiscard]] SpatialRccsdSinglesEvaluation
evaluate_spatial_rccsd_singles_direct(
    const SpatialRhfHamiltonian& hamiltonian,
    const SpatialRccsdAmplitudes& amplitudes,
    const CcsdOptions& options = {});

[[nodiscard]] SpatialRccsdDoublesEvaluation
evaluate_spatial_rccsd_doubles_direct(
    const SpatialRhfHamiltonian& hamiltonian,
    const SpatialRccsdAmplitudes& amplitudes,
    const CcsdOptions& options = {});

// Exact reference solver with spatial T1/T2 and DIIS storage. Residual options
// include the retained spin-orbital reference, direct spatial R1/R2 references,
// and opt-in tiled spatial contractions.
[[nodiscard]] SpatialRccsdResult solve_spatial_rccsd_reference(
    const SpatialRhfHamiltonian& hamiltonian,
    const CcsdOptions& options = {});

[[nodiscard]] CcsdResidualEvaluation evaluate_ccsd_residual(
    const Matrix& fock, const Tensor4& eri, const Matrix& t1,
    const Tensor4& t2, std::size_t nocc,
    const CcsdOptions& options = {});

CcsdResult solve_ccsd(const Matrix& fock, const Tensor4& eri, std::size_t nocc,
                      const CcsdOptions& options = {});
CcsdResult solve_ccsd(const SpatialRhfHamiltonian& hamiltonian,
                      const CcsdOptions& options = {});
CcsdResult solve_ccsd(const Matrix& spatial_fock,
                      std::size_t spatial_occupied,
                      std::size_t spatial_virtuals,
                      const SelectedIntegralStore& integrals,
                      const CcsdOptions& options = {});
double ccsd_energy(const Matrix& fock, const Tensor4& eri, const Matrix& t1,
                   const Tensor4& t2, std::size_t nocc);
double t2_antisymmetry_error(const Tensor4& t2);

} // namespace localized_cc
