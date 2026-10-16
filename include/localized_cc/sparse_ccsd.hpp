#pragma once

#include "localized_cc/ccsd.hpp"
#include "localized_cc/checkpoint.hpp"
#include "localized_cc/selected_ccsd_manifold.hpp"
#include "localized_cc/selected_ccsd_storage.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace localized_cc {

// Baseline traffic record for one explicit sparse-CCSD phase cache fill. The
// current executor still replicates every declared source block at these
// boundaries. These counters make that cost visible before Stage A replaces
// the replication with fail-closed dependency caches.
struct SparseCcsdPhaseStatistics {
  std::size_t iteration{};
  std::string phase;
  std::size_t rank{};
  std::size_t ranks{};
  std::size_t owned_output_blocks{};
  std::size_t declared_source_blocks{};
  std::size_t locally_owned_source_blocks{};
  std::size_t unique_blocks_fetched{};
  std::size_t remote_unique_blocks_fetched{};
  std::size_t fetched_bytes{};
  std::size_t remote_fetched_bytes{};
  std::size_t cache_hits{};
  std::size_t dgemm_contractions{};
  std::size_t ga_gets_inside_kernel{};
  std::size_t replicate_calls{};
  double fetch_seconds{};
};
using SparseCcsdPhaseObserver =
    std::function<void(const SparseCcsdPhaseStatistics&)>;

struct SparseCcsdOptions {
  CcsdOptions equations;
  std::size_t maximum_phase_cache_bytes{512U * 1024U * 1024U};
  std::size_t maximum_monomers{};
  std::optional<SparseCcsdCheckpointState> restart;
  std::function<void(const SparseCcsdCheckpointState&)> observer;
  SparseCcsdPhaseObserver phase_observer;
};

struct SparseCcsdMemory {
  std::size_t declared_spin_integral_elements{};
  std::size_t selected_spatial_integral_elements{};
  std::size_t selected_spatial_integral_bytes{};
  std::size_t selected_t2_elements{};
  std::size_t selected_tau_elements{};
  std::size_t selected_wmnij_elements{};
  std::size_t selected_wmbej_elements{};
  std::size_t largest_phase_cache_bytes{};
  std::size_t global_arrays_allocated_bytes{};
};

struct SparseCcsdResult {
  double correlation_energy{};
  Matrix t1;
  std::vector<std::vector<double>> t2_blocks;
  double r1_norm{};
  double r2_norm{};
  std::vector<CcsdIteration> history;
  bool converged{};
  CcsdIterationTiming cumulative_timing;
  SparseCcsdMemory memory;
  std::vector<SparseCcsdPhaseStatistics> phase_statistics;
};

// Solve the projected localized spin-orbital CCSD equations using one global
// copy of every selected T2/tau/residual/DIIS block. Source blocks are fetched
// only at explicit phase boundaries; no GA call occurs inside a contraction
// loop. Integral access is fail-closed against the equation dependency plan
// and reconstructs antisymmetrized spin values from selected spatial
// chemists' blocks. The current producer packs those blocks from the supplied
// exact spatial Hamiltonian. Its distributed AO-direct successor must pass
// the same gate before this executor may claim pre-transformation AO-to-MO
// sparsity.
[[nodiscard]] SparseCcsdResult solve_sparse_ccsd(
    const SpatialRhfHamiltonian& hamiltonian,
    const SelectedCcsdManifoldPlan& plan,
    const SparseCcsdOptions& options);

[[nodiscard]] SparseCcsdResult solve_sparse_ccsd(
    const Matrix& spatial_fock, std::size_t spatial_occupied,
    std::size_t spatial_virtuals, const SelectedIntegralStore& integrals,
    const SelectedCcsdManifoldPlan& plan,
    const SparseCcsdOptions& options);

[[nodiscard]] SparseCcsdCheckpointState make_sparse_ccsd_checkpoint(
    const SparseCcsdResult& result,
    const std::vector<int>& occupied_monomers,
    const std::vector<int>& virtual_monomers,
    std::size_t maximum_monomers);

} // namespace localized_cc
