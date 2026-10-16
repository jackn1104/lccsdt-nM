#pragma once

#include "modernqc/integrals/integral_provider.hpp"
#include "modernqc/linalg/matrix.hpp"
#include "modernqc/mp2/laplace_fit.hpp"
#include "modernqc/mp2/nm_decomposition.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace modernqc::mp2 {

enum class LocalizedLaplaceMp2Backend {
  automatic,
  direct_tiled,
  // Experimental comparison backend for 1M+2M only. Monomer assignments
  // define the retained localized orbital blocks before any OVOV tensor is
  // formed. Each rank evaluates AO shell quartets once per Laplace point for
  // its occupied slice, reuses the exact two-index-transformed intermediate
  // to complete selected blocks, and then discards those blocks. No full
  // canonical or localized four-index OVOV tensor is allocated.
  localized_direct_selected,
  // Exact comparison backend for full 1M--4M or truncated 1M+2M energies.
  // The unscaled four-index OVOV tensor is built directly in the fixed
  // localized orbital basis and distributed exactly once. At every Laplace
  // point dense localized-basis propagators U^T exp(+/- epsilon t/2) U
  // recover the canonical denominator without constructing or storing a
  // canonical OVOV tensor.
  localized_cached_ovov,
  cached_canonical_ovov,
  shared_selected_ovov,
  replicated_selected_ovov,
  distributed_selected_ovov,
};

struct LocalizedTiledLaplaceMp2Options {
  std::size_t occupied_tile_size{4};
  std::size_t virtual_tile_size{16};
  // Number of Laplace points sharing one AO shell-quartet traversal in the
  // exact localized-direct-selected backend.
  std::size_t quadrature_batch_size{1};
  // Target lambda-AO extent of its first-index streaming block. Shells are
  // never split, so the realized extent can be slightly larger.
  std::size_t lambda_ao_block_size{16};
  std::size_t maximum_additional_memory_bytes{
      512U * 1024U * 1024U};
  std::size_t threads{1};
  std::size_t mpi_rank{0};
  std::size_t mpi_ranks{1};
  double eri_schwarz_threshold{0.0};
  // Complete quadrature-summed upper bound, in hartree, below which a
  // localized excitation block may be omitted before its direct/exchange
  // integrals are formed. Zero disables localized pre-energy screening.
  double localized_energy_schwarz_threshold_hartree{0.0};
  // Per-quadrature-point upper bound, in hartree, below which an ordered
  // localized ijab contribution may be omitted before its exact left-pair
  // contraction and exchange lookup. Zero disables pointwise screening.
  // The cumulative reported omitted bound, rather than this local cutoff,
  // controls the rigorous error guarantee.
  double localized_point_energy_threshold_hartree{0.0};
  bool report_laplace_point_progress{false};
  double assignment_confidence_threshold{0.8};
  NmExecutionMode execution_mode{NmExecutionMode::full};
  LocalizedLaplaceMp2Backend backend{
      LocalizedLaplaceMp2Backend::automatic};
};

struct LocalizedTiledLaplaceMp2Result {
  NmEnergyDecomposition decomposition;
  NmExecutionMode execution_mode;
  LocalizedLaplaceMp2Backend backend;
  double correlation_energy_hartree;
  std::uint64_t computed_excitation_terms;
  std::uint64_t omitted_excitation_terms;
  std::uint64_t computed_point_tasks;
  std::uint64_t omitted_point_tasks;
  std::uint64_t direct_transform_count;
  std::uint64_t exchange_transform_count;
  std::uint64_t reused_exchange_tile_count;
  std::uint64_t evaluated_shell_quartets;
  std::uint64_t schwarz_screened_shell_quartets;
  std::uint64_t localized_screened_excitation_terms;
  std::uint64_t localized_screened_block_tasks;
  double localized_screening_omitted_energy_bound_hartree;
  std::uint64_t localized_screened_point_contributions;
  double localized_point_screening_omitted_energy_bound_hartree;
  std::size_t low_confidence_occupied_orbitals;
  std::size_t low_confidence_virtual_orbitals;
  std::size_t estimated_peak_additional_bytes;
  // Global canonical OVOV size and maximum rank-local distributed slice.
  std::size_t canonical_ovov_bytes;
  std::size_t canonical_ovov_max_local_bytes;
  // Sum of physically stored canonical OVOV bytes over all MPI ranks. For
  // exact distributed ownership this equals canonical_ovov_bytes; a value
  // larger by mpi_ranks exposes replicated storage directly.
  std::size_t canonical_ovov_stored_bytes_global;
  // Global localized OVOV size, maximum rank-local slice, and sum of the
  // physically stored slices. These fields are nonzero only for the exact
  // localized-cached backend and expose accidental replication explicitly.
  std::size_t localized_ovov_bytes;
  std::size_t localized_ovov_max_local_bytes;
  std::size_t localized_ovov_stored_bytes_global;
  double canonical_ovov_build_seconds;
  double canonical_ovov_broadcast_seconds;
  double localized_ovov_build_seconds;
  // Critical-rank wall time spent in direct AO-shell-to-localized-OVOV tile
  // transformations. This is zero for canonical-cache backends.
  double direct_ovov_transform_seconds;
  double localized_rotation_seconds;
  double pair_transpose_seconds;
  double energy_accumulation_seconds;
  double distributed_communication_seconds;
  double localized_screening_seconds;
};

// Exact Wilson--Almlof half scaling is applied either in the canonical basis
// before orbital rotation or through its algebraically equivalent localized
// propagators. Localized orbitals are grouped by their hard monomer labels so
// 1M2M mode can avoid accumulating 3M/4M excitation terms. Full mode retains
// and reports every 1M--4M contribution.
[[nodiscard]] LocalizedTiledLaplaceMp2Result
compute_localized_tiled_laplace_mp2(
    const integrals::Libint2IntegralProvider& provider,
    const linalg::Matrix& canonical_coefficients,
    const std::vector<double>& orbital_energies,
    const std::vector<std::size_t>& active_occupied_indices,
    const std::vector<std::size_t>& virtual_indices,
    const linalg::Matrix& occupied_rotation,
    const linalg::Matrix& virtual_rotation,
    const std::vector<MonomerAssignment>& occupied_assignments,
    const std::vector<MonomerAssignment>& virtual_assignments,
    const LaplaceFitResult& fit,
    const LocalizedTiledLaplaceMp2Options& options = {});

}  // namespace modernqc::mp2
