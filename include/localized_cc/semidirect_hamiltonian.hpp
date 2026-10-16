#pragma once

#include "localized_cc/frozen_core.hpp"

#include "modernqc/integrals/integral_provider.hpp"
#include "modernqc/linalg/matrix.hpp"
#include "modernqc/scf/rhf.hpp"

#include <cstddef>
#include <cstdint>

namespace localized_cc {

struct SemidirectTransformDiagnostics {
  std::size_t ao_count{};
  std::size_t mo_count{};
  std::size_t local_first_indices{};
  std::size_t cooperative_first_index_groups{};
  std::size_t ranks_in_first_index_group{};
  std::size_t first_indices_in_group{};
  std::size_t first_index_slice_bytes{};
  std::size_t transform_work_bytes{};
  std::size_t full_mo_eri_bytes{};
  std::size_t active_mo_eri_bytes{};
  std::size_t estimated_peak_bytes{};
  std::uint64_t unique_shell_quartets{};
  std::uint64_t evaluated_shell_quartets{};
};

struct SemidirectHamiltonianResult {
  SpatialRhfHamiltonian hamiltonian;
  SemidirectTransformDiagnostics diagnostics;
};

// Exact chemists'-notation AO-to-MO transformation without an AO N^4 tensor.
// Bounded MPI subgroups own disjoint first-MO-index slices and cooperatively
// partition their Libint2 shell quartets. Subgroup reductions complete each
// first-index slice, ranks transform disjoint members of that slice, and a
// final collective assembles the current replicated spatial-MO oracle tensor.
// The tensor is mathematically dense but stores exactly one representative of
// each eightfold chemists'-integral permutation class.
// maximum_additional_memory_bytes is a per-rank guard, not a Slurm request.
// cooperative_slice_cap_bytes separately bounds the reduced first-index
// buffer and must not exceed the per-rank guard after fixed tensors.
SemidirectHamiltonianResult build_semidirect_spatial_hamiltonian(
    const modernqc::scf::RhfResult& rhf,
    modernqc::integrals::Libint2IntegralProvider& integrals,
    const modernqc::linalg::Matrix& active_occupied,
    const modernqc::linalg::Matrix& virtuals,
    std::size_t frozen_core_count,
    std::size_t maximum_additional_memory_bytes,
    double orthonormality_tolerance = 1.0e-10,
    double reference_tolerance = 1.0e-8,
    std::size_t cooperative_slice_cap_bytes = 512U * 1024U * 1024U);

} // namespace localized_cc
