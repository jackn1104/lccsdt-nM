#pragma once

#include "localized_cc/selected_ccsd_storage.hpp"

#include "modernqc/integrals/integral_provider.hpp"
#include "modernqc/linalg/matrix.hpp"
#include "modernqc/scf/rhf.hpp"

#include <cstddef>
#include <cstdint>

namespace localized_cc {

struct SelectedSemidirectTransformDiagnostics {
  std::size_t ao_count{};
  std::size_t active_mo_count{};
  std::size_t cooperative_first_index_groups{};
  std::size_t ranks_in_first_index_group{};
  std::size_t first_indices_in_group{};
  std::size_t first_index_slice_bytes{};
  std::size_t maximum_selected_transform_work_bytes{};
  std::size_t requested_spatial_integral_blocks{};
  std::size_t symmetry_unique_spatial_integral_blocks{};
  std::size_t selected_integral_bytes{};
  std::size_t selected_integral_tuple_lookup_bytes{};
  std::size_t avoided_full_active_mo_eri_bytes{};
  std::size_t estimated_peak_bytes{};
  std::uint64_t unique_shell_quartets{};
  std::uint64_t evaluated_shell_quartets_across_groups{};
};

struct SelectedSpatialHamiltonian {
  Matrix fock;
  Matrix hcore;
  std::size_t nocc{};
  std::size_t nvirt{};
  double reference_energy{};
  SelectedIntegralStore integrals;
};

// Exact selected AO-to-localized-MO transformation. The dependency closure is
// planned before this call. Only its spatial chemists' ERI blocks are
// completed; the full active-space MO four-index tensor is never allocated.
// Every selected value is computed by one first-index owner and is then
// assembled collectively into the current fail-closed replicated store. This
// replication is an explicit admission boundary; the later phase-cache gate
// will replace it with distributed block ownership without changing values.
SelectedSpatialHamiltonian build_selected_semidirect_spatial_hamiltonian(
    const modernqc::scf::RhfResult& rhf,
    modernqc::integrals::Libint2IntegralProvider& integrals,
    const modernqc::linalg::Matrix& active_occupied,
    const modernqc::linalg::Matrix& virtuals,
    const SelectedCcsdManifoldPlan& plan,
    std::size_t maximum_additional_memory_bytes,
    SelectedSemidirectTransformDiagnostics& diagnostics,
    double orthonormality_tolerance = 1.0e-10);

} // namespace localized_cc
