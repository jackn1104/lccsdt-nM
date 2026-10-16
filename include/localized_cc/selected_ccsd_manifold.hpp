#pragma once

#include "localized_cc/ccsd.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace localized_cc {

struct SelectedCcsdManifoldOptions {
  // Retain an excitation block only when the number of distinct monomer
  // labels across all external indices does not exceed this value.
  // Full-space planning is the default. Setting a finite value below the
  // number of assigned monomers is an explicit selected-model approximation.
  std::size_t maximum_monomers{std::numeric_limits<std::size_t>::max()};
  // An empty list with complete_interaction_graph=true admits every monomer
  // pair. Otherwise, a multi-monomer block is retained only when every pair
  // of participating monomers is present in this undirected graph. Self
  // edges are always implicit.
  bool complete_interaction_graph{true};
  std::vector<std::pair<std::size_t, std::size_t>> interacting_monomer_pairs;
};

struct SinglesMonomerBlock {
  std::size_t occupied_monomer{};
  std::size_t virtual_monomer{};
  std::size_t occupied_extent{};
  std::size_t virtual_extent{};
  std::size_t element_count{};
};

struct DoublesMonomerBlock {
  // Ordered labels correspond to (i,j,a,b). Keeping the order at this
  // planning boundary avoids assuming an amplitude symmetry that differs
  // between the retained spin-orbital and future spatial CCSD equations.
  std::array<std::size_t, 4> monomers{};
  std::array<std::size_t, 4> extents{};
  std::size_t monomer_span{};
  std::size_t element_count{};
};

enum class CcsdOrbitalSpace : std::uint8_t { occupied, virtual_space };

enum class CcsdDependencyUse : std::uint32_t {
  denominator = 1U << 0U,
  energy = 1U << 1U,
  fae = 1U << 2U,
  fmi = 1U << 3U,
  fme = 1U << 4U,
  wmnij = 1U << 5U,
  wmbej = 1U << 6U,
  singles_residual = 1U << 7U,
  doubles_residual = 1U << 8U,
  wabef = 1U << 9U,
};

// One monomer-block dependency in the exact projected spin-orbital CCSD
// equations. Axis order is the same as the Fock or antisymmetrized ERI call
// in src/ccsd.cpp. uses is an OR of CcsdDependencyUse values and is retained
// as provenance for diagnostics and equation-audit tests.
struct CcsdFockMonomerBlock {
  std::array<CcsdOrbitalSpace, 2> spaces{};
  std::array<std::size_t, 2> monomers{};
  std::array<std::size_t, 2> extents{};
  std::size_t element_count{};
  std::uint32_t uses{};
};

struct CcsdIntegralMonomerBlock {
  std::array<CcsdOrbitalSpace, 4> spaces{};
  std::array<std::size_t, 4> monomers{};
  std::array<std::size_t, 4> extents{};
  std::size_t element_count{};
  std::uint32_t uses{};
};

struct CcsdIntermediateDependencyCounts {
  std::size_t fae_blocks{};
  std::size_t fmi_blocks{};
  std::size_t fme_blocks{};
  std::size_t wmnij_blocks{};
  std::size_t wmbej_blocks{};
  std::size_t wabef_blocks{};
};

struct CcsdIntermediateMonomerBlock {
  std::array<CcsdOrbitalSpace, 4> spaces{};
  std::array<std::size_t, 4> monomers{};
  std::array<std::size_t, 4> extents{};
  std::size_t element_count{};
};

struct SelectedCcsdDependencyStatus {
  // Amplitudes outside the selected manifold are defined to be zero in the
  // projected model, so the retained T1/T2 amplitude set is structurally
  // closed by construction. Integral and factorized-intermediate closure
  // still require an equation-by-equation audit before execution.
  bool projected_amplitude_manifold_closed{};
  bool integral_tile_closure_complete{};
  bool factorized_intermediate_closure_complete{};

  [[nodiscard]] bool ready_for_sparse_ccsd() const noexcept {
    return projected_amplitude_manifold_closed &&
           integral_tile_closure_complete &&
           factorized_intermediate_closure_complete;
  }
};

struct SelectedCcsdManifoldPlan {
  std::size_t monomer_count{};
  std::vector<std::vector<std::size_t>> occupied_orbitals_by_monomer;
  std::vector<std::vector<std::size_t>> virtual_orbitals_by_monomer;
  std::vector<SinglesMonomerBlock> singles_blocks;
  std::vector<DoublesMonomerBlock> doubles_blocks;
  std::size_t full_singles_elements{};
  std::size_t selected_singles_elements{};
  std::size_t full_doubles_elements{};
  std::size_t selected_doubles_elements{};
  std::vector<CcsdFockMonomerBlock> required_fock_blocks;
  std::vector<CcsdIntegralMonomerBlock> required_integral_blocks;
  CcsdIntermediateDependencyCounts required_intermediate_blocks;
  std::vector<CcsdIntermediateMonomerBlock> required_wmnij_blocks;
  std::vector<CcsdIntermediateMonomerBlock> required_wmbej_blocks;
  std::vector<CcsdIntermediateMonomerBlock> required_tau_blocks;
  // W_abef is streamed, but its output-block list is retained so the
  // executor constructs only equation-required tiles.
  std::vector<CcsdIntermediateMonomerBlock> required_wabef_blocks;
  SelectedCcsdDependencyStatus dependency_status;
};

// Build the structural sparse-amplitude manifold before any T1/T2 tensor or
// localized MO integral tensor is allocated. Labels must be confident,
// nonnegative, zero-based monomer identifiers. This function is a planner;
// it does not change the production CCSD equations or claim a speedup.
SelectedCcsdManifoldPlan plan_selected_ccsd_manifold(
    const std::vector<int>& occupied_labels,
    const std::vector<int>& virtual_labels,
    const SelectedCcsdManifoldOptions& options = {});

// Materialize a binary dense projection only for numerical-oracle tests.
// The labels correspond to the actual CCSD occupied and virtual indices
// (spin orbitals for the retained production equations). The production
// sparse executor consumes the block plan instead of allocating these masks.
CcsdProjectionMask make_selected_ccsd_projection_mask(
    const std::vector<int>& occupied_labels,
    const std::vector<int>& virtual_labels,
    const SelectedCcsdManifoldOptions& options = {});

} // namespace localized_cc
