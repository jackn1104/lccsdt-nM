#pragma once

#include "localized_cc/frozen_core.hpp"

#include "modernqc/linalg/matrix.hpp"

#include <cstddef>

namespace localized_cc {

struct SpatialHamiltonianRotationDiagnostics {
  std::size_t orbitals{};
  std::size_t logical_eri_bytes{};
  std::size_t stored_eri_bytes{};
  std::size_t estimated_peak_additional_bytes{};
  double wall_seconds{};
};

struct SpatialHamiltonianRotationResult {
  SpatialRhfHamiltonian hamiltonian;
  SpatialHamiltonianRotationDiagnostics diagnostics;
};

// Apply independent occupied and virtual unitary rotations to an exact
// spatial Hamiltonian.  The source is consumed so its compressed ERI storage
// can be released before the two dense four-pass work buffers coexist.  The
// operation changes representation only; no integral or excitation is
// screened.  maximum_additional_memory_bytes is a per-process guard.
SpatialHamiltonianRotationResult rotate_spatial_hamiltonian(
    SpatialRhfHamiltonian source,
    const modernqc::linalg::Matrix& occupied_rotation,
    const modernqc::linalg::Matrix& virtual_rotation,
    std::size_t maximum_additional_memory_bytes);

} // namespace localized_cc
