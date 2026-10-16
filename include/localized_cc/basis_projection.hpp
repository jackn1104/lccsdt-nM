#pragma once

#include "localized_cc/tensor.hpp"
#include "modernqc/linalg/matrix.hpp"

#include <cstddef>

namespace localized_cc {

// Numerical controls for same-geometry basis progression. Eigenvalues are
// tested relative to the largest eigenvalue of the corresponding metric.
struct BasisProjectionOptions {
  double rank_relative_tolerance{1.0e-10};
  double orthonormality_tolerance{1.0e-8};
};

struct BasisProjectionDiagnostics {
  std::size_t source_ao_count{};
  std::size_t target_ao_count{};
  std::size_t source_orbital_count{};
  std::size_t target_metric_rank{};
  double source_orthonormality_error{};
  double minimum_projection_gram_eigenvalue{};
  double projected_orthonormality_error{};
  double spin_summed_electron_count{};
};

struct ProjectedRhfGuess {
  modernqc::linalg::Matrix occupied_coefficients;
  modernqc::linalg::Matrix spin_summed_density;
  BasisProjectionDiagnostics diagnostics;
};

// S_target^+ S_target,source C_source followed by symmetric metric
// reorthonormalization. The source and target must describe the same geometry.
[[nodiscard]] ProjectedRhfGuess project_rhf_occupied_guess(
    const modernqc::linalg::Matrix& source_overlap,
    const modernqc::linalg::Matrix& source_occupied_coefficients,
    const modernqc::linalg::Matrix& target_overlap,
    const modernqc::linalg::Matrix& target_source_overlap,
    const BasisProjectionOptions& options = {});

struct OrbitalRotationProjection {
  // target-orbital by source-orbital semi-orthogonal transformation.
  modernqc::linalg::Matrix source_to_target;
  // Square target-space rotation. Its leading columns are source_to_target;
  // the remaining columns are a deterministic orthogonal complement.
  modernqc::linalg::Matrix initial_rotation;
  double minimum_projection_gram_eigenvalue{};
  double orthogonality_error{};
};

// Projects a source orbital space into an already orthonormal target orbital
// space. target_reference is AO x target_orbitals and source_coefficients is
// source-AO x source_orbitals.
[[nodiscard]] OrbitalRotationProjection project_orbital_rotation(
    const modernqc::linalg::Matrix& target_reference,
    const modernqc::linalg::Matrix& source_coefficients,
    const modernqc::linalg::Matrix& target_source_overlap,
    const BasisProjectionOptions& options = {});

struct ProjectedCcsdGuess {
  Matrix t1;
  Tensor4 t2;
  double t2_antisymmetry_error{};
};

// The source CCSD amplitudes use interleaved alpha/beta spin orbitals. The
// supplied spatial transformations are target-by-source and do not mix spin.
// New target virtual-complement amplitudes are zero except for contributions
// obtained by the rectangular source-to-target transformation.
[[nodiscard]] ProjectedCcsdGuess project_ccsd_amplitudes(
    const Matrix& source_t1, const Tensor4& source_t2,
    const modernqc::linalg::Matrix& occupied_source_to_target,
    const modernqc::linalg::Matrix& virtual_source_to_target);

} // namespace localized_cc
