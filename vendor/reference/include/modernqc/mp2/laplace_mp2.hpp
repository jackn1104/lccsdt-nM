#pragma once

#include "modernqc/linalg/matrix.hpp"
#include "modernqc/mp2/canonical_mp2.hpp"
#include "modernqc/mp2/laplace_fit.hpp"

#include <cstddef>
#include <vector>

namespace modernqc::mp2 {

struct IncoreLaplaceMp2Options {
  std::size_t maximum_additional_memory_bytes{
      512U * 1024U * 1024U};
  double rotation_orthogonality_tolerance{2.0e-10};
  double minimum_positive_denominator{1.0e-10};
};

struct IncoreLaplaceMp2Result {
  double canonical_reference_energy;
  double canonical_laplace_energy;
  double localized_laplace_energy;
  bool localized_energy_computed;
  double quadrature_error;
  double localization_error;
  double minimum_denominator;
  double maximum_denominator;
  double minimum_half_scaling_exponent;
  double maximum_half_scaling_exponent;
  double maximum_half_scaling_factor_error;
  std::size_t estimated_peak_additional_bytes;
  std::size_t quadrature_points;
};

// Validation-only in-core Laplace MP2 over the Phase 8 OVOV tensor.
[[nodiscard]] IncoreLaplaceMp2Result compute_incore_laplace_mp2(
    const CanonicalMp2Result& canonical_mp2,
    const std::vector<double>& orbital_energies,
    const std::vector<std::size_t>& active_occupied_indices,
    const std::vector<std::size_t>& virtual_indices,
    const LaplaceFitResult& fit,
    const linalg::Matrix* occupied_rotation = nullptr,
    const linalg::Matrix* virtual_rotation = nullptr,
    const IncoreLaplaceMp2Options& options = {});

}  // namespace modernqc::mp2
