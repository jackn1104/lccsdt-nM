#pragma once

#include "modernqc/linalg/matrix.hpp"

#include <cstddef>

namespace modernqc::scf {

struct ProjectedRhfGuess {
  linalg::Matrix density;
  linalg::Matrix occupied_coefficients;
  std::size_t target_metric_rank;
  double minimum_occupied_gram_eigenvalue;
  double occupied_orthonormality_maximum_error;
  double electron_count;
};

// Project a converged occupied RHF space from a smaller source basis into a
// target basis at exactly the same molecular geometry. The rectangular
// overlap is <target AO|source AO>. The projected occupied space is
// symmetrically orthonormalized in the retained target overlap metric before
// the spin-summed density P = 2 C_occ C_occ^T is formed.
[[nodiscard]] ProjectedRhfGuess project_occupied_rhf_guess(
    const linalg::Matrix& source_coefficients,
    std::size_t occupied_orbitals,
    const linalg::Matrix& target_overlap,
    const linalg::Matrix& target_source_overlap,
    double target_linear_dependency_tolerance = 1.0e-8,
    double occupied_rank_tolerance = 1.0e-10);

}  // namespace modernqc::scf
