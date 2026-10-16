#pragma once

#include "modernqc/localization/pipek_mezey.hpp"

#include <vector>

namespace modernqc::localization::detail {

struct PipekMezeyCiahResult {
  linalg::Matrix rotation;
  std::vector<double> objective_history;
  int completed_macro_iterations{0};
  double final_objective_gain{0.0};
  double final_gradient_norm{0.0};
  int total_keyframes{0};
  int total_hessian_actions{0};
};

// Matrix-free co-iterative augmented-Hessian optimizer matching the real-
// orbital path in PySCF 2.11 pyscf.lo.pipek, pyscf.lo.boys, and
// pyscf.soscf.ciah. The input matrices are the atom-resolved orbital-pair
// populations in the canonical space.
[[nodiscard]] PipekMezeyCiahResult optimize_pipek_mezey_ciah(
    const std::vector<linalg::Matrix>& canonical_populations,
    const linalg::Matrix& initial_rotation,
    const PipekMezeyOptions& options);

}  // namespace modernqc::localization::detail
