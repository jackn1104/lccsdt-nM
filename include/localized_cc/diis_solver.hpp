#pragma once

#include <cstddef>
#include <optional>
#include <vector>

namespace localized_cc {

// Coefficients for the newest numerically independent suffix of a DIIS
// history. first_history is the offset of that suffix in the input Gram
// matrix. The coefficients always sum to one within floating-point error.
struct DiisCoefficientSolution {
  std::size_t first_history{};
  std::vector<double> coefficients;
};

// Solve the Pulay constrained least-squares problem from the residual Gram
// matrix. The Gram matrix is scaled before elimination, so the answer is
// invariant to a common change in residual magnitude. Ill-conditioned old
// vectors are pruned deterministically; a small relative (never absolute)
// diagonal shift is used only for the final two-vector fallback.
[[nodiscard]] std::optional<DiisCoefficientSolution>
solve_diis_coefficients(
    const std::vector<std::vector<double>>& error_products);

} // namespace localized_cc
