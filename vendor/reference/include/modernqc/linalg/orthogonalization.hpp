#pragma once

#include "modernqc/linalg/matrix.hpp"

#include <cstddef>
#include <vector>

namespace modernqc::linalg {

enum class OrthogonalizationMethod {
  symmetric,
  canonical,
};

struct Orthogonalization {
  Matrix transformation;
  std::vector<double> overlap_eigenvalues;
  std::size_t removed_vectors;
};

[[nodiscard]] Orthogonalization build_orthogonalizer(
    const Matrix& overlap, OrthogonalizationMethod method,
    double linear_dependency_tolerance);

}  // namespace modernqc::linalg
