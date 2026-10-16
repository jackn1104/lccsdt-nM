#include "modernqc/linalg/orthogonalization.hpp"

#include "modernqc/linalg/eigensolver.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace modernqc::linalg {

Orthogonalization build_orthogonalizer(
    const Matrix& overlap, OrthogonalizationMethod method,
    double linear_dependency_tolerance) {
  if(!std::isfinite(linear_dependency_tolerance) ||
     linear_dependency_tolerance <= 0.0) {
    throw std::invalid_argument(
        "linear-dependency tolerance must be finite and positive");
  }
  SymmetricEigendecomposition decomposition =
      diagonalize_symmetric(overlap);

  std::size_t first_retained = 0;
  while(first_retained < decomposition.eigenvalues.size() &&
        decomposition.eigenvalues[first_retained] <=
            linear_dependency_tolerance) {
    if(decomposition.eigenvalues[first_retained] <
       -linear_dependency_tolerance) {
      throw std::runtime_error(
          "overlap matrix has a materially negative eigenvalue");
    }
    ++first_retained;
  }
  if(first_retained == decomposition.eigenvalues.size()) {
    throw std::runtime_error(
        "linear-dependency removal discarded the complete AO space");
  }
  if(method == OrthogonalizationMethod::symmetric && first_retained != 0) {
    throw std::runtime_error(
        "symmetric orthogonalization cannot remove linearly dependent vectors; "
        "use canonical orthogonalization");
  }

  const std::size_t rows = overlap.rows();
  const std::size_t retained = rows - first_retained;
  Matrix scaled_vectors{rows, retained};
  for(std::size_t retained_column = 0; retained_column < retained;
      ++retained_column) {
    const std::size_t eigenvector_column =
        first_retained + retained_column;
    const double inverse_sqrt =
        1.0 / std::sqrt(
                  decomposition.eigenvalues[eigenvector_column]);
    for(std::size_t row = 0; row < rows; ++row) {
      scaled_vectors(row, retained_column) =
          decomposition.eigenvectors(row, eigenvector_column) * inverse_sqrt;
    }
  }

  Matrix transformation;
  if(method == OrthogonalizationMethod::canonical) {
    transformation = std::move(scaled_vectors);
  } else {
    transformation =
        multiply(scaled_vectors, transpose(decomposition.eigenvectors));
    symmetrize_in_place(transformation);
  }
  return Orthogonalization{
      .transformation = std::move(transformation),
      .overlap_eigenvalues = std::move(decomposition.eigenvalues),
      .removed_vectors = first_retained,
  };
}

}  // namespace modernqc::linalg
