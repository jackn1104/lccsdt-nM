#pragma once

#include "localized_cc/laplace_triples.hpp"

#include <cstddef>
#include <vector>

namespace localized_cc::detail {

struct LaplaceTriplesQuadrature {
  std::vector<double> nodes;
  std::vector<double> weights;
  double maximum_reciprocal_error{};
};

void validate_laplace_triples_quadrature(
    LaplaceTriplesQuadratureScheme scheme, std::size_t points);

[[nodiscard]] LaplaceTriplesQuadrature build_laplace_triples_quadrature(
    double denominator_minimum, double denominator_maximum,
    LaplaceTriplesQuadratureScheme scheme, std::size_t points,
    std::size_t validation_grid_size);

} // namespace localized_cc::detail
