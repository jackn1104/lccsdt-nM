#include "laplace_triples_quadrature.hpp"

#include "modernqc/mp2/laplace_fit.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace localized_cc {
namespace {

modernqc::mp2::DenominatorHistogram interval_histogram(
    const double minimum, const double maximum, const std::size_t bins) {
  if (!(minimum > 0.0) || !(maximum > minimum) || bins < 2)
    throw std::invalid_argument("invalid triples denominator interval");
  modernqc::mp2::DenominatorHistogram result;
  result.denominator_minimum = minimum;
  result.denominator_maximum = maximum;
  result.total_ordered_denominators = static_cast<std::uint64_t>(bins);
  result.edges.resize(bins + 1);
  result.midpoints.resize(bins);
  result.counts.assign(bins, 1U);
  result.frequencies.assign(bins, 1.0 / static_cast<double>(bins));
  const auto logarithmic_minimum = std::log(minimum);
  const auto width = (std::log(maximum) - logarithmic_minimum) /
                     static_cast<double>(bins);
  for (std::size_t bin = 0; bin <= bins; ++bin)
    result.edges[bin] =
        std::exp(logarithmic_minimum + width * static_cast<double>(bin));
  result.edges.front() = minimum;
  result.edges.back() = maximum;
  for (std::size_t bin = 0; bin < bins; ++bin)
    result.midpoints[bin] = std::exp(
        logarithmic_minimum + width * (static_cast<double>(bin) + 0.5));
  return result;
}

} // namespace

const char* laplace_triples_quadrature_scheme_name(
    const LaplaceTriplesQuadratureScheme scheme) {
  if (scheme == LaplaceTriplesQuadratureScheme::minimax) return "minimax";
  if (scheme == LaplaceTriplesQuadratureScheme::scaled_gauss_laguerre)
    return "scaled-gauss-laguerre";
  throw std::invalid_argument("unknown Laplace triples quadrature scheme");
}

namespace detail {

void validate_laplace_triples_quadrature(
    const LaplaceTriplesQuadratureScheme scheme, const std::size_t points) {
  if (scheme == LaplaceTriplesQuadratureScheme::minimax) {
    if (points < 5 || points > 7)
      throw std::invalid_argument(
          "Laplace triples minimax quadrature supports 5, 6, or 7 points");
    return;
  }
  if (scheme == LaplaceTriplesQuadratureScheme::scaled_gauss_laguerre) {
    if (points < 8 || points > 64)
      throw std::invalid_argument(
          "Laplace triples scaled Gauss-Laguerre quadrature supports 8 "
          "through 64 points");
    return;
  }
  throw std::invalid_argument("unknown Laplace triples quadrature scheme");
}

LaplaceTriplesQuadrature build_laplace_triples_quadrature(
    const double denominator_minimum, const double denominator_maximum,
    const LaplaceTriplesQuadratureScheme scheme, const std::size_t points,
    const std::size_t validation_grid_size) {
  validate_laplace_triples_quadrature(scheme, points);
  if (!(denominator_minimum > 0.0) ||
      !(denominator_maximum > denominator_minimum) ||
      validation_grid_size < 2)
    throw std::invalid_argument("invalid triples quadrature interval");

  modernqc::mp2::LaplaceFitOptions options;
  options.number_of_points = points;
  options.validation_grid_size = validation_grid_size;
  options.threads = 1;

  double denominator_scale = 1.0;
  double fit_minimum = denominator_minimum;
  double fit_maximum = denominator_maximum;
  if (scheme == LaplaceTriplesQuadratureScheme::minimax) {
    options.scheme = modernqc::mp2::LaplaceQuadratureScheme::minimax;
  } else {
    // Standard Gauss-Laguerre is most accurate when the smallest normalized
    // denominator is one. For x = scale*y, rescale both nodes and weights by
    // scale so sum_k w_k exp(-t_k*x) continues to approximate 1/x.
    options.scheme = modernqc::mp2::LaplaceQuadratureScheme::gauss_laguerre;
    denominator_scale = denominator_minimum;
    fit_minimum = 1.0;
    fit_maximum = denominator_maximum / denominator_scale;
  }

  const auto fit = modernqc::mp2::fit_laplace_quadrature(
      interval_histogram(fit_minimum, fit_maximum, options.histogram_bins),
      options);
  LaplaceTriplesQuadrature result;
  result.nodes = fit.nodes;
  result.weights = fit.weights;
  for (auto& node : result.nodes) node /= denominator_scale;
  for (auto& weight : result.weights) weight /= denominator_scale;
  result.maximum_reciprocal_error =
      fit.validation_maximum_absolute_error / denominator_scale;
  return result;
}

} // namespace detail
} // namespace localized_cc
