#include "localized_cc/laplace_triples.hpp"

#include "localized_cc/triples.hpp"
#include "laplace_triples_quadrature.hpp"

#include "modernqc/linalg/eigensolver.hpp"
#include "modernqc/linalg/matrix.hpp"
#include "modernqc/mp2/laplace_fit.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace localized_cc {
namespace {

using ModernMatrix = modernqc::linalg::Matrix;

std::size_t checked_product(const std::size_t left,
                            const std::size_t right,
                            const char* label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(std::string(label) + " size overflows");
  return left * right;
}

ModernMatrix square_block(const Matrix& source, const std::size_t first,
                          const std::size_t count) {
  const auto dimensions = source.dimensions();
  if (count == 0 || dimensions[0] != dimensions[1] ||
      first > dimensions[0] || count > dimensions[0] - first)
    throw std::invalid_argument("invalid Laplace triples Fock block");
  ModernMatrix result(count, count);
  for (std::size_t row = 0; row < count; ++row)
    for (std::size_t column = 0; column < count; ++column)
      result(row, column) = source(first + row, first + column);
  return result;
}

double maximum_block_offdiagonal(const Matrix& fock,
                                 const std::size_t occupied) {
  const auto orbitals = fock.dimensions()[0];
  double maximum = 0.0;
  for (std::size_t row = 0; row < orbitals; ++row)
    for (std::size_t column = 0; column < orbitals; ++column)
      if (row != column && ((row < occupied && column < occupied) ||
                            (row >= occupied && column >= occupied)))
        maximum = std::max(maximum, std::abs(fock(row, column)));
  return maximum;
}

Tensor6 transform_axis(const Tensor6& source, const ModernMatrix& rotation,
                       const std::size_t axis) {
  const auto dimensions = source.dimensions();
  if (axis >= dimensions.size() || rotation.rows() != dimensions[axis] ||
      rotation.columns() != dimensions[axis])
    throw std::invalid_argument("Laplace triples rotation mismatch");
  Tensor6 result(dimensions);
  for (std::size_t i = 0; i < dimensions[0]; ++i)
    for (std::size_t j = 0; j < dimensions[1]; ++j)
      for (std::size_t k = 0; k < dimensions[2]; ++k)
        for (std::size_t a = 0; a < dimensions[3]; ++a)
          for (std::size_t b = 0; b < dimensions[4]; ++b)
            for (std::size_t c = 0; c < dimensions[5]; ++c) {
              const std::array<std::size_t, 6> target{i, j, k, a, b, c};
              double value = 0.0;
              for (std::size_t source_index = 0;
                   source_index < dimensions[axis]; ++source_index) {
                auto source_indices = target;
                source_indices[axis] = source_index;
                value += source(source_indices[0], source_indices[1],
                                source_indices[2], source_indices[3],
                                source_indices[4], source_indices[5]) *
                         rotation(source_index, target[axis]);
              }
              result(i, j, k, a, b, c) = value;
            }
  return result;
}

Tensor6 transform_triples(const Tensor6& source,
                          const ModernMatrix& occupied_rotation,
                          const ModernMatrix& virtual_rotation) {
  Tensor6 transformed = source;
  for (std::size_t axis = 0; axis < 3; ++axis)
    transformed = transform_axis(transformed, occupied_rotation, axis);
  for (std::size_t axis = 3; axis < 6; ++axis)
    transformed = transform_axis(transformed, virtual_rotation, axis);
  return transformed;
}

} // namespace

LaplaceTriplesResult solve_localized_triples_laplace_dense_oracle(
    const SpatialRhfHamiltonian& localized_hamiltonian,
    const Matrix& localized_t1, const Tensor4& localized_t2,
    const LaplaceTriplesOptions& options) {
  detail::validate_laplace_triples_quadrature(
      options.quadrature_scheme, options.quadrature_points);
  if (!(options.requested_energy_error_hartree > 0.0) ||
      !std::isfinite(options.requested_energy_error_hartree) ||
      !(options.denominator_floor > 0.0) ||
      !std::isfinite(options.denominator_floor) ||
      options.validation_grid_size < 2)
    throw std::invalid_argument("invalid Laplace triples options");
  const auto spatial_occupied = localized_hamiltonian.nocc;
  const auto spatial_virtuals = localized_hamiltonian.nvirt;
  if (spatial_occupied >
          std::numeric_limits<std::size_t>::max() - spatial_virtuals ||
      spatial_occupied > std::numeric_limits<std::size_t>::max() / 2 ||
      spatial_virtuals > std::numeric_limits<std::size_t>::max() / 2)
    throw std::overflow_error("localized triples orbital count overflows");
  const auto spatial_orbitals = spatial_occupied + spatial_virtuals;
  if (spatial_occupied == 0 || spatial_virtuals == 0 ||
      localized_hamiltonian.fock.dimensions() !=
          std::array<std::size_t, 2>{spatial_orbitals, spatial_orbitals})
    throw std::invalid_argument("invalid localized triples Hamiltonian");
  const auto occupied = checked_product(2, spatial_occupied,
                                        "spin occupied orbitals");
  const auto virtuals = checked_product(2, spatial_virtuals,
                                       "spin virtual orbitals");
  if (localized_t1.dimensions() !=
          std::array<std::size_t, 2>{occupied, virtuals} ||
      localized_t2.dimensions() !=
          std::array<std::size_t, 4>{occupied, occupied, virtuals, virtuals})
    throw std::invalid_argument("Laplace triples amplitude dimensions mismatch");

  const auto tensor_elements = checked_product(
      checked_product(checked_product(occupied, occupied, "triples occupied"),
                      occupied, "triples occupied"),
      checked_product(checked_product(virtuals, virtuals, "triples virtual"),
                      virtuals, "triples virtual"),
      "triples tensor");
  const auto tensor_bytes =
      checked_product(tensor_elements, sizeof(double), "triples tensor bytes");
  const auto peak_bytes = checked_product(tensor_bytes, 4,
                                          "Laplace triples peak bytes");
  if (peak_bytes > options.maximum_additional_memory_bytes)
    throw std::runtime_error(
        "dense Laplace triples oracle exceeds the configured memory guard");

  const Matrix spin_fock =
      expand_spatial_one_body_to_spin_orbital(localized_hamiltonian.fock);
  auto moments = build_triples_moments(localized_hamiltonian, localized_t1,
                                       localized_t2);
  const auto occupied_eigen = modernqc::linalg::diagonalize_symmetric(
      square_block(spin_fock, 0, occupied));
  const auto virtual_eigen = modernqc::linalg::diagonalize_symmetric(
      square_block(spin_fock, occupied, virtuals));
  Tensor6 right = transform_triples(moments.first,
                                    occupied_eigen.eigenvectors,
                                    virtual_eigen.eigenvectors);
  Tensor6 left = transform_triples(moments.second,
                                   occupied_eigen.eigenvectors,
                                   virtual_eigen.eigenvectors);

  const auto [occupied_minimum, occupied_maximum] = std::minmax_element(
      occupied_eigen.eigenvalues.begin(), occupied_eigen.eigenvalues.end());
  const auto [virtual_minimum, virtual_maximum] = std::minmax_element(
      virtual_eigen.eigenvalues.begin(), virtual_eigen.eigenvalues.end());
  double denominator_minimum =
      3.0 * (*virtual_minimum - *occupied_maximum);
  double denominator_maximum =
      3.0 * (*virtual_maximum - *occupied_minimum);
  if (!(denominator_minimum > options.denominator_floor) ||
      !(denominator_maximum > denominator_minimum))
    throw std::runtime_error(
        "localized triples denominator interval is invalid");

  // The precomputed minimax rules cover ratios in [2,100]. Gauss-Laguerre
  // is generated for the requested interval and is instead governed by the
  // reported reciprocal-error and energy-bound checks.
  if (options.quadrature_scheme ==
      LaplaceTriplesQuadratureScheme::minimax) {
    if (denominator_maximum / denominator_minimum < 2.0)
      denominator_minimum = denominator_maximum / 2.0;
    if (denominator_maximum / denominator_minimum > 100.0)
      throw std::runtime_error(
          "triples denominator ratio exceeds the validated minimax interval");
  }

  const auto quadrature = detail::build_laplace_triples_quadrature(
      denominator_minimum, denominator_maximum, options.quadrature_scheme,
      options.quadrature_points, options.validation_grid_size);

  LaplaceTriplesResult result;
  result.denominator_minimum = denominator_minimum;
  result.denominator_maximum = denominator_maximum;
  result.localized_fock_offdiagonal =
      maximum_block_offdiagonal(spin_fock, occupied);
  result.quadrature_points = quadrature.nodes.size();
  result.nodes = quadrature.nodes;
  result.weights = quadrature.weights;
  result.peak_additional_mib =
      static_cast<double>(peak_bytes) / (1024.0 * 1024.0);

  long double approximate_energy = 0.0L;
  long double exact_energy = 0.0L;
  long double absolute_product_sum = 0.0L;
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t j = 0; j < occupied; ++j)
      for (std::size_t k = 0; k < occupied; ++k)
        for (std::size_t a = 0; a < virtuals; ++a)
          for (std::size_t b = 0; b < virtuals; ++b)
            for (std::size_t c = 0; c < virtuals; ++c) {
              const double denominator =
                  virtual_eigen.eigenvalues[a] +
                  virtual_eigen.eigenvalues[b] +
                  virtual_eigen.eigenvalues[c] -
                  occupied_eigen.eigenvalues[i] -
                  occupied_eigen.eigenvalues[j] -
                  occupied_eigen.eigenvalues[k];
              if (!(denominator > options.denominator_floor))
                throw std::runtime_error("small Laplace triples denominator");
              const double reciprocal =
                  modernqc::mp2::evaluate_laplace_reciprocal(
                      denominator, quadrature.nodes, quadrature.weights);
              const double reciprocal_error =
                  std::abs(reciprocal - 1.0 / denominator);
              result.maximum_reciprocal_error =
                  std::max(result.maximum_reciprocal_error,
                           reciprocal_error);
              const long double product =
                  static_cast<long double>(left(i, j, k, a, b, c)) *
                  static_cast<long double>(right(i, j, k, a, b, c)) / 36.0L;
              approximate_energy -= product * reciprocal;
              exact_energy -= product / denominator;
              absolute_product_sum += std::abs(product);
            }
  result.parenthetical_energy = static_cast<double>(approximate_energy);
  result.exact_resolvent_energy = static_cast<double>(exact_energy);
  result.quadrature_energy_error =
      result.parenthetical_energy - result.exact_resolvent_energy;
  result.certified_energy_error_bound = static_cast<double>(
      absolute_product_sum * result.maximum_reciprocal_error);
  if (std::abs(result.quadrature_energy_error) >
          result.certified_energy_error_bound +
              64.0 * std::numeric_limits<double>::epsilon() *
                  std::max(1.0, std::abs(result.exact_resolvent_energy)))
    throw std::runtime_error(
        "Laplace triples quadrature exceeded its certified error bound");
  if (result.certified_energy_error_bound >
      options.requested_energy_error_hartree) {
    std::ostringstream message;
    message << "Laplace triples "
            << laplace_triples_quadrature_scheme_name(
                   options.quadrature_scheme)
            << " q" << options.quadrature_points
            << " fails the requested energy bound: reciprocal_error="
            << result.maximum_reciprocal_error
            << " Eh^-1, certified_bound="
            << result.certified_energy_error_bound
            << " Eh, requested=" << options.requested_energy_error_hartree
            << " Eh, denominator_interval=[" << result.denominator_minimum
            << "," << result.denominator_maximum << "] Eh";
    throw std::runtime_error(message.str());
  }
  return result;
}

} // namespace localized_cc
