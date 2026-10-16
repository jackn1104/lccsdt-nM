#include "localized_cc/basis_projection.hpp"

#include "modernqc/linalg/eigensolver.hpp"
#include "modernqc/linalg/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace localized_cc {
namespace {

using ModernMatrix = modernqc::linalg::Matrix;

std::size_t checked_twice(const std::size_t value, const char* description) {
  if (value > std::numeric_limits<std::size_t>::max() / 2)
    throw std::overflow_error(std::string(description) + " size overflow");
  return 2 * value;
}

std::size_t checked_product(const std::size_t left,
                            const std::size_t right,
                            const char* description) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(std::string(description) + " size overflow");
  return left * right;
}

void validate_options(const BasisProjectionOptions& options) {
  if (!std::isfinite(options.rank_relative_tolerance) ||
      options.rank_relative_tolerance <= 0.0 ||
      options.rank_relative_tolerance >= 1.0 ||
      !std::isfinite(options.orthonormality_tolerance) ||
      options.orthonormality_tolerance <= 0.0)
    throw std::invalid_argument("basis-projection tolerances are invalid");
}

double identity_error(const ModernMatrix& matrix) {
  if (matrix.rows() != matrix.columns() || matrix.empty())
    throw std::invalid_argument("identity check requires a square matrix");
  return modernqc::linalg::maximum_absolute_value(
      modernqc::linalg::subtract(
          matrix, ModernMatrix::identity(matrix.rows())));
}

ModernMatrix metric_gram(const ModernMatrix& coefficients,
                         const ModernMatrix& overlap) {
  return modernqc::linalg::multiply(
      modernqc::linalg::transpose(coefficients),
      modernqc::linalg::multiply(overlap, coefficients));
}

struct InverseSqrtResult {
  ModernMatrix matrix;
  double minimum_eigenvalue{};
};

InverseSqrtResult inverse_symmetric_square_root(
    const ModernMatrix& matrix, const BasisProjectionOptions& options,
    const char* description) {
  const auto eigen = modernqc::linalg::diagonalize_symmetric(matrix);
  const double largest = eigen.eigenvalues.back();
  const double minimum = eigen.eigenvalues.front();
  if (!std::isfinite(largest) || largest <= 0.0 ||
      minimum <= options.rank_relative_tolerance * largest)
    throw std::runtime_error(std::string(description) +
                             " is rank deficient at the requested tolerance");
  ModernMatrix scaled = eigen.eigenvectors;
  for (std::size_t column = 0; column < scaled.columns(); ++column) {
    const double factor = 1.0 / std::sqrt(eigen.eigenvalues[column]);
    for (std::size_t row = 0; row < scaled.rows(); ++row)
      scaled(row, column) *= factor;
  }
  auto result = modernqc::linalg::multiply(
      scaled, modernqc::linalg::transpose(eigen.eigenvectors));
  modernqc::linalg::symmetrize_in_place(result);
  return {std::move(result), minimum};
}

ModernMatrix apply_metric_pseudoinverse(
    const ModernMatrix& metric, const ModernMatrix& right_hand_side,
    const BasisProjectionOptions& options, std::size_t& retained_rank) {
  if (metric.rows() != metric.columns() ||
      metric.rows() != right_hand_side.rows())
    throw std::invalid_argument("metric pseudoinverse dimensions mismatch");
  const auto eigen = modernqc::linalg::diagonalize_symmetric(metric);
  const double largest = eigen.eigenvalues.back();
  if (!std::isfinite(largest) || largest <= 0.0)
    throw std::runtime_error("target AO overlap is not positive semidefinite");
  auto transformed = modernqc::linalg::multiply(
      modernqc::linalg::transpose(eigen.eigenvectors), right_hand_side);
  retained_rank = 0;
  for (std::size_t row = 0; row < transformed.rows(); ++row) {
    const double eigenvalue = eigen.eigenvalues[row];
    if (eigenvalue > options.rank_relative_tolerance * largest) {
      ++retained_rank;
      for (std::size_t column = 0; column < transformed.columns(); ++column)
        transformed(row, column) /= eigenvalue;
    } else {
      for (std::size_t column = 0; column < transformed.columns(); ++column)
        transformed(row, column) = 0.0;
    }
  }
  return modernqc::linalg::multiply(eigen.eigenvectors, transformed);
}

ModernMatrix orthonormalize_columns(
    const ModernMatrix& raw, const BasisProjectionOptions& options,
    double& minimum_gram_eigenvalue) {
  const auto inverse_sqrt = inverse_symmetric_square_root(
      modernqc::linalg::multiply(modernqc::linalg::transpose(raw), raw),
      options, "projected orbital Gram matrix");
  minimum_gram_eigenvalue = inverse_sqrt.minimum_eigenvalue;
  return modernqc::linalg::multiply(raw, inverse_sqrt.matrix);
}

ModernMatrix complete_orthogonal_columns(
    const ModernMatrix& leading, const BasisProjectionOptions& options) {
  if (leading.empty() || leading.rows() < leading.columns())
    throw std::invalid_argument(
        "orthogonal completion requires rows >= leading columns");
  ModernMatrix result(leading.rows(), leading.rows());
  for (std::size_t column = 0; column < leading.columns(); ++column)
    for (std::size_t row = 0; row < leading.rows(); ++row)
      result(row, column) = leading(row, column);

  std::size_t completed = leading.columns();
  const double acceptance =
      std::max(1.0e-13, options.rank_relative_tolerance);
  for (std::size_t candidate = 0;
       candidate < leading.rows() && completed < leading.rows(); ++candidate) {
    std::vector<double> vector(leading.rows(), 0.0);
    vector[candidate] = 1.0;
    for (int pass = 0; pass < 2; ++pass) {
      for (std::size_t column = 0; column < completed; ++column) {
        double projection = 0.0;
        for (std::size_t row = 0; row < leading.rows(); ++row)
          projection += result(row, column) * vector[row];
        for (std::size_t row = 0; row < leading.rows(); ++row)
          vector[row] -= projection * result(row, column);
      }
    }
    double squared_norm = 0.0;
    for (const double value : vector)
      squared_norm += value * value;
    if (squared_norm <= acceptance * acceptance)
      continue;
    const double inverse_norm = 1.0 / std::sqrt(squared_norm);
    for (std::size_t row = 0; row < leading.rows(); ++row)
      result(row, completed) = vector[row] * inverse_norm;
    ++completed;
  }
  if (completed != leading.rows())
    throw std::runtime_error(
        "failed to construct deterministic orbital-space complement");
  const double error = identity_error(modernqc::linalg::multiply(
      modernqc::linalg::transpose(result), result));
  if (error > options.orthonormality_tolerance)
    throw std::runtime_error(
        "completed orbital rotation violates orthogonality tolerance");
  return result;
}

ModernMatrix spin_expansion(const ModernMatrix& spatial) {
  ModernMatrix spin(checked_twice(spatial.rows(), "spin target"),
                    checked_twice(spatial.columns(), "spin source"));
  for (std::size_t target = 0; target < spatial.rows(); ++target)
    for (std::size_t source = 0; source < spatial.columns(); ++source)
      for (std::size_t sigma = 0; sigma < 2; ++sigma)
        spin(2 * target + sigma, 2 * source + sigma) =
            spatial(target, source);
  return spin;
}

double antisymmetry_error(const Tensor4& t2) {
  const auto d = t2.dimensions();
  if (d[0] != d[1] || d[2] != d[3])
    throw std::invalid_argument("T2 dimensions are not paired");
  double error = 0.0;
  for (std::size_t i = 0; i < d[0]; ++i)
    for (std::size_t j = 0; j < d[1]; ++j)
      for (std::size_t a = 0; a < d[2]; ++a)
        for (std::size_t b = 0; b < d[3]; ++b) {
          error = std::max(error, std::abs(t2(i, j, a, b) +
                                           t2(j, i, a, b)));
          error = std::max(error, std::abs(t2(i, j, a, b) +
                                           t2(i, j, b, a)));
        }
  return error;
}

} // namespace

ProjectedRhfGuess project_rhf_occupied_guess(
    const ModernMatrix& source_overlap,
    const ModernMatrix& source_occupied_coefficients,
    const ModernMatrix& target_overlap,
    const ModernMatrix& target_source_overlap,
    const BasisProjectionOptions& options) {
  validate_options(options);
  if (source_overlap.empty() ||
      source_overlap.rows() != source_overlap.columns() ||
      target_overlap.empty() || target_overlap.rows() != target_overlap.columns() ||
      source_occupied_coefficients.rows() != source_overlap.rows() ||
      source_occupied_coefficients.columns() == 0 ||
      target_source_overlap.rows() != target_overlap.rows() ||
      target_source_overlap.columns() != source_overlap.rows())
    throw std::invalid_argument("RHF basis-projection dimensions mismatch");
  modernqc::linalg::require_finite(source_overlap, "source AO overlap");
  modernqc::linalg::require_finite(source_occupied_coefficients,
                                   "source occupied coefficients");
  modernqc::linalg::require_finite(target_overlap, "target AO overlap");
  modernqc::linalg::require_finite(target_source_overlap,
                                   "target-source AO overlap");
  const double source_error = identity_error(
      metric_gram(source_occupied_coefficients, source_overlap));
  if (source_error > options.orthonormality_tolerance)
    throw std::runtime_error(
        "source occupied orbitals violate metric orthonormality tolerance");

  const auto rhs = modernqc::linalg::multiply(
      target_source_overlap, source_occupied_coefficients);
  std::size_t target_rank = 0;
  const auto raw = apply_metric_pseudoinverse(
      target_overlap, rhs, options, target_rank);
  if (target_rank < source_occupied_coefficients.columns())
    throw std::runtime_error(
        "target AO metric rank is smaller than the occupied space");
  const auto gram = metric_gram(raw, target_overlap);
  const auto inverse_sqrt = inverse_symmetric_square_root(
      gram, options, "projected occupied Gram matrix");
  auto occupied = modernqc::linalg::multiply(raw, inverse_sqrt.matrix);
  const double projected_error = identity_error(
      metric_gram(occupied, target_overlap));
  if (projected_error > options.orthonormality_tolerance)
    throw std::runtime_error(
        "projected occupied orbitals violate metric orthonormality tolerance");
  auto density = modernqc::linalg::scaled(
      modernqc::linalg::multiply(
          occupied, modernqc::linalg::transpose(occupied)),
      2.0);
  modernqc::linalg::symmetrize_in_place(density);
  double electron_count = 0.0;
  for (std::size_t mu = 0; mu < density.rows(); ++mu)
    for (std::size_t nu = 0; nu < density.columns(); ++nu)
      electron_count += density(mu, nu) * target_overlap(nu, mu);
  const double expected_electrons =
      2.0 * static_cast<double>(source_occupied_coefficients.columns());
  if (std::abs(electron_count - expected_electrons) >
      10.0 * options.orthonormality_tolerance * expected_electrons)
    throw std::runtime_error(
        "projected RHF density has the wrong electron count");
  return {std::move(occupied),
          std::move(density),
          BasisProjectionDiagnostics{
              source_overlap.rows(), target_overlap.rows(),
              source_occupied_coefficients.columns(), target_rank,
              source_error, inverse_sqrt.minimum_eigenvalue,
              projected_error, electron_count}};
}

OrbitalRotationProjection project_orbital_rotation(
    const ModernMatrix& target_reference,
    const ModernMatrix& source_coefficients,
    const ModernMatrix& target_source_overlap,
    const BasisProjectionOptions& options) {
  validate_options(options);
  if (target_reference.empty() || source_coefficients.empty() ||
      target_reference.columns() < source_coefficients.columns() ||
      target_source_overlap.rows() != target_reference.rows() ||
      target_source_overlap.columns() != source_coefficients.rows())
    throw std::invalid_argument("orbital-rotation projection dimensions mismatch");
  const auto raw = modernqc::linalg::multiply(
      modernqc::linalg::transpose(target_reference),
      modernqc::linalg::multiply(target_source_overlap,
                                 source_coefficients));
  double minimum = 0.0;
  auto leading = orthonormalize_columns(raw, options, minimum);
  auto rotation = complete_orthogonal_columns(leading, options);
  const double error = identity_error(modernqc::linalg::multiply(
      modernqc::linalg::transpose(rotation), rotation));
  return {std::move(leading), std::move(rotation), minimum, error};
}

ProjectedCcsdGuess project_ccsd_amplitudes(
    const Matrix& source_t1, const Tensor4& source_t2,
    const ModernMatrix& occupied_source_to_target,
    const ModernMatrix& virtual_source_to_target) {
  const auto t1d = source_t1.dimensions();
  const auto t2d = source_t2.dimensions();
  if (occupied_source_to_target.empty() ||
      virtual_source_to_target.empty() ||
      occupied_source_to_target.rows() <
          occupied_source_to_target.columns() ||
      virtual_source_to_target.rows() < virtual_source_to_target.columns())
    throw std::invalid_argument(
        "CCSD orbital projections must be nonempty and target-by-source");
  modernqc::linalg::require_finite(occupied_source_to_target,
                                   "occupied CCSD projection");
  modernqc::linalg::require_finite(virtual_source_to_target,
                                   "virtual CCSD projection");
  const double occupied_projection_error = identity_error(
      modernqc::linalg::multiply(
          modernqc::linalg::transpose(occupied_source_to_target),
          occupied_source_to_target));
  const double virtual_projection_error = identity_error(
      modernqc::linalg::multiply(
          modernqc::linalg::transpose(virtual_source_to_target),
          virtual_source_to_target));
  if (occupied_projection_error > 1.0e-8 ||
      virtual_projection_error > 1.0e-8)
    throw std::runtime_error(
        "CCSD orbital projections violate semi-orthogonality tolerance");
  const std::size_t source_occupied = occupied_source_to_target.columns();
  const std::size_t source_virtual = virtual_source_to_target.columns();
  const std::size_t source_occupied_spin =
      checked_twice(source_occupied, "source occupied spin");
  const std::size_t source_virtual_spin =
      checked_twice(source_virtual, "source virtual spin");
  if (source_occupied == 0 || source_virtual == 0 ||
      t1d != std::array<std::size_t, 2>{source_occupied_spin,
                                        source_virtual_spin} ||
      t2d != std::array<std::size_t, 4>{source_occupied_spin,
                                        source_occupied_spin,
                                        source_virtual_spin,
                                        source_virtual_spin})
    throw std::invalid_argument(
        "source CCSD amplitudes do not match spatial projection dimensions");
  for (const double value : source_t1.data())
    if (!std::isfinite(value))
      throw std::invalid_argument("source CCSD T1 contains non-finite data");
  for (const double value : source_t2.data())
    if (!std::isfinite(value))
      throw std::invalid_argument("source CCSD T2 contains non-finite data");
  if (antisymmetry_error(source_t2) > 1.0e-9)
    throw std::invalid_argument(
        "source CCSD T2 violates spin-orbital antisymmetry tolerance");
  const auto occupied = spin_expansion(occupied_source_to_target);
  const auto virtuals = spin_expansion(virtual_source_to_target);
  const std::size_t os = occupied.columns();
  const std::size_t ot = occupied.rows();
  const std::size_t vs = virtuals.columns();
  const std::size_t vt = virtuals.rows();
  Matrix target_t1({ot, vt});
  for (std::size_t i = 0; i < ot; ++i)
    for (std::size_t a = 0; a < vt; ++a)
      for (std::size_t source_i = 0; source_i < os; ++source_i)
        for (std::size_t source_a = 0; source_a < vs; ++source_a)
          target_t1(i, a) += occupied(i, source_i) *
                             virtuals(a, source_a) *
                             source_t1(source_i, source_a);

  Tensor4 target_t2({ot, ot, vt, vt});
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // Stream one target occupied-pair transformation per worker. This avoids
    // retaining O(o^2 v_target v_source) and O(o^2 v_source^2) tensors in
    // addition to the unavoidable target T2 initial guess.
    std::vector<double> source_pair(
        checked_product(vs, vs, "source virtual-pair workspace"), 0.0);
    std::vector<double> left_transformed(
        checked_product(vt, vs, "target-source virtual workspace"), 0.0);
    std::vector<double> occupied_left(os, 0.0);
#ifdef _OPENMP
#pragma omp for collapse(2) schedule(static)
#endif
    for (std::size_t i = 0; i < ot; ++i)
      for (std::size_t j = 0; j < ot; ++j) {
        std::fill(source_pair.begin(), source_pair.end(), 0.0);
        std::fill(left_transformed.begin(), left_transformed.end(), 0.0);
        for (std::size_t source_a = 0; source_a < vs; ++source_a)
          for (std::size_t source_b = 0; source_b < vs; ++source_b) {
            std::fill(occupied_left.begin(), occupied_left.end(), 0.0);
            for (std::size_t source_j = 0; source_j < os; ++source_j)
              for (std::size_t source_i = 0; source_i < os; ++source_i)
                occupied_left[source_j] +=
                    occupied(i, source_i) *
                    source_t2(source_i, source_j, source_a, source_b);
            for (std::size_t source_j = 0; source_j < os; ++source_j)
              source_pair[source_a * vs + source_b] +=
                  occupied(j, source_j) * occupied_left[source_j];
          }
        for (std::size_t a = 0; a < vt; ++a)
          for (std::size_t source_b = 0; source_b < vs; ++source_b)
            for (std::size_t source_a = 0; source_a < vs; ++source_a)
              left_transformed[a * vs + source_b] +=
                  virtuals(a, source_a) *
                  source_pair[source_a * vs + source_b];
        for (std::size_t a = 0; a < vt; ++a)
          for (std::size_t b = 0; b < vt; ++b)
            for (std::size_t source_b = 0; source_b < vs; ++source_b)
              target_t2(i, j, a, b) +=
                  virtuals(b, source_b) *
                  left_transformed[a * vs + source_b];
      }
  }
  const double error = antisymmetry_error(target_t2);
  if (!std::isfinite(error) || error > 1.0e-9)
    throw std::runtime_error(
        "projected CCSD T2 violates spin-orbital antisymmetry tolerance");
  return {std::move(target_t1), std::move(target_t2), error};
}

} // namespace localized_cc
