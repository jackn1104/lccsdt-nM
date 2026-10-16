#include "modernqc/localization/orbital_localization.hpp"

#include "modernqc/linalg/eigensolver.hpp"
#include "modernqc/linalg/matrix.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace modernqc::localization {
namespace {

using Matrix = linalg::Matrix;

constexpr double input_orthonormality_tolerance = 1.0e-8;
constexpr double restart_repair_tolerance = 1.0e-6;
constexpr double moment_negative_tolerance = 1.0e-10;

void sum_parallel(std::vector<double>& values,
                  const LocalizationParallelOptions& parallel) {
  if(parallel.ranks == 1) {
    return;
  }
  parallel.sum_in_place(values);
}

void validate_parallel(const LocalizationParallelOptions& parallel) {
  if(parallel.ranks == 0 || parallel.rank >= parallel.ranks) {
    throw std::invalid_argument(
        "localization MPI rank/count are outside their domains");
  }
  if(parallel.ranks > 1 && !parallel.sum_in_place) {
    throw std::invalid_argument(
        "distributed localization requires an elementwise-sum callback");
  }
}

[[nodiscard]] std::size_t index2(std::size_t a, std::size_t b) {
  return 3 * a + b;
}

[[nodiscard]] std::size_t index3(std::size_t a, std::size_t b,
                                 std::size_t c) {
  return 3 * index2(a, b) + c;
}

[[nodiscard]] std::size_t index4(std::size_t a, std::size_t b,
                                 std::size_t c, std::size_t d) {
  return 3 * index3(a, b, c) + d;
}

[[nodiscard]] double maximum_identity_error(const Matrix& matrix) {
  if(matrix.empty() || matrix.rows() != matrix.columns()) {
    throw std::invalid_argument(
        "orthogonality check requires a nonempty square matrix");
  }
  const Matrix gram =
      linalg::multiply(linalg::transpose(matrix), matrix);
  return linalg::maximum_absolute_value(linalg::subtract(
      gram, Matrix::identity(matrix.columns())));
}

[[nodiscard]] double metric_orthonormality_error(
    const Matrix& coefficients, const Matrix& overlap) {
  const Matrix gram = linalg::multiply(
      linalg::transpose(coefficients),
      linalg::multiply(overlap, coefficients));
  return linalg::maximum_absolute_value(linalg::subtract(
      gram, Matrix::identity(coefficients.columns())));
}

[[nodiscard]] double projector_error(const Matrix& before,
                                     const Matrix& after) {
  return linalg::maximum_absolute_value(linalg::subtract(
      linalg::multiply(before, linalg::transpose(before)),
      linalg::multiply(after, linalg::transpose(after))));
}

[[nodiscard]] Matrix inverse_symmetric_square_root(
    const Matrix& matrix, double positive_tolerance,
    const char* description) {
  const linalg::SymmetricEigendecomposition eigensystem =
      linalg::diagonalize_symmetric(matrix);
  const double largest = eigensystem.eigenvalues.back();
  if(!std::isfinite(largest) || largest <= 0.0) {
    throw std::invalid_argument(std::string{description} +
                                " is not positive definite");
  }
  Matrix scaled_vectors = eigensystem.eigenvectors;
  for(std::size_t column = 0; column < matrix.columns(); ++column) {
    const double eigenvalue = eigensystem.eigenvalues[column];
    if(eigenvalue <= positive_tolerance * largest) {
      throw std::invalid_argument(std::string{description} +
                                  " is not positive definite");
    }
    const double scale = 1.0 / std::sqrt(eigenvalue);
    for(std::size_t row = 0; row < matrix.rows(); ++row) {
      scaled_vectors(row, column) *= scale;
    }
  }
  Matrix result = linalg::multiply(
      scaled_vectors, linalg::transpose(eigensystem.eigenvectors));
  linalg::symmetrize_in_place(result);
  return result;
}

[[nodiscard]] Matrix symmetric_square_root(const Matrix& overlap) {
  const linalg::SymmetricEigendecomposition eigensystem =
      linalg::diagonalize_symmetric(overlap);
  const double largest = eigensystem.eigenvalues.back();
  if(!std::isfinite(largest) || largest <= 0.0) {
    throw std::invalid_argument("overlap is not positive definite");
  }
  Matrix scaled_vectors = eigensystem.eigenvectors;
  for(std::size_t column = 0; column < overlap.columns(); ++column) {
    const double eigenvalue = eigensystem.eigenvalues[column];
    if(eigenvalue <= 1.0e-12 * largest) {
      throw std::invalid_argument("overlap is not positive definite");
    }
    const double scale = std::sqrt(eigenvalue);
    for(std::size_t row = 0; row < overlap.rows(); ++row) {
      scaled_vectors(row, column) *= scale;
    }
  }
  Matrix result = linalg::multiply(
      scaled_vectors, linalg::transpose(eigensystem.eigenvectors));
  linalg::symmetrize_in_place(result);
  return result;
}

[[nodiscard]] Matrix polar_orthogonalize(const Matrix& matrix) {
  const Matrix gram =
      linalg::multiply(linalg::transpose(matrix), matrix);
  return linalg::multiply(
      matrix, inverse_symmetric_square_root(
                  gram, 1.0e-14, "polar-retraction Gram matrix"));
}

[[nodiscard]] Matrix polar_factor_from_tangent(const Matrix& tangent) {
  Matrix trial = linalg::add(Matrix::identity(tangent.rows()), tangent);
  return polar_orthogonalize(trial);
}

[[nodiscard]] Matrix retract(const Matrix& rotation,
                             const Matrix& tangent) {
  return linalg::multiply(rotation, polar_factor_from_tangent(tangent));
}

[[nodiscard]] Matrix skew(const Matrix& matrix) {
  if(matrix.empty() || matrix.rows() != matrix.columns()) {
    throw std::invalid_argument(
        "skew projection requires a nonempty square matrix");
  }
  Matrix result{matrix.rows(), matrix.columns()};
  for(std::size_t j = 0; j < matrix.columns(); ++j) {
    for(std::size_t i = 0; i < j; ++i) {
      const double value = 0.5 * (matrix(i, j) - matrix(j, i));
      result(i, j) = value;
      result(j, i) = -value;
    }
  }
  return result;
}

[[nodiscard]] Matrix tangent_gradient(
    const Matrix& rotation, const Matrix& euclidean_gradient) {
  return linalg::scaled(
      skew(linalg::multiply(linalg::transpose(rotation),
                            euclidean_gradient)),
      2.0);
}

[[nodiscard]] double tangent_inner(const Matrix& left,
                                   const Matrix& right) {
  if(left.rows() != right.rows() ||
     left.columns() != right.columns()) {
    throw std::invalid_argument(
        "tangent inner product requires equal dimensions");
  }
  long double sum = 0.0L;
  for(std::size_t index = 0; index < left.size(); ++index) {
    sum += static_cast<long double>(left.values()[index]) *
           static_cast<long double>(right.values()[index]);
  }
  return 0.5 * static_cast<double>(sum);
}

[[nodiscard]] double tangent_norm(const Matrix& tangent) {
  return std::sqrt(std::max(0.0, tangent_inner(tangent, tangent)));
}

[[nodiscard]] Matrix transform_operator(const Matrix& coefficients,
                                        const Matrix& operation) {
  return linalg::multiply(
      linalg::transpose(coefficients),
      linalg::multiply(operation, coefficients));
}

[[nodiscard]] double quadratic_column(const Matrix& operation,
                                      const Matrix& rotation,
                                      std::size_t orbital) {
  long double result = 0.0L;
  for(std::size_t j = 0; j < operation.columns(); ++j) {
    long double row_value = 0.0L;
    for(std::size_t i = 0; i < operation.rows(); ++i) {
      row_value +=
          static_cast<long double>(operation(i, j)) *
          static_cast<long double>(rotation(i, orbital));
    }
    result += static_cast<long double>(rotation(j, orbital)) * row_value;
  }
  return static_cast<double>(result);
}

void add_matvec_column(Matrix& target, std::size_t orbital,
                       const Matrix& operation, const Matrix& rotation,
                       double scale) {
  if(scale == 0.0) {
    return;
  }
  for(std::size_t row = 0; row < operation.rows(); ++row) {
    long double value = 0.0L;
    for(std::size_t column = 0; column < operation.columns(); ++column) {
      value +=
          static_cast<long double>(operation(row, column)) *
          static_cast<long double>(rotation(column, orbital));
    }
    target(row, orbital) += scale * static_cast<double>(value);
  }
}

[[nodiscard]] Matrix symmetrized_copy(const Matrix& matrix,
                                      std::size_t dimension,
                                      const char* description) {
  if(matrix.rows() != dimension || matrix.columns() != dimension) {
    throw std::invalid_argument(std::string{description} +
                                " has incompatible AO dimensions");
  }
  linalg::require_finite(matrix, description);
  Matrix result = matrix;
  linalg::symmetrize_in_place(result);
  return result;
}

[[nodiscard]] CartesianMomentIntegrals validate_and_symmetrize_moments(
    const CartesianMomentIntegrals& moments, std::size_t dimension) {
  CartesianMomentIntegrals result = moments;
  for(Matrix& matrix : result.first) {
    matrix = symmetrized_copy(matrix, dimension, "first AO moment");
  }
  for(Matrix& matrix : result.second) {
    matrix = symmetrized_copy(matrix, dimension, "second AO moment");
  }
  for(Matrix& matrix : result.third) {
    matrix = symmetrized_copy(matrix, dimension, "third AO moment");
  }
  for(Matrix& matrix : result.fourth) {
    matrix = symmetrized_copy(matrix, dimension, "fourth AO moment");
  }

  const CartesianMomentIntegrals original = result;
  for(std::size_t a = 0; a < 3; ++a) {
    for(std::size_t b = 0; b < 3; ++b) {
      result.second[index2(a, b)] = linalg::scaled(
          linalg::add(original.second[index2(a, b)],
                      original.second[index2(b, a)]),
          0.5);
      for(std::size_t c = 0; c < 3; ++c) {
        const std::array<std::array<std::size_t, 3>, 6> permutations{{
            {a, b, c}, {a, c, b}, {b, a, c},
            {b, c, a}, {c, a, b}, {c, b, a},
        }};
        Matrix average{dimension, dimension};
        for(const auto& p : permutations) {
          average = linalg::add(
              average, original.third[index3(p[0], p[1], p[2])]);
        }
        result.third[index3(a, b, c)] =
            linalg::scaled(average, 1.0 / 6.0);
        for(std::size_t d = 0; d < 3; ++d) {
          std::array<std::size_t, 4> values{a, b, c, d};
          std::sort(values.begin(), values.end());
          Matrix fourth_average{dimension, dimension};
          std::size_t count = 0;
          do {
            fourth_average = linalg::add(
                fourth_average,
                original.fourth[index4(values[0], values[1],
                                       values[2], values[3])]);
            ++count;
          } while(std::next_permutation(values.begin(), values.end()));
          result.fourth[index4(a, b, c, d)] =
              linalg::scaled(fourth_average,
                             1.0 / static_cast<double>(count));
        }
      }
    }
  }
  return result;
}

void validate_common_inputs(const Matrix& canonical_coefficients,
                            Matrix& overlap,
                            const TrustRegionLocalizationOptions& options) {
  validate_parallel(options.parallel);
  if(canonical_coefficients.empty() || overlap.empty() ||
     overlap.rows() != overlap.columns() ||
     canonical_coefficients.rows() != overlap.rows()) {
    throw std::invalid_argument(
        "localization AO/orbital dimensions are incompatible");
  }
  linalg::require_finite(canonical_coefficients,
                         "canonical localization coefficients");
  linalg::require_finite(overlap, "localization overlap");
  linalg::symmetrize_in_place(overlap);
  static_cast<void>(symmetric_square_root(overlap));
  if(metric_orthonormality_error(canonical_coefficients, overlap) >
     input_orthonormality_tolerance) {
    throw std::invalid_argument(
        "canonical orbitals are not orthonormal in the AO metric");
  }
  if(options.maximum_iterations <= 0 ||
     !std::isfinite(options.gradient_tolerance) ||
     options.gradient_tolerance <= 0.0 ||
     !std::isfinite(options.objective_tolerance) ||
     options.objective_tolerance <= 0.0 ||
     !std::isfinite(options.initial_trust_radius) ||
     options.initial_trust_radius <= 0.0 ||
     !std::isfinite(options.maximum_trust_radius) ||
     options.maximum_trust_radius < options.initial_trust_radius ||
     !std::isfinite(options.acceptance_threshold) ||
     options.acceptance_threshold <= 0.0 ||
     options.acceptance_threshold >= 1.0 ||
     !std::isfinite(options.minimum_trust_radius) ||
     options.minimum_trust_radius <= 0.0 ||
     options.minimum_trust_radius >= options.initial_trust_radius ||
     !std::isfinite(options.finite_difference_step) ||
     options.finite_difference_step <= 0.0 ||
     !std::isfinite(options.gradient_check_tolerance) ||
     options.gradient_check_tolerance <= 0.0 ||
     options.gradient_check_directions <= 0 ||
     !std::isfinite(options.invariant_tolerance) ||
     options.invariant_tolerance <= 0.0) {
    throw std::invalid_argument(
        "trust-region localization options are outside their domains");
  }
  if(options.restart_state && !options.initial_rotation.empty()) {
    throw std::invalid_argument(
        "localization initial_rotation and restart_state are mutually exclusive");
  }
  if(options.restart_state) {
    const auto& state = *options.restart_state;
    if(state.completed_iterations < 0 || state.accepted_steps < 0 ||
       state.rejected_steps < 0 ||
       state.accepted_steps + state.rejected_steps >
           state.completed_iterations ||
       !std::isfinite(state.trust_radius) ||
       state.trust_radius <= 0.0 ||
       state.trust_radius > options.maximum_trust_radius ||
       state.objective_history.empty() ||
       std::any_of(state.objective_history.begin(),
                   state.objective_history.end(),
                   [](const double value) { return !std::isfinite(value); })) {
      throw std::invalid_argument(
          "localization restart state is outside its domain");
    }
  }
}

[[nodiscard]] Matrix initial_rotation(
    std::size_t orbitals,
    const TrustRegionLocalizationOptions& options) {
  const Matrix* requested = nullptr;
  if(options.restart_state) {
    requested = &options.restart_state->rotation;
  } else if(!options.initial_rotation.empty()) {
    requested = &options.initial_rotation;
  }
  if(requested == nullptr || requested->empty()) {
    return Matrix::identity(orbitals);
  }
  if(requested->rows() != orbitals ||
     requested->columns() != orbitals) {
    throw std::invalid_argument(
        "localization restart rotation has incompatible dimensions");
  }
  linalg::require_finite(*requested,
                         "localization restart rotation");
  const double error = maximum_identity_error(*requested);
  if(error > restart_repair_tolerance) {
    throw std::invalid_argument(
        "localization restart rotation is too far from orthogonal");
  }
  if(error > options.invariant_tolerance) {
    return polar_orthogonalize(*requested);
  }
  return *requested;
}

struct ObjectiveEvaluation {
  double objective;
  double loss;
  Matrix loss_gradient;
};

using ObjectiveFunction =
    std::function<ObjectiveEvaluation(const Matrix&, bool)>;

struct MomentObjective {
  std::array<Matrix, 3> first;
  std::array<Matrix, 9> second;
  std::array<Matrix, 27> third;
  std::array<Matrix, 81> fourth;
  Matrix second_trace;
  Matrix fourth_trace;
  std::array<Matrix, 3> third_contractions;
};

[[nodiscard]] MomentObjective transform_moments(
    const Matrix& coefficients,
    const CartesianMomentIntegrals& moments,
    const LocalizationParallelOptions& parallel) {
  MomentObjective result;
  const std::size_t n = coefficients.columns();
  result.second_trace = Matrix{n, n};
  result.fourth_trace = Matrix{n, n};
  for(Matrix& matrix : result.third_contractions) {
    matrix = Matrix{n, n};
  }
  std::size_t flat_index = 0;
  const auto transform_collection =
      [&](auto& target, const auto& source) {
        for(std::size_t index = 0; index < target.size(); ++index) {
          if(flat_index % parallel.ranks == parallel.rank) {
            target[index] =
                transform_operator(coefficients, source[index]);
          } else {
            target[index] = Matrix{n, n};
          }
          ++flat_index;
        }
      };
  transform_collection(result.first, moments.first);
  transform_collection(result.second, moments.second);
  transform_collection(result.third, moments.third);
  transform_collection(result.fourth, moments.fourth);

  if(parallel.ranks > 1) {
    const std::size_t matrix_count = result.first.size() +
                                     result.second.size() +
                                     result.third.size() +
                                     result.fourth.size();
    if(n != 0 && matrix_count >
                     std::numeric_limits<std::size_t>::max() / n / n) {
      throw std::overflow_error(
          "distributed localization moment buffer overflows size_t");
    }
    std::vector<double> packed;
    packed.reserve(matrix_count * n * n);
    const auto pack = [&](const auto& collection) {
      for(const Matrix& matrix : collection) {
        packed.insert(packed.end(), matrix.values().begin(),
                      matrix.values().end());
      }
    };
    pack(result.first);
    pack(result.second);
    pack(result.third);
    pack(result.fourth);
    sum_parallel(packed, parallel);
    std::size_t offset = 0;
    const auto unpack = [&](auto& collection) {
      for(Matrix& matrix : collection) {
        std::copy_n(packed.begin() + static_cast<std::ptrdiff_t>(offset),
                    matrix.size(), matrix.values().begin());
        offset += matrix.size();
      }
    };
    unpack(result.first);
    unpack(result.second);
    unpack(result.third);
    unpack(result.fourth);
  }
  for(std::size_t a = 0; a < 3; ++a) {
    result.second_trace = linalg::add(
        result.second_trace, result.second[index2(a, a)]);
    for(std::size_t b = 0; b < 3; ++b) {
      result.third_contractions[a] = linalg::add(
          result.third_contractions[a],
          result.third[index3(a, b, b)]);
      result.fourth_trace = linalg::add(
          result.fourth_trace,
          result.fourth[index4(a, a, b, b)]);
    }
  }
  return result;
}

struct SpatialValues {
  std::array<double, 3> centroid{};
  double omega2{0.0};
  double omega4{0.0};
};

[[nodiscard]] SpatialValues spatial_values(
    const MomentObjective& moments, const Matrix& rotation,
    std::size_t orbital) {
  SpatialValues values;
  double centroid_norm2 = 0.0;
  for(std::size_t a = 0; a < 3; ++a) {
    values.centroid[a] =
        quadratic_column(moments.first[a], rotation, orbital);
    centroid_norm2 += values.centroid[a] * values.centroid[a];
  }
  const double raw_second =
      quadratic_column(moments.second_trace, rotation, orbital);
  values.omega2 = raw_second - centroid_norm2;

  double centroid_third = 0.0;
  double centroid_second = 0.0;
  for(std::size_t a = 0; a < 3; ++a) {
    centroid_third +=
        values.centroid[a] *
        quadratic_column(moments.third_contractions[a], rotation, orbital);
    for(std::size_t b = 0; b < 3; ++b) {
      centroid_second +=
          values.centroid[a] * values.centroid[b] *
          quadratic_column(moments.second[index2(a, b)], rotation,
                           orbital);
    }
  }
  const double raw_fourth =
      quadratic_column(moments.fourth_trace, rotation, orbital);
  values.omega4 =
      raw_fourth - 4.0 * centroid_third +
      2.0 * centroid_norm2 * raw_second +
      4.0 * centroid_second -
      3.0 * centroid_norm2 * centroid_norm2;

  const double scale =
      std::max({1.0, std::abs(raw_second), std::abs(raw_fourth)});
  if(values.omega2 < -moment_negative_tolerance * scale ||
     values.omega4 < -moment_negative_tolerance * scale) {
    throw std::runtime_error(
        "AO moments produce a negative central spatial moment");
  }
  values.omega2 = std::max(0.0, values.omega2);
  values.omega4 = std::max(0.0, values.omega4);
  return values;
}

[[nodiscard]] ObjectiveFunction make_fm_objective(
    std::shared_ptr<const MomentObjective> moments, int power,
    LocalizationParallelOptions parallel) {
  return [moments = std::move(moments), power,
          parallel = std::move(parallel)](
             const Matrix& rotation,
             bool gradient_required) -> ObjectiveEvaluation {
    const std::size_t n = rotation.columns();
    Matrix gradient{n, n};
    std::vector<double> objective_terms(n, 0.0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for(std::ptrdiff_t raw_orbital = 0;
        raw_orbital < static_cast<std::ptrdiff_t>(n); ++raw_orbital) {
      const auto orbital = static_cast<std::size_t>(raw_orbital);
      if(orbital % parallel.ranks != parallel.rank) {
        continue;
      }
      const SpatialValues values =
          spatial_values(*moments, rotation, orbital);
      const double omega = values.omega4;
      objective_terms[orbital] = std::pow(omega, power);
      if(!gradient_required) {
        continue;
      }
      const double centroid_norm2 =
          values.centroid[0] * values.centroid[0] +
          values.centroid[1] * values.centroid[1] +
          values.centroid[2] * values.centroid[2];
      const double raw_second =
          quadratic_column(moments->second_trace, rotation, orbital);
      std::array<double, 3> third{};
      std::array<double, 3> second_centroid{};
      for(std::size_t a = 0; a < 3; ++a) {
        third[a] = quadratic_column(
            moments->third_contractions[a], rotation, orbital);
        for(std::size_t b = 0; b < 3; ++b) {
          second_centroid[a] +=
              quadratic_column(moments->second[index2(a, b)],
                               rotation, orbital) *
              values.centroid[b];
        }
      }

      Matrix omega_gradient{n, n};
      add_matvec_column(omega_gradient, orbital, moments->fourth_trace,
                        rotation, 2.0);
      for(std::size_t a = 0; a < 3; ++a) {
        add_matvec_column(omega_gradient, orbital, moments->first[a],
                          rotation, -8.0 * third[a]);
        add_matvec_column(omega_gradient, orbital,
                          moments->third_contractions[a], rotation,
                          -8.0 * values.centroid[a]);
        add_matvec_column(omega_gradient, orbital, moments->first[a],
                          rotation,
                          8.0 * raw_second * values.centroid[a]);
        add_matvec_column(omega_gradient, orbital, moments->first[a],
                          rotation, 16.0 * second_centroid[a]);
        add_matvec_column(omega_gradient, orbital, moments->first[a],
                          rotation,
                          -24.0 * centroid_norm2 *
                              values.centroid[a]);
      }
      add_matvec_column(omega_gradient, orbital, moments->second_trace,
                        rotation, 4.0 * centroid_norm2);
      for(std::size_t a = 0; a < 3; ++a) {
        for(std::size_t b = 0; b < 3; ++b) {
          add_matvec_column(
              omega_gradient, orbital, moments->second[index2(a, b)],
              rotation,
              8.0 * values.centroid[a] * values.centroid[b]);
        }
      }
      const double power_factor =
          static_cast<double>(power) *
          (power == 1 ? 1.0 : std::pow(omega, power - 1));
      for(std::size_t row = 0; row < n; ++row) {
        gradient(row, orbital) =
            power_factor * omega_gradient(row, orbital);
      }
    }
    long double objective = 0.0L;
    for(const auto term : objective_terms)
      objective += static_cast<long double>(term);
    std::vector<double> reduced(
        1 + (gradient_required ? gradient.size() : 0), 0.0);
    reduced[0] = static_cast<double>(objective);
    if(gradient_required) {
      std::copy(gradient.values().begin(), gradient.values().end(),
                reduced.begin() + 1);
    }
    sum_parallel(reduced, parallel);
    const double objective_double = reduced[0];
    if(gradient_required) {
      std::copy(reduced.begin() + 1, reduced.end(),
                gradient.values().begin());
    }
    if(!std::isfinite(objective_double)) {
      throw std::runtime_error(
          "fourth-moment localization objective became nonfinite");
    }
    if(gradient_required) {
      linalg::require_finite(
          gradient, "fourth-moment localization gradient");
    }
    return ObjectiveEvaluation{
        .objective = objective_double,
        .loss = objective_double,
        .loss_gradient = std::move(gradient),
    };
  };
}

[[nodiscard]] ObjectiveFunction make_pm_objective(
    const Matrix& canonical_coefficients, const Matrix& overlap,
    const std::vector<std::vector<std::size_t>>& centers,
    LocalizationParallelOptions parallel) {
  const std::size_t nao = canonical_coefficients.rows();
  std::vector<bool> seen(nao, false);
  if(centers.empty()) {
    throw std::invalid_argument(
        "Pipek-Mezey localization requires at least one AO center");
  }
  for(const auto& center : centers) {
    if(center.empty()) {
      throw std::invalid_argument(
          "Pipek-Mezey AO centers may not be empty");
    }
    for(const std::size_t ao : center) {
      if(ao >= nao || seen[ao]) {
        throw std::invalid_argument(
            "Pipek-Mezey AO centers must be a disjoint AO partition");
      }
      seen[ao] = true;
    }
  }
  if(std::find(seen.begin(), seen.end(), false) != seen.end()) {
    throw std::invalid_argument(
        "Pipek-Mezey AO centers must include every AO");
  }

  const Matrix lowdin = linalg::multiply(
      symmetric_square_root(overlap), canonical_coefficients);
  const std::size_t n = canonical_coefficients.columns();
  std::vector<Matrix> populations;
  populations.reserve(centers.size());
  for(const auto& center : centers) {
    Matrix matrix{n, n};
    for(const std::size_t ao : center) {
      if(ao % parallel.ranks != parallel.rank) {
        continue;
      }
      for(std::size_t j = 0; j < n; ++j) {
        for(std::size_t i = 0; i < n; ++i) {
          matrix(i, j) += lowdin(ao, i) * lowdin(ao, j);
        }
      }
    }
    populations.push_back(std::move(matrix));
  }

  if(parallel.ranks > 1) {
    if(n != 0 && populations.size() >
                     std::numeric_limits<std::size_t>::max() / n / n) {
      throw std::overflow_error(
          "distributed Pipek-Mezey population buffer overflows size_t");
    }
    std::vector<double> packed;
    packed.reserve(populations.size() * n * n);
    for(const Matrix& matrix : populations) {
      packed.insert(packed.end(), matrix.values().begin(),
                    matrix.values().end());
    }
    sum_parallel(packed, parallel);
    std::size_t offset = 0;
    for(Matrix& matrix : populations) {
      std::copy_n(packed.begin() + static_cast<std::ptrdiff_t>(offset),
                  matrix.size(), matrix.values().begin());
      offset += matrix.size();
    }
  }

  return [populations = std::move(populations),
          parallel = std::move(parallel)](
             const Matrix& rotation,
             bool gradient_required) -> ObjectiveEvaluation {
    const std::size_t n = rotation.columns();
    Matrix objective_gradient{n, n};
    std::vector<double> objective_terms(n, 0.0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for(std::ptrdiff_t raw_orbital = 0;
        raw_orbital < static_cast<std::ptrdiff_t>(n); ++raw_orbital) {
      const auto orbital = static_cast<std::size_t>(raw_orbital);
      if(orbital % parallel.ranks != parallel.rank) {
        continue;
      }
      long double orbital_objective = 0.0L;
      for(const Matrix& center : populations) {
        const double population =
            quadratic_column(center, rotation, orbital);
        orbital_objective += static_cast<long double>(population) *
                             static_cast<long double>(population);
        if(gradient_required) {
          add_matvec_column(objective_gradient, orbital, center, rotation,
                            4.0 * population);
        }
      }
      objective_terms[orbital] = static_cast<double>(orbital_objective);
    }
    long double objective = 0.0L;
    for(const auto term : objective_terms)
      objective += static_cast<long double>(term);
    std::vector<double> reduced(
        1 + (gradient_required ? objective_gradient.size() : 0), 0.0);
    reduced[0] = static_cast<double>(objective);
    if(gradient_required) {
      std::copy(objective_gradient.values().begin(),
                objective_gradient.values().end(), reduced.begin() + 1);
    }
    sum_parallel(reduced, parallel);
    const double objective_double = reduced[0];
    if(gradient_required) {
      std::copy(reduced.begin() + 1, reduced.end(),
                objective_gradient.values().begin());
    }
    if(!std::isfinite(objective_double)) {
      throw std::runtime_error(
          "Pipek-Mezey localization objective became nonfinite");
    }
    Matrix loss_gradient =
        linalg::scaled(objective_gradient, -1.0);
    return ObjectiveEvaluation{
        .objective = objective_double,
        .loss = -objective_double,
        .loss_gradient = std::move(loss_gradient),
    };
  };
}

[[nodiscard]] double gradient_check(
    const ObjectiveFunction& objective, const Matrix& rotation,
    const TrustRegionLocalizationOptions& options) {
  const ObjectiveEvaluation at_origin = objective(rotation, true);
  const Matrix gradient =
      tangent_gradient(rotation, at_origin.loss_gradient);
  if(rotation.rows() == 1) {
    return 0.0;
  }
  std::mt19937_64 generator{options.gradient_check_seed};
  std::normal_distribution<double> normal{0.0, 1.0};
  double maximum_error = 0.0;
  for(int direction_index = 0;
      direction_index < options.gradient_check_directions;
      ++direction_index) {
    Matrix direction{rotation.rows(), rotation.columns()};
    for(std::size_t j = 0; j < direction.columns(); ++j) {
      for(std::size_t i = 0; i < j; ++i) {
        const double value = normal(generator);
        direction(i, j) = value;
        direction(j, i) = -value;
      }
    }
    const double norm = tangent_norm(direction);
    if(norm == 0.0) {
      throw std::runtime_error(
          "gradient check generated a zero tangent direction");
    }
    direction = linalg::scaled(direction, 1.0 / norm);
    const double step = options.finite_difference_step;
    const double plus =
        objective(retract(rotation, linalg::scaled(direction, step)),
                  false)
            .loss;
    const double minus =
        objective(retract(rotation, linalg::scaled(direction, -step)),
                  false)
            .loss;
    const double numerical = (plus - minus) / (2.0 * step);
    const double analytic = tangent_inner(gradient, direction);
    // The max(1, |d_num|, |d_analytic|) convention makes this a relative
    // check for large derivatives and a meaningful absolute check near a
    // stationary point, where cancellation dominates the central difference.
    const double denominator =
        std::max({1.0, std::abs(numerical), std::abs(analytic)});
    const double relative_error =
        std::abs(numerical - analytic) / denominator;
    if(!std::isfinite(relative_error)) {
      throw std::runtime_error(
          "localization gradient check became nonfinite");
    }
    maximum_error = std::max(maximum_error, relative_error);
  }
  if(maximum_error > options.gradient_check_tolerance) {
    throw std::runtime_error(
        "localization analytic gradient check failed: relative error " +
        std::to_string(maximum_error) + " exceeds tolerance " +
        std::to_string(options.gradient_check_tolerance));
  }
  return maximum_error;
}

[[nodiscard]] Matrix transport_to_base(const Matrix& polar_factor,
                                       const Matrix& tangent_at_trial) {
  return skew(linalg::multiply(polar_factor, tangent_at_trial));
}

[[nodiscard]] Matrix hessian_action(
    const ObjectiveFunction& objective, const Matrix& rotation,
    const Matrix& direction, double finite_difference_step) {
  const double norm = tangent_norm(direction);
  if(norm == 0.0) {
    return Matrix{direction.rows(), direction.columns()};
  }
  const double step = finite_difference_step / std::max(1.0, norm);
  const Matrix plus_tangent = linalg::scaled(direction, step);
  const Matrix minus_tangent = linalg::scaled(direction, -step);
  const Matrix plus_factor = polar_factor_from_tangent(plus_tangent);
  const Matrix minus_factor = polar_factor_from_tangent(minus_tangent);
  const Matrix plus_rotation = linalg::multiply(rotation, plus_factor);
  const Matrix minus_rotation = linalg::multiply(rotation, minus_factor);
  const Matrix plus_gradient = tangent_gradient(
      plus_rotation, objective(plus_rotation, true).loss_gradient);
  const Matrix minus_gradient = tangent_gradient(
      minus_rotation, objective(minus_rotation, true).loss_gradient);
  return linalg::scaled(
      linalg::subtract(
          transport_to_base(plus_factor, plus_gradient),
          transport_to_base(minus_factor, minus_gradient)),
      1.0 / (2.0 * step));
}

[[nodiscard]] double boundary_intersection(
    const Matrix& current, const Matrix& direction, double radius) {
  const double aa = tangent_inner(direction, direction);
  const double ab = tangent_inner(current, direction);
  const double cc =
      tangent_inner(current, current) - radius * radius;
  const double discriminant =
      std::max(0.0, ab * ab - aa * cc);
  if(aa <= 0.0) {
    throw std::runtime_error(
        "trust-region boundary direction has zero norm");
  }
  return (-ab + std::sqrt(discriminant)) / aa;
}

struct CgResult {
  Matrix step;
  bool reached_boundary;
  bool negative_curvature;
};

[[nodiscard]] CgResult truncated_conjugate_gradient(
    const ObjectiveFunction& objective, const Matrix& rotation,
    const Matrix& gradient, double radius,
    double finite_difference_step) {
  Matrix step{gradient.rows(), gradient.columns()};
  Matrix residual = gradient;
  Matrix direction = linalg::scaled(residual, -1.0);
  double residual_squared = tangent_inner(residual, residual);
  const double gradient_norm = std::sqrt(residual_squared);
  const double target =
      std::min(0.5, std::sqrt(std::max(gradient_norm, 0.0))) *
      gradient_norm;
  const std::size_t coordinates =
      gradient.rows() * (gradient.rows() - 1) / 2;
  const std::size_t maximum_cg_iterations =
      std::max<std::size_t>(1, std::min<std::size_t>(coordinates, 100));
  for(std::size_t iteration = 0;
      iteration < maximum_cg_iterations; ++iteration) {
    const Matrix hessian_direction =
        hessian_action(objective, rotation, direction,
                       finite_difference_step);
    const double curvature =
        tangent_inner(direction, hessian_direction);
    if(curvature <= 0.0 || !std::isfinite(curvature)) {
      const double tau =
          boundary_intersection(step, direction, radius);
      return CgResult{
          .step = linalg::add(step,
                              linalg::scaled(direction, tau)),
          .reached_boundary = true,
          .negative_curvature = true,
      };
    }
    const double alpha = residual_squared / curvature;
    const Matrix candidate =
        linalg::add(step, linalg::scaled(direction, alpha));
    if(tangent_norm(candidate) >= radius) {
      const double tau =
          boundary_intersection(step, direction, radius);
      return CgResult{
          .step = linalg::add(step,
                              linalg::scaled(direction, tau)),
          .reached_boundary = true,
          .negative_curvature = false,
      };
    }
    step = candidate;
    const Matrix next_residual =
        linalg::add(residual,
                    linalg::scaled(hessian_direction, alpha));
    const double next_residual_squared =
        tangent_inner(next_residual, next_residual);
    if(std::sqrt(std::max(0.0, next_residual_squared)) <= target) {
      return CgResult{
          .step = std::move(step),
          .reached_boundary = false,
          .negative_curvature = false,
      };
    }
    const double beta = next_residual_squared / residual_squared;
    direction = linalg::add(
        linalg::scaled(next_residual, -1.0),
        linalg::scaled(direction, beta));
    residual = next_residual;
    residual_squared = next_residual_squared;
  }
  return CgResult{
      .step = std::move(step),
      .reached_boundary = false,
      .negative_curvature = false,
  };
}

[[nodiscard]] LocalizationResult optimize(
    const Matrix& canonical_coefficients, const Matrix& overlap,
    LocalizationMethod method, int fm_power,
    const ObjectiveFunction& objective,
    const TrustRegionLocalizationOptions& options,
    const MomentObjective* moments) {
  Matrix rotation =
      initial_rotation(canonical_coefficients.columns(), options);
  const double checked_gradient_error =
      gradient_check(objective, rotation, options);
  ObjectiveEvaluation current = objective(rotation, true);
  Matrix gradient =
      tangent_gradient(rotation, current.loss_gradient);
  const double initial_gradient_norm = tangent_norm(gradient);
  double initial_objective = current.objective;
  std::vector<double> objective_history{initial_objective};
  double radius = options.initial_trust_radius;
  int accepted_steps = 0;
  int rejected_steps = 0;
  int iterations = 0;
  if(options.restart_state) {
    const auto& saved = *options.restart_state;
    const double saved_objective = saved.objective_history.back();
    const double mismatch = std::abs(current.objective - saved_objective);
    const double tolerance =
        1.0e-9 * std::max(1.0, std::abs(saved_objective));
    if(mismatch > tolerance) {
      throw std::runtime_error(
          "localization restart objective does not match its saved rotation");
    }
    initial_objective = saved.objective_history.front();
    objective_history = saved.objective_history;
    objective_history.back() = current.objective;
    radius = saved.trust_radius;
    accepted_steps = saved.accepted_steps;
    rejected_steps = saved.rejected_steps;
    iterations = saved.completed_iterations;
  }
  bool converged = false;
  std::string reason = "maximum_iterations";

  if(initial_gradient_norm <= options.gradient_tolerance) {
    converged = true;
    reason = "gradient_tolerance";
  }

  const auto report = [&](const bool step_accepted) {
    if(options.progress) {
      options.progress(
          LocalizationIteration{
              .iteration = iterations,
              .accepted_steps = accepted_steps,
              .rejected_steps = rejected_steps,
              .step_accepted = step_accepted,
              .converged = converged,
              .objective = current.objective,
              .gradient_norm = tangent_norm(gradient),
              .trust_radius = radius,
              .convergence_reason = converged ? reason : std::string{},
          },
          rotation, objective_history);
    }
  };
  report(false);

  while(!converged && iterations < options.maximum_iterations) {
    ++iterations;
    CgResult cg = truncated_conjugate_gradient(
        objective, rotation, gradient, radius,
        options.finite_difference_step);
    Matrix step = std::move(cg.step);
    Matrix hessian_step = hessian_action(
        objective, rotation, step, options.finite_difference_step);
    double predicted_reduction =
        -(tangent_inner(gradient, step) +
          0.5 * tangent_inner(step, hessian_step));
    if(!std::isfinite(predicted_reduction) ||
       predicted_reduction <= 0.0) {
      const double norm = tangent_norm(gradient);
      step = linalg::scaled(gradient, -radius / norm);
      hessian_step = hessian_action(
          objective, rotation, step, options.finite_difference_step);
      predicted_reduction =
          -(tangent_inner(gradient, step) +
            0.5 * tangent_inner(step, hessian_step));
    }

    const Matrix trial_rotation = retract(rotation, step);
    const ObjectiveEvaluation trial =
        objective(trial_rotation, true);
    const double actual_reduction = current.loss - trial.loss;
    const double ratio =
        predicted_reduction > 0.0
            ? actual_reduction / predicted_reduction
            : -std::numeric_limits<double>::infinity();

    if(ratio < 0.25 || !std::isfinite(ratio)) {
      radius *= 0.25;
    } else if(ratio > 0.75 && cg.reached_boundary) {
      radius = std::min(2.0 * radius,
                        options.maximum_trust_radius);
    }

    bool step_accepted = false;
    if(actual_reduction > 0.0 &&
       ratio > options.acceptance_threshold) {
      const double previous_loss = current.loss;
      rotation = trial_rotation;
      if(maximum_identity_error(rotation) >
         options.invariant_tolerance) {
        rotation = polar_orthogonalize(rotation);
      }
      current = objective(rotation, true);
      gradient =
          tangent_gradient(rotation, current.loss_gradient);
      ++accepted_steps;
      step_accepted = true;
      objective_history.push_back(current.objective);
      const double gradient_norm = tangent_norm(gradient);
      if(gradient_norm <= options.gradient_tolerance) {
        converged = true;
        reason = "gradient_tolerance";
      } else if(std::abs(previous_loss - current.loss) <=
                options.objective_tolerance *
                    std::max(1.0, std::abs(previous_loss))) {
        converged = true;
        reason = "objective_tolerance";
      }
    } else {
      ++rejected_steps;
    }

    if(!converged && radius < options.minimum_trust_radius) {
      reason = "trust_radius_too_small";
      report(step_accepted);
      break;
    }
    report(step_accepted);
  }

  const double final_gradient_norm = tangent_norm(gradient);
  Matrix localized =
      linalg::multiply(canonical_coefficients, rotation);
  const double rotation_error = maximum_identity_error(rotation);
  const double orbital_error =
      metric_orthonormality_error(localized, overlap);
  const double preserved_projector_error =
      projector_error(canonical_coefficients, localized);
  if(rotation_error > options.invariant_tolerance ||
     orbital_error > options.invariant_tolerance ||
     preserved_projector_error > options.invariant_tolerance) {
    throw std::runtime_error(
        "localized orbitals violate orthogonality or projector invariants");
  }

  std::vector<OrbitalSpatialDiagnostic> diagnostics;
  if(moments != nullptr) {
    diagnostics.resize(localized.columns());
    const Matrix identity = Matrix::identity(localized.columns());
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for(std::ptrdiff_t raw_orbital = 0;
        raw_orbital < static_cast<std::ptrdiff_t>(localized.columns());
        ++raw_orbital) {
      const auto orbital = static_cast<std::size_t>(raw_orbital);
      const SpatialValues values =
          spatial_values(*moments, rotation, orbital);
      diagnostics[orbital] = OrbitalSpatialDiagnostic{
          .orbital = orbital,
          .centroid_bohr = values.centroid,
          .omega2_bohr2 = values.omega2,
          .sigma2_bohr = std::sqrt(values.omega2),
          .omega4_bohr4 = values.omega4,
          .sigma4_bohr = std::sqrt(std::sqrt(values.omega4)),
      };
    }
  }
  const double improvement =
      method == LocalizationMethod::pipek_mezey
          ? current.objective - initial_objective
          : initial_objective - current.objective;
  return LocalizationResult{
      .coefficients = std::move(localized),
      .rotation = std::move(rotation),
      .method = method,
      .fm_power = method == LocalizationMethod::fourth_moment
                      ? fm_power
                      : 0,
      .objective_initial = initial_objective,
      .objective_final = current.objective,
      .absolute_objective_reduction = improvement,
      .relative_objective_reduction =
          improvement / std::max(1.0, std::abs(initial_objective)),
      .gradient_norm_initial = initial_gradient_norm,
      .gradient_norm_final = final_gradient_norm,
      .iterations = iterations,
      .accepted_steps = accepted_steps,
      .rejected_steps = rejected_steps,
      .converged = converged,
      .convergence_reason = std::move(reason),
      .final_trust_radius = radius,
      .rotation_orthogonality_maximum_error = rotation_error,
      .orbital_orthonormality_maximum_error = orbital_error,
      .projector_maximum_error = preserved_projector_error,
      .gradient_check_maximum_relative_error =
          checked_gradient_error,
      .objective_history = std::move(objective_history),
      .diagnostics = std::move(diagnostics),
  };
}

[[nodiscard]] std::filesystem::path output_path(
    const std::filesystem::path& prefix, const char* suffix) {
  return std::filesystem::path{prefix.string() + suffix};
}

void require_writable_outputs(
    const std::array<std::filesystem::path, 4>& paths,
    bool overwrite) {
  for(const auto& path : paths) {
    if(std::filesystem::exists(path) && !overwrite) {
      throw std::runtime_error(
          "localization output already exists: " + path.string());
    }
    const std::filesystem::path parent = path.parent_path();
    if(!parent.empty() && !std::filesystem::exists(parent)) {
      throw std::runtime_error(
          "localization output directory does not exist: " +
          parent.string());
    }
  }
}

void write_little_endian_u16(std::ostream& stream, std::uint16_t value) {
  const std::array<char, 2> bytes{
      static_cast<char>(value & 0xffU),
      static_cast<char>((value >> 8U) & 0xffU),
  };
  stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void write_little_endian_double(std::ostream& stream, double value) {
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
  std::array<char, 8> bytes{};
  for(std::size_t index = 0; index < bytes.size(); ++index) {
    bytes[index] =
        static_cast<char>((bits >> (8U * index)) & 0xffU);
  }
  stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void write_numpy_matrix(const std::filesystem::path& path,
                        const Matrix& matrix) {
  if(matrix.empty()) {
    throw std::invalid_argument("cannot save an empty localization matrix");
  }
  std::ofstream stream{path, std::ios::binary | std::ios::trunc};
  if(!stream) {
    throw std::runtime_error(
        "failed to open localization matrix output: " + path.string());
  }
  constexpr std::array<char, 6> magic{
      static_cast<char>(0x93), 'N', 'U', 'M', 'P', 'Y'};
  stream.write(magic.data(), static_cast<std::streamsize>(magic.size()));
  stream.put(static_cast<char>(1));
  stream.put(static_cast<char>(0));
  std::ostringstream header_builder;
  header_builder
      << "{'descr': '<f8', 'fortran_order': True, 'shape': ("
      << matrix.rows() << ", " << matrix.columns() << "), }";
  std::string header = header_builder.str();
  const std::size_t preamble = 10;
  const std::size_t padding =
      (16 - ((preamble + header.size() + 1) % 16)) % 16;
  header.append(padding, ' ');
  header.push_back('\n');
  if(header.size() >
     static_cast<std::size_t>(
         std::numeric_limits<std::uint16_t>::max())) {
    throw std::overflow_error("NumPy localization header is too large");
  }
  write_little_endian_u16(
      stream, static_cast<std::uint16_t>(header.size()));
  stream.write(header.data(),
               static_cast<std::streamsize>(header.size()));
  for(const double value : matrix.values()) {
    write_little_endian_double(stream, value);
  }
  if(!stream) {
    throw std::runtime_error(
        "failed while writing localization matrix: " + path.string());
  }
}

[[nodiscard]] const char* method_name(LocalizationMethod method) {
  return method == LocalizationMethod::pipek_mezey
             ? "pipek-mezey"
             : "fm";
}

}  // namespace

LocalizationResult localize_pipek_mezey_trust_region(
    const Matrix& canonical_coefficients, const Matrix& overlap_input,
    const std::vector<std::vector<std::size_t>>& center_ao_indices,
    const TrustRegionLocalizationOptions& options) {
  Matrix overlap = overlap_input;
  validate_common_inputs(canonical_coefficients, overlap, options);
  const ObjectiveFunction objective = make_pm_objective(
      canonical_coefficients, overlap, center_ao_indices,
      options.parallel);
  return optimize(canonical_coefficients, overlap,
                  LocalizationMethod::pipek_mezey, 0, objective,
                  options, nullptr);
}

linalg::Matrix lowdin_center_populations(
    const linalg::Matrix& coefficients, const linalg::Matrix& overlap,
    const std::vector<std::vector<std::size_t>>& center_ao_indices) {
  Matrix checked_overlap = overlap;
  validate_common_inputs(coefficients, checked_overlap,
                         TrustRegionLocalizationOptions{});
  const std::size_t nao = coefficients.rows();
  std::vector<bool> seen(nao, false);
  if(center_ao_indices.empty())
    throw std::invalid_argument("Lowdin populations require AO centers");
  for(const auto& center : center_ao_indices) {
    if(center.empty())
      throw std::invalid_argument("Lowdin population center may not be empty");
    for(const auto ao : center) {
      if(ao >= nao || seen[ao])
        throw std::invalid_argument(
            "Lowdin AO centers must be a disjoint AO partition");
      seen[ao] = true;
    }
  }
  if(std::find(seen.begin(), seen.end(), false) != seen.end())
    throw std::invalid_argument("Lowdin AO centers must include every AO");
  const Matrix lowdin =
      linalg::multiply(symmetric_square_root(checked_overlap), coefficients);
  Matrix result{center_ao_indices.size(), coefficients.columns()};
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for(std::ptrdiff_t raw_orbital = 0;
      raw_orbital < static_cast<std::ptrdiff_t>(coefficients.columns());
      ++raw_orbital) {
    const auto orbital = static_cast<std::size_t>(raw_orbital);
    for(std::size_t center = 0; center < center_ao_indices.size(); ++center)
      for(const auto ao : center_ao_indices[center])
        result(center, orbital) += lowdin(ao, orbital) * lowdin(ao, orbital);
  }
  linalg::require_finite(result, "Lowdin center populations");
  return result;
}

LocalizationResult localize_fourth_moment(
    const Matrix& canonical_coefficients, const Matrix& overlap_input,
    const CartesianMomentIntegrals& moments_input, int power,
    const TrustRegionLocalizationOptions& options) {
  if(power < 1 || power > 4) {
    throw std::invalid_argument(
        "fourth-moment localization power must be in [1,4]");
  }
  Matrix overlap = overlap_input;
  validate_common_inputs(canonical_coefficients, overlap, options);
  const CartesianMomentIntegrals moments =
      validate_and_symmetrize_moments(
          moments_input, canonical_coefficients.rows());
  const auto transformed = std::make_shared<const MomentObjective>(
      transform_moments(canonical_coefficients, moments,
                        options.parallel));
  const ObjectiveFunction raw_objective =
      make_fm_objective(transformed, power, options.parallel);
  // FM4 values and derivatives can span many orders of magnitude. Scaling
  // only the trust-region loss (not the reported physical objective) makes
  // the finite-difference Hessian and acceptance ratio dimensionless without
  // changing the minimizer.
  const double loss_scale = std::max(
      1.0, std::abs(raw_objective(
                        Matrix::identity(canonical_coefficients.columns()),
                        false)
                        .loss));
  const ObjectiveFunction objective =
      [raw_objective, loss_scale](const Matrix& rotation,
                                  const bool gradient_required) {
        ObjectiveEvaluation result =
            raw_objective(rotation, gradient_required);
        result.loss /= loss_scale;
        result.loss_gradient =
            linalg::scaled(result.loss_gradient, 1.0 / loss_scale);
        return result;
      };
  return optimize(canonical_coefficients, overlap,
                  LocalizationMethod::fourth_moment, power,
                  objective, options, transformed.get());
}

LocalizationResult localize_orbitals(
    const Matrix& canonical_coefficients, const Matrix& overlap,
    LocalizationMethod method,
    const std::vector<std::vector<std::size_t>>& center_ao_indices,
    const CartesianMomentIntegrals* moments, int fm_power,
    const TrustRegionLocalizationOptions& options) {
  if(method == LocalizationMethod::pipek_mezey) {
    return localize_pipek_mezey_trust_region(
        canonical_coefficients, overlap, center_ao_indices, options);
  }
  if(moments == nullptr) {
    throw std::invalid_argument(
        "fourth-moment localization requires AO moment integrals");
  }
  return localize_fourth_moment(canonical_coefficients, overlap,
                               *moments, fm_power, options);
}

LocalizationResult localize_orbitals(
    const Matrix& canonical_coefficients, const Matrix& overlap,
    const std::vector<std::vector<std::size_t>>& center_ao_indices,
    const TrustRegionLocalizationOptions& options) {
  return localize_pipek_mezey_trust_region(
      canonical_coefficients, overlap, center_ao_indices, options);
}

std::vector<OrbitalSpatialDiagnostic> evaluate_spatial_diagnostics(
    const Matrix& coefficients,
    const CartesianMomentIntegrals& moments_input) {
  if(coefficients.empty()) {
    throw std::invalid_argument(
        "spatial diagnostics require nonempty orbital coefficients");
  }
  const CartesianMomentIntegrals moments =
      validate_and_symmetrize_moments(moments_input,
                                      coefficients.rows());
  const MomentObjective transformed =
      transform_moments(coefficients, moments,
                        LocalizationParallelOptions{});
  const Matrix identity = Matrix::identity(coefficients.columns());
  std::vector<OrbitalSpatialDiagnostic> result;
  result.reserve(coefficients.columns());
  for(std::size_t orbital = 0; orbital < coefficients.columns();
      ++orbital) {
    const SpatialValues values =
        spatial_values(transformed, identity, orbital);
    result.push_back(OrbitalSpatialDiagnostic{
        .orbital = orbital,
        .centroid_bohr = values.centroid,
        .omega2_bohr2 = values.omega2,
        .sigma2_bohr = std::sqrt(values.omega2),
        .omega4_bohr4 = values.omega4,
        .sigma4_bohr = std::sqrt(std::sqrt(values.omega4)),
    });
  }
  return result;
}

void save_localization_outputs(const LocalizationResult& result,
                               const std::filesystem::path& prefix,
                               bool overwrite) {
  if(prefix.empty()) {
    throw std::invalid_argument(
        "localization output prefix may not be empty");
  }
  const std::array<std::filesystem::path, 4> paths{
      output_path(prefix, "_C_localized.npy"),
      output_path(prefix, "_U_localization.npy"),
      output_path(prefix, "_orbital_diagnostics.csv"),
      output_path(prefix, "_localization_summary.json"),
  };
  require_writable_outputs(paths, overwrite);
  write_numpy_matrix(paths[0], result.coefficients);
  write_numpy_matrix(paths[1], result.rotation);

  {
    std::ofstream stream{paths[2], std::ios::trunc};
    if(!stream) {
      throw std::runtime_error(
          "failed to open localization diagnostics: " +
          paths[2].string());
    }
    stream << "orbital,centroid_x_bohr,centroid_y_bohr,"
              "centroid_z_bohr,omega2_bohr2,sigma2_bohr,"
              "omega4_bohr4,sigma4_bohr\n";
    stream << std::setprecision(17);
    for(const OrbitalSpatialDiagnostic& diagnostic :
        result.diagnostics) {
      stream << diagnostic.orbital << ','
             << diagnostic.centroid_bohr[0] << ','
             << diagnostic.centroid_bohr[1] << ','
             << diagnostic.centroid_bohr[2] << ','
             << diagnostic.omega2_bohr2 << ','
             << diagnostic.sigma2_bohr << ','
             << diagnostic.omega4_bohr4 << ','
             << diagnostic.sigma4_bohr << '\n';
    }
    if(!stream) {
      throw std::runtime_error(
          "failed while writing localization diagnostics: " +
          paths[2].string());
    }
  }

  {
    std::ofstream stream{paths[3], std::ios::trunc};
    if(!stream) {
      throw std::runtime_error(
          "failed to open localization summary: " +
          paths[3].string());
    }
    double sigma2_minimum = 0.0;
    double sigma2_mean = 0.0;
    double sigma2_maximum = 0.0;
    double sigma4_minimum = 0.0;
    double sigma4_mean = 0.0;
    double sigma4_maximum = 0.0;
    std::size_t largest_sigma4_orbital = 0;
    if(!result.diagnostics.empty()) {
      sigma2_minimum = result.diagnostics.front().sigma2_bohr;
      sigma2_maximum = sigma2_minimum;
      sigma4_minimum = result.diagnostics.front().sigma4_bohr;
      sigma4_maximum = sigma4_minimum;
      long double sigma2_sum = 0.0L;
      long double sigma4_sum = 0.0L;
      for(const auto& diagnostic : result.diagnostics) {
        sigma2_minimum =
            std::min(sigma2_minimum, diagnostic.sigma2_bohr);
        sigma2_maximum =
            std::max(sigma2_maximum, diagnostic.sigma2_bohr);
        sigma4_minimum =
            std::min(sigma4_minimum, diagnostic.sigma4_bohr);
        if(diagnostic.sigma4_bohr > sigma4_maximum) {
          sigma4_maximum = diagnostic.sigma4_bohr;
          largest_sigma4_orbital = diagnostic.orbital;
        }
        sigma2_sum += diagnostic.sigma2_bohr;
        sigma4_sum += diagnostic.sigma4_bohr;
      }
      sigma2_mean = static_cast<double>(
          sigma2_sum /
          static_cast<long double>(result.diagnostics.size()));
      sigma4_mean = static_cast<double>(
          sigma4_sum /
          static_cast<long double>(result.diagnostics.size()));
    }
    stream << std::setprecision(17);
    stream << "{\n"
           << "  \"method\": \"" << method_name(result.method)
           << "\",\n"
           << "  \"fm_power\": " << result.fm_power << ",\n"
           << "  \"objective_initial\": "
           << result.objective_initial << ",\n"
           << "  \"objective_final\": "
           << result.objective_final << ",\n"
           << "  \"absolute_objective_reduction\": "
           << result.absolute_objective_reduction << ",\n"
           << "  \"relative_objective_reduction\": "
           << result.relative_objective_reduction << ",\n"
           << "  \"gradient_norm_initial\": "
           << result.gradient_norm_initial << ",\n"
           << "  \"gradient_norm_final\": "
           << result.gradient_norm_final << ",\n"
           << "  \"iterations\": " << result.iterations << ",\n"
           << "  \"accepted_steps\": " << result.accepted_steps
           << ",\n"
           << "  \"rejected_steps\": " << result.rejected_steps
           << ",\n"
           << "  \"converged\": "
           << (result.converged ? "true" : "false") << ",\n"
           << "  \"convergence_reason\": \""
           << result.convergence_reason << "\",\n"
           << "  \"final_trust_radius\": "
           << result.final_trust_radius << ",\n"
           << "  \"rotation_orthogonality_maximum_error\": "
           << result.rotation_orthogonality_maximum_error << ",\n"
           << "  \"orbital_orthonormality_maximum_error\": "
           << result.orbital_orthonormality_maximum_error << ",\n"
           << "  \"projector_maximum_error\": "
           << result.projector_maximum_error << ",\n"
           << "  \"gradient_check_maximum_relative_error\": "
           << result.gradient_check_maximum_relative_error << ",\n"
           << "  \"spatial_diagnostics_available\": "
           << (!result.diagnostics.empty() ? "true" : "false")
           << ",\n";
    if(result.diagnostics.empty()) {
      stream << "  \"sigma2_bohr\": null,\n"
             << "  \"sigma4_bohr\": null,\n"
             << "  \"largest_sigma4_orbital\": null\n";
    } else {
      stream << "  \"sigma2_bohr\": {\"minimum\": "
             << sigma2_minimum << ", \"mean\": " << sigma2_mean
             << ", \"maximum\": " << sigma2_maximum << "},\n"
             << "  \"sigma4_bohr\": {\"minimum\": "
             << sigma4_minimum << ", \"mean\": " << sigma4_mean
             << ", \"maximum\": " << sigma4_maximum << "},\n"
             << "  \"largest_sigma4_orbital\": "
             << largest_sigma4_orbital << '\n';
    }
    stream << "}\n";
    if(!stream) {
      throw std::runtime_error(
          "failed while writing localization summary: " +
          paths[3].string());
    }
  }
}

}  // namespace modernqc::localization
