#include "localized_cc/diis_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace localized_cc {
namespace {

std::optional<std::vector<double>> solve_augmented(
    const std::vector<std::vector<double>>& products,
    const std::size_t first, const double relative_shift) {
  const auto count = products.size() - first;
  if (count < 2)
    return std::nullopt;

  double scale = 0.0;
  for (std::size_t i = first; i < products.size(); ++i)
    for (std::size_t j = first; j < products.size(); ++j)
      scale = std::max(scale, std::abs(products[i][j]));
  if (!(scale > 0.0) || !std::isfinite(scale))
    return std::nullopt;

  const auto dimension = count + 1;
  std::vector<std::vector<double>> matrix(
      dimension, std::vector<double>(dimension));
  std::vector<double> right_hand_side(dimension);
  right_hand_side[count] = -1.0;
  for (std::size_t i = 0; i < count; ++i)
    for (std::size_t j = 0; j < count; ++j) {
      const auto left = products[first + i][first + j];
      const auto right = products[first + j][first + i];
      matrix[i][j] = 0.5 * (left + right) / scale;
    }
  for (std::size_t i = 0; i < count; ++i) {
    matrix[i][i] += relative_shift;
    matrix[i][count] = -1.0;
    matrix[count][i] = -1.0;
  }

  const auto pivot_tolerance =
      256.0 * std::numeric_limits<double>::epsilon() *
      static_cast<double>(dimension);
  for (std::size_t column = 0; column < dimension; ++column) {
    auto pivot = column;
    for (std::size_t row = column + 1; row < dimension; ++row)
      if (std::abs(matrix[row][column]) >
          std::abs(matrix[pivot][column]))
        pivot = row;
    if (!std::isfinite(matrix[pivot][column]) ||
        std::abs(matrix[pivot][column]) <= pivot_tolerance)
      return std::nullopt;
    std::swap(matrix[column], matrix[pivot]);
    std::swap(right_hand_side[column], right_hand_side[pivot]);
    const auto diagonal = matrix[column][column];
    for (std::size_t j = column; j < dimension; ++j)
      matrix[column][j] /= diagonal;
    right_hand_side[column] /= diagonal;
    for (std::size_t row = 0; row < dimension; ++row) {
      if (row == column)
        continue;
      const auto multiplier = matrix[row][column];
      for (std::size_t j = column; j < dimension; ++j)
        matrix[row][j] -= multiplier * matrix[column][j];
      right_hand_side[row] -= multiplier * right_hand_side[column];
    }
  }

  right_hand_side.resize(count);
  double sum = 0.0;
  double maximum = 0.0;
  for (const auto coefficient : right_hand_side) {
    if (!std::isfinite(coefficient))
      return std::nullopt;
    sum += coefficient;
    maximum = std::max(maximum, std::abs(coefficient));
  }
  if (!std::isfinite(sum) || std::abs(sum) <= pivot_tolerance ||
      maximum > 1.0e6)
    return std::nullopt;
  for (auto& coefficient : right_hand_side)
    coefficient /= sum;
  return right_hand_side;
}

} // namespace

std::optional<DiisCoefficientSolution> solve_diis_coefficients(
    const std::vector<std::vector<double>>& error_products) {
  const auto count = error_products.size();
  if (count < 2)
    return std::nullopt;
  for (const auto& row : error_products) {
    if (row.size() != count)
      throw std::invalid_argument("DIIS error-product matrix is not square");
    for (const auto value : row)
      if (!std::isfinite(value))
        throw std::invalid_argument(
            "DIIS error-product matrix contains a non-finite value");
  }

  // Retain as much recent history as the unregularized Pulay system can
  // support. Oldest vectors are removed first because the newest residuals
  // best represent the current local nonlinear model.
  for (std::size_t first = 0; first + 1 < count; ++first)
    if (auto coefficients = solve_augmented(error_products, first, 0.0))
      return DiisCoefficientSolution{first, std::move(*coefficients)};

  // Exact duplicate residuals make every unregularized suffix singular. A
  // relative shift on only the newest pair supplies a deterministic bounded
  // fallback without reintroducing the old absolute-scale failure mode.
  constexpr double relative_shift = 1.0e-12;
  const auto first = count - 2;
  if (auto coefficients =
          solve_augmented(error_products, first, relative_shift))
    return DiisCoefficientSolution{first, std::move(*coefficients)};
  return std::nullopt;
}

} // namespace localized_cc
