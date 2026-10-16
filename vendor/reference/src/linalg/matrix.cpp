#include "modernqc/linalg/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace modernqc::linalg {
namespace {

extern "C" {
void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb, const double* beta, double* c,
            const int* ldc);
}

[[nodiscard]] std::size_t checked_size(std::size_t rows,
                                       std::size_t columns) {
  if(rows != 0 && columns > std::numeric_limits<std::size_t>::max() / rows) {
    throw std::overflow_error("matrix allocation size overflow");
  }
  return rows * columns;
}

[[nodiscard]] int lapack_dimension(std::size_t value, const char* name) {
  if(value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string{name} +
                              " exceeds the BLAS integer range");
  }
  return static_cast<int>(value);
}

void require_same_shape(const Matrix& left, const Matrix& right,
                        const char* operation) {
  if(left.rows() != right.rows() || left.columns() != right.columns()) {
    throw std::invalid_argument(std::string{operation} +
                                " requires equal matrix dimensions");
  }
}

}  // namespace

Matrix::Matrix(std::size_t rows, std::size_t columns, double value)
    : rows_{rows},
      columns_{columns},
      values_(checked_size(rows, columns), value) {
  if((rows == 0) != (columns == 0)) {
    throw std::invalid_argument(
        "a matrix must have either two positive dimensions or be 0 x 0");
  }
  if(!std::isfinite(value)) {
    throw std::invalid_argument("matrix initialization value must be finite");
  }
}

std::size_t Matrix::rows() const noexcept { return rows_; }

std::size_t Matrix::columns() const noexcept { return columns_; }

std::size_t Matrix::size() const noexcept { return values_.size(); }

bool Matrix::empty() const noexcept { return values_.empty(); }

double* Matrix::data() noexcept { return values_.data(); }

const double* Matrix::data() const noexcept { return values_.data(); }

std::vector<double>& Matrix::values() noexcept { return values_; }

const std::vector<double>& Matrix::values() const noexcept { return values_; }

double& Matrix::operator()(std::size_t row, std::size_t column) {
  if(row >= rows_ || column >= columns_) {
    throw std::out_of_range("matrix index is out of range");
  }
  return values_[column * rows_ + row];
}

const double& Matrix::operator()(std::size_t row, std::size_t column) const {
  if(row >= rows_ || column >= columns_) {
    throw std::out_of_range("matrix index is out of range");
  }
  return values_[column * rows_ + row];
}

Matrix Matrix::identity(std::size_t dimension) {
  if(dimension == 0) {
    throw std::invalid_argument("identity matrix dimension must be positive");
  }
  Matrix result{dimension, dimension};
  for(std::size_t index = 0; index < dimension; ++index) {
    result(index, index) = 1.0;
  }
  return result;
}

Matrix transpose(const Matrix& matrix) {
  if(matrix.empty()) {
    return {};
  }
  Matrix result{matrix.columns(), matrix.rows()};
  for(std::size_t column = 0; column < matrix.columns(); ++column) {
    for(std::size_t row = 0; row < matrix.rows(); ++row) {
      result(column, row) = matrix(row, column);
    }
  }
  return result;
}

Matrix multiply(const Matrix& left, const Matrix& right) {
  if(left.empty() || right.empty() || left.columns() != right.rows()) {
    throw std::invalid_argument(
        "matrix multiplication dimensions are incompatible");
  }
  Matrix result{left.rows(), right.columns()};
  const int m = lapack_dimension(left.rows(), "left row count");
  const int n = lapack_dimension(right.columns(), "right column count");
  const int k = lapack_dimension(left.columns(), "contracted dimension");
  const int lda = m;
  const int ldb = lapack_dimension(right.rows(), "right row count");
  const int ldc = m;
  constexpr char no_transpose = 'N';
  constexpr double alpha = 1.0;
  constexpr double beta = 0.0;
  dgemm_(&no_transpose, &no_transpose, &m, &n, &k, &alpha, left.data(), &lda,
         right.data(), &ldb, &beta, result.data(), &ldc);
  require_finite(result, "matrix product");
  return result;
}

Matrix add(const Matrix& left, const Matrix& right) {
  require_same_shape(left, right, "matrix addition");
  Matrix result{left.rows(), left.columns()};
  for(std::size_t index = 0; index < result.size(); ++index) {
    result.values()[index] = left.values()[index] + right.values()[index];
  }
  require_finite(result, "matrix sum");
  return result;
}

Matrix subtract(const Matrix& left, const Matrix& right) {
  require_same_shape(left, right, "matrix subtraction");
  Matrix result{left.rows(), left.columns()};
  for(std::size_t index = 0; index < result.size(); ++index) {
    result.values()[index] = left.values()[index] - right.values()[index];
  }
  require_finite(result, "matrix difference");
  return result;
}

Matrix scaled(const Matrix& matrix, double factor) {
  if(!std::isfinite(factor)) {
    throw std::invalid_argument("matrix scale factor must be finite");
  }
  Matrix result{matrix.rows(), matrix.columns()};
  for(std::size_t index = 0; index < result.size(); ++index) {
    result.values()[index] = factor * matrix.values()[index];
  }
  require_finite(result, "scaled matrix");
  return result;
}

double frobenius_norm(const Matrix& matrix) {
  double scale = 0.0;
  double sum_squares = 1.0;
  for(const double value : matrix.values()) {
    if(value != 0.0) {
      const double magnitude = std::abs(value);
      if(scale < magnitude) {
        const double ratio = scale / magnitude;
        sum_squares = 1.0 + sum_squares * ratio * ratio;
        scale = magnitude;
      } else {
        const double ratio = magnitude / scale;
        sum_squares += ratio * ratio;
      }
    }
  }
  return scale == 0.0 ? 0.0 : scale * std::sqrt(sum_squares);
}

double maximum_absolute_value(const Matrix& matrix) {
  double result = 0.0;
  for(const double value : matrix.values()) {
    result = std::max(result, std::abs(value));
  }
  return result;
}

double maximum_asymmetry(const Matrix& matrix) {
  if(matrix.rows() != matrix.columns()) {
    throw std::invalid_argument("asymmetry requires a square matrix");
  }
  double result = 0.0;
  for(std::size_t column = 0; column < matrix.columns(); ++column) {
    for(std::size_t row = 0; row < column; ++row) {
      result =
          std::max(result, std::abs(matrix(row, column) - matrix(column, row)));
    }
  }
  return result;
}

void symmetrize_in_place(Matrix& matrix) {
  if(matrix.rows() != matrix.columns()) {
    throw std::invalid_argument("symmetrization requires a square matrix");
  }
  for(std::size_t column = 0; column < matrix.columns(); ++column) {
    for(std::size_t row = 0; row < column; ++row) {
      const double average =
          0.5 * (matrix(row, column) + matrix(column, row));
      matrix(row, column) = average;
      matrix(column, row) = average;
    }
  }
}

void require_finite(const Matrix& matrix, const char* name) {
  const auto invalid = std::find_if(
      matrix.values().begin(), matrix.values().end(),
      [](double value) { return !std::isfinite(value); });
  if(invalid != matrix.values().end()) {
    throw std::runtime_error(std::string{name} + " contains a nonfinite value");
  }
}

}  // namespace modernqc::linalg
