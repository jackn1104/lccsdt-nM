#pragma once

#include <cstddef>
#include <vector>

namespace modernqc::linalg {

class Matrix {
 public:
  Matrix() = default;
  Matrix(std::size_t rows, std::size_t columns, double value = 0.0);

  [[nodiscard]] std::size_t rows() const noexcept;
  [[nodiscard]] std::size_t columns() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] bool empty() const noexcept;

  [[nodiscard]] double* data() noexcept;
  [[nodiscard]] const double* data() const noexcept;
  [[nodiscard]] std::vector<double>& values() noexcept;
  [[nodiscard]] const std::vector<double>& values() const noexcept;

  [[nodiscard]] double& operator()(std::size_t row, std::size_t column);
  [[nodiscard]] const double& operator()(std::size_t row,
                                         std::size_t column) const;

  [[nodiscard]] static Matrix identity(std::size_t dimension);

 private:
  std::size_t rows_{0};
  std::size_t columns_{0};
  std::vector<double> values_;
};

[[nodiscard]] Matrix transpose(const Matrix& matrix);
[[nodiscard]] Matrix multiply(const Matrix& left, const Matrix& right);
[[nodiscard]] Matrix add(const Matrix& left, const Matrix& right);
[[nodiscard]] Matrix subtract(const Matrix& left, const Matrix& right);
[[nodiscard]] Matrix scaled(const Matrix& matrix, double factor);
[[nodiscard]] double frobenius_norm(const Matrix& matrix);
[[nodiscard]] double maximum_absolute_value(const Matrix& matrix);
[[nodiscard]] double maximum_asymmetry(const Matrix& matrix);
void symmetrize_in_place(Matrix& matrix);
void require_finite(const Matrix& matrix, const char* name);

}  // namespace modernqc::linalg
