#include "modernqc/linalg/eigensolver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace modernqc::linalg {
namespace {

extern "C" {
void dsyevd_(const char* jobz, const char* uplo, const int* n, double* a,
             const int* lda, double* w, double* work, const int* lwork,
             int* iwork, const int* liwork, int* info);
void dgesv_(const int* n, const int* nrhs, double* a, const int* lda,
            int* ipiv, double* b, const int* ldb, int* info);
void dposv_(const char* uplo, const int* n, const int* nrhs, double* a,
            const int* lda, double* b, const int* ldb, int* info);
void dgelss_(const int* m, const int* n, const int* nrhs, double* a,
             const int* lda, double* b, const int* ldb, double* s,
             const double* rcond, int* rank, double* work, const int* lwork,
             int* info);
}

[[nodiscard]] int lapack_dimension(std::size_t value, const char* name) {
  if(value == 0 ||
     value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string{name} +
                              " is outside the LAPACK integer range");
  }
  return static_cast<int>(value);
}

}  // namespace

SymmetricEigendecomposition diagonalize_symmetric(
    const Matrix& matrix, double symmetry_tolerance) {
  if(matrix.empty() || matrix.rows() != matrix.columns()) {
    throw std::invalid_argument(
        "symmetric diagonalization requires a nonempty square matrix");
  }
  if(!std::isfinite(symmetry_tolerance) || symmetry_tolerance < 0.0) {
    throw std::invalid_argument(
        "eigensolver symmetry tolerance must be finite and nonnegative");
  }
  require_finite(matrix, "symmetric eigensolver input");
  if(maximum_asymmetry(matrix) > symmetry_tolerance) {
    throw std::invalid_argument(
        "symmetric eigensolver input exceeds the symmetry tolerance");
  }

  Matrix eigenvectors = matrix;
  symmetrize_in_place(eigenvectors);
  const int n = lapack_dimension(matrix.rows(), "matrix dimension");
  const int lda = n;
  std::vector<double> eigenvalues(matrix.rows());
  constexpr char compute_vectors = 'V';
  constexpr char lower_triangle = 'L';

  int lwork = -1;
  int liwork = -1;
  double work_query = 0.0;
  int iwork_query = 0;
  int info = 0;
  dsyevd_(&compute_vectors, &lower_triangle, &n, eigenvectors.data(), &lda,
          eigenvalues.data(), &work_query, &lwork, &iwork_query, &liwork,
          &info);
  if(info != 0 || !std::isfinite(work_query) || work_query < 1.0 ||
     iwork_query < 1) {
    throw std::runtime_error(
        "LAPACK dsyevd workspace query failed with info=" +
        std::to_string(info));
  }
  if(work_query >
     static_cast<double>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("LAPACK dsyevd workspace exceeds integer range");
  }
  lwork = static_cast<int>(std::ceil(work_query));
  liwork = iwork_query;
  std::vector<double> work(static_cast<std::size_t>(lwork));
  std::vector<int> iwork(static_cast<std::size_t>(liwork));
  dsyevd_(&compute_vectors, &lower_triangle, &n, eigenvectors.data(), &lda,
          eigenvalues.data(), work.data(), &lwork, iwork.data(), &liwork,
          &info);
  if(info != 0) {
    throw std::runtime_error("LAPACK dsyevd failed with info=" +
                             std::to_string(info));
  }
  require_finite(eigenvectors, "symmetric eigenvectors");
  for(const double value : eigenvalues) {
    if(!std::isfinite(value)) {
      throw std::runtime_error(
          "symmetric eigensolver returned a nonfinite eigenvalue");
    }
  }
  return SymmetricEigendecomposition{
      .eigenvalues = std::move(eigenvalues),
      .eigenvectors = std::move(eigenvectors),
  };
}

Matrix solve_linear_system(Matrix coefficients, Matrix right_hand_sides) {
  if(coefficients.empty() || coefficients.rows() != coefficients.columns()) {
    throw std::invalid_argument(
        "linear solve requires a nonempty square coefficient matrix");
  }
  if(right_hand_sides.rows() != coefficients.rows() ||
     right_hand_sides.columns() == 0) {
    throw std::invalid_argument(
        "linear solve right-hand-side dimensions are incompatible");
  }
  require_finite(coefficients, "linear solve coefficients");
  require_finite(right_hand_sides, "linear solve right-hand sides");
  const int n = lapack_dimension(coefficients.rows(), "coefficient dimension");
  const int nrhs =
      lapack_dimension(right_hand_sides.columns(), "right-hand-side count");
  const int lda = n;
  const int ldb = n;
  std::vector<int> pivots(coefficients.rows());
  int info = 0;
  dgesv_(&n, &nrhs, coefficients.data(), &lda, pivots.data(),
         right_hand_sides.data(), &ldb, &info);
  if(info < 0) {
    throw std::runtime_error("LAPACK dgesv rejected argument " +
                             std::to_string(-info));
  }
  if(info > 0) {
    throw std::runtime_error(
        "linear system is singular at pivot " + std::to_string(info));
  }
  require_finite(right_hand_sides, "linear solve result");
  return right_hand_sides;
}

Matrix solve_symmetric_positive_definite(
    Matrix coefficients, Matrix right_hand_sides) {
  if(coefficients.empty() || coefficients.rows() != coefficients.columns()) {
    throw std::invalid_argument(
        "positive-definite solve requires a nonempty square matrix");
  }
  if(right_hand_sides.rows() != coefficients.rows() ||
     right_hand_sides.columns() == 0) {
    throw std::invalid_argument(
        "positive-definite solve right-hand-side dimensions are incompatible");
  }
  require_finite(coefficients, "positive-definite solve coefficients");
  require_finite(right_hand_sides,
                 "positive-definite solve right-hand sides");
  if(maximum_asymmetry(coefficients) > 1.0e-12) {
    throw std::invalid_argument(
        "positive-definite solve coefficients exceed symmetry tolerance");
  }
  symmetrize_in_place(coefficients);
  const int n = lapack_dimension(coefficients.rows(), "coefficient dimension");
  const int nrhs =
      lapack_dimension(right_hand_sides.columns(), "right-hand-side count");
  const int lda = n;
  const int ldb = n;
  constexpr char lower_triangle = 'L';
  int info = 0;
  dposv_(&lower_triangle, &n, &nrhs, coefficients.data(), &lda,
         right_hand_sides.data(), &ldb, &info);
  if(info < 0) {
    throw std::runtime_error("LAPACK dposv rejected argument " +
                             std::to_string(-info));
  }
  if(info > 0) {
    throw std::runtime_error(
        "positive-definite system failed Cholesky factorization at minor " +
        std::to_string(info));
  }
  require_finite(right_hand_sides, "positive-definite solve result");
  return right_hand_sides;
}

LeastSquaresResult solve_least_squares_svd(
    Matrix design, const Matrix& right_hand_sides,
    double relative_singular_value_cutoff) {
  if(design.empty() || design.rows() < design.columns()) {
    throw std::invalid_argument(
        "SVD least squares requires a nonempty overdetermined design matrix");
  }
  if(right_hand_sides.rows() != design.rows() ||
     right_hand_sides.columns() == 0) {
    throw std::invalid_argument(
        "SVD least-squares right-hand-side dimensions are incompatible");
  }
  if(!std::isfinite(relative_singular_value_cutoff) ||
     relative_singular_value_cutoff <= 0.0 ||
     relative_singular_value_cutoff >= 1.0) {
    throw std::invalid_argument(
        "SVD least-squares cutoff must be finite and between zero and one");
  }
  require_finite(design, "SVD least-squares design");
  require_finite(right_hand_sides, "SVD least-squares right-hand sides");

  const int m = lapack_dimension(design.rows(), "least-squares row count");
  const int n =
      lapack_dimension(design.columns(), "least-squares column count");
  const int nrhs = lapack_dimension(right_hand_sides.columns(),
                                    "least-squares right-hand-side count");
  const int lda = m;
  const int ldb = std::max(m, n);
  Matrix padded_right_hand_sides{static_cast<std::size_t>(ldb),
                                 right_hand_sides.columns()};
  for(std::size_t column = 0; column < right_hand_sides.columns(); ++column) {
    for(std::size_t row = 0; row < right_hand_sides.rows(); ++row) {
      padded_right_hand_sides(row, column) =
          right_hand_sides(row, column);
    }
  }
  std::vector<double> singular_values(
      static_cast<std::size_t>(std::min(m, n)));
  int rank = 0;
  int lwork = -1;
  double work_query = 0.0;
  int info = 0;
  dgelss_(&m, &n, &nrhs, design.data(), &lda,
          padded_right_hand_sides.data(), &ldb, singular_values.data(),
          &relative_singular_value_cutoff, &rank, &work_query, &lwork,
          &info);
  if(info != 0 || !std::isfinite(work_query) || work_query < 1.0 ||
     work_query >
         static_cast<double>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "LAPACK dgelss workspace query failed with info=" +
        std::to_string(info));
  }
  lwork = static_cast<int>(std::ceil(work_query));
  std::vector<double> work(static_cast<std::size_t>(lwork));
  dgelss_(&m, &n, &nrhs, design.data(), &lda,
          padded_right_hand_sides.data(), &ldb, singular_values.data(),
          &relative_singular_value_cutoff, &rank, work.data(), &lwork,
          &info);
  if(info < 0) {
    throw std::runtime_error("LAPACK dgelss rejected argument " +
                             std::to_string(-info));
  }
  if(info > 0) {
    throw std::runtime_error(
        "LAPACK dgelss failed to converge after " +
        std::to_string(info) + " off-diagonal elements");
  }
  if(rank < 0 || rank > n) {
    throw std::runtime_error(
        "LAPACK dgelss returned an invalid numerical rank");
  }
  Matrix solution{design.columns(), right_hand_sides.columns()};
  for(std::size_t column = 0; column < solution.columns(); ++column) {
    for(std::size_t row = 0; row < solution.rows(); ++row) {
      solution(row, column) = padded_right_hand_sides(row, column);
    }
  }
  require_finite(solution, "SVD least-squares solution");
  for(const double singular_value : singular_values) {
    if(!std::isfinite(singular_value) || singular_value < 0.0) {
      throw std::runtime_error(
          "SVD least-squares returned an invalid singular value");
    }
  }
  return LeastSquaresResult{
      .solution = std::move(solution),
      .singular_values = std::move(singular_values),
      .numerical_rank = static_cast<std::size_t>(rank),
  };
}

}  // namespace modernqc::linalg
