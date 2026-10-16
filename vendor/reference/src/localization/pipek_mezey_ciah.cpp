#include "pipek_mezey_ciah.hpp"

#include "modernqc/linalg/eigensolver.hpp"
#include "modernqc/linalg/matrix.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef MODERNQC_HAS_OPENMP
#include <omp.h>
#endif

namespace modernqc::localization::detail {
namespace {

extern "C" {
void dgemm_(const char* transa, const char* transb, const int* m,
            const int* n, const int* k, const double* alpha,
            const double* a, const int* lda, const double* b,
            const int* ldb, const double* beta, double* c,
            const int* ldc);
}

// This is an independent C++ implementation of the real-orbital molecular
// Pipek--Mezey CIAH algorithm in PySCF 2.11.0 (Apache-2.0). The public source
// contracts used for parity are pyscf.lo.pipek, pyscf.lo.boys, and
// pyscf.soscf.ciah. No PySCF runtime or Python dependency is required.

using Vector = std::vector<double>;

void sum_parallel(Vector& values,
                  const LocalizationParallelOptions& parallel) {
  if(parallel.ranks > 1) {
    parallel.sum_in_place(values);
  }
}

void sum_parallel(linalg::Matrix& matrix,
                  const LocalizationParallelOptions& parallel) {
  if(parallel.ranks > 1) {
    parallel.sum_in_place(matrix.values());
  }
}

[[nodiscard]] double dot(const Vector& left, const Vector& right) {
  if(left.size() != right.size()) {
    throw std::invalid_argument("CIAH vector dimensions disagree");
  }
  long double result = 0.0L;
  for(std::size_t index = 0; index < left.size(); ++index) {
    result += static_cast<long double>(left[index]) * right[index];
  }
  return static_cast<double>(result);
}

[[nodiscard]] double norm(const Vector& values) {
  return std::sqrt(std::max(0.0, dot(values, values)));
}

[[nodiscard]] double maximum_absolute_value(const Vector& values) {
  double result = 0.0;
  for(const double value : values) {
    result = std::max(result, std::abs(value));
  }
  return result;
}

[[nodiscard]] linalg::Matrix sort_like_input(
    const linalg::Matrix& rotation) {
  if(rotation.rows() != rotation.columns()) {
    throw std::invalid_argument("CIAH final rotation is nonsquare");
  }
  const std::size_t n = rotation.rows();
  std::vector<bool> used(n, false);
  std::vector<std::size_t> order(n, 0);
  for(std::size_t input = 0; input < n; ++input) {
    std::size_t selected = n;
    double largest = -1.0;
    for(std::size_t localized = 0; localized < n; ++localized) {
      if(used[localized]) {
        continue;
      }
      const double overlap = std::abs(rotation(input, localized));
      if(overlap > largest) {
        largest = overlap;
        selected = localized;
      }
    }
    if(selected == n) {
      throw std::runtime_error("CIAH final orbital sorting failed");
    }
    used[selected] = true;
    order[input] = selected;
  }
  linalg::Matrix sorted{n, n};
  for(std::size_t column = 0; column < n; ++column) {
    for(std::size_t row = 0; row < n; ++row) {
      sorted(row, column) = rotation(row, order[column]);
    }
  }
  return sorted;
}

void add_scaled_in_place(Vector& target, const Vector& source,
                         double scale) {
  if(target.size() != source.size()) {
    throw std::invalid_argument("CIAH vector dimensions disagree");
  }
  for(std::size_t index = 0; index < target.size(); ++index) {
    target[index] += scale * source[index];
  }
}

[[nodiscard]] Vector scaled(const Vector& source, double factor) {
  Vector result = source;
  for(double& value : result) {
    value *= factor;
  }
  return result;
}

[[nodiscard]] Vector linear_combination(
    const Vector& coefficients, const std::vector<Vector>& vectors) {
  if(coefficients.size() != vectors.size() || vectors.empty()) {
    throw std::invalid_argument("CIAH linear-combination dimensions disagree");
  }
  Vector result(vectors.front().size(), 0.0);
  for(std::size_t vector = 0; vector < vectors.size(); ++vector) {
    add_scaled_in_place(result, vectors[vector], coefficients[vector]);
  }
  return result;
}

[[nodiscard]] std::size_t parameter_count(std::size_t orbitals) {
  if(orbitals > 0 && orbitals - 1 >
                         std::numeric_limits<std::size_t>::max() / orbitals) {
    throw std::overflow_error("CIAH rotation parameter count overflows");
  }
  return orbitals * (orbitals - 1) / 2;
}

[[nodiscard]] int atom_reduction_threads(std::size_t orbitals,
                                         std::size_t atoms) {
#ifdef MODERNQC_HAS_OPENMP
  if(orbitals >= 32 && atoms >= 2) {
    const std::size_t limit = std::min(
        atoms, static_cast<std::size_t>(omp_get_max_threads()));
    return std::max(1, static_cast<int>(limit));
  }
#else
  (void)orbitals;
  (void)atoms;
#endif
  return 1;
}

[[nodiscard]] int blas_dimension(std::size_t value, const char* name) {
  if(value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string{name} +
                              " exceeds the BLAS integer range");
  }
  return static_cast<int>(value);
}

[[nodiscard]] int row_block_threads(std::size_t rows) {
#ifdef MODERNQC_HAS_OPENMP
  if(rows >= 32 && !omp_in_parallel()) {
    // Four output rows per worker still leave O(n^2) work in every DGEMM for
    // the complete virtual space, while avoiding hundreds of tiny calls.
    const std::size_t useful = (rows + 3U) / 4U;
    return std::max(
        1, std::min(omp_get_max_threads(),
                    blas_dimension(useful, "CIAH row blocks")));
  }
#else
  (void)rows;
#endif
  return 1;
}

[[nodiscard]] bool use_intra_atom_parallelism(std::size_t orbitals,
                                              std::size_t atoms) {
#ifdef MODERNQC_HAS_OPENMP
  return orbitals >= 32 && atoms > 0 &&
         atoms < static_cast<std::size_t>(
                     std::max(2, omp_get_max_threads() / 2));
#else
  (void)orbitals;
  (void)atoms;
  return false;
#endif
}

// LibSci is linked in its serial form. Split the output rows among OpenMP
// workers so each worker executes an independent BLAS contraction without
// nested math-library threading. Matrix storage remains column major; a row
// submatrix is addressed by an offset while retaining the parent leading
// dimension.
[[nodiscard]] linalg::Matrix multiply_openmp_rows(
    const linalg::Matrix& left, const linalg::Matrix& right) {
  if(left.empty() || right.empty() || left.columns() != right.rows()) {
    throw std::invalid_argument(
        "CIAH matrix multiplication dimensions are incompatible");
  }
  const int threads = row_block_threads(left.rows());
  if(threads == 1) {
    return linalg::multiply(left, right);
  }

  linalg::Matrix result{left.rows(), right.columns()};
  const int total_rows = blas_dimension(left.rows(), "CIAH left row count");
  const int columns =
      blas_dimension(right.columns(), "CIAH right column count");
  const int contracted =
      blas_dimension(left.columns(), "CIAH contracted dimension");
  const int left_leading = total_rows;
  const int right_leading =
      blas_dimension(right.rows(), "CIAH right row count");
  const int result_leading = total_rows;
  constexpr char no_transpose = 'N';
  constexpr double alpha = 1.0;
  constexpr double beta = 0.0;

#ifdef MODERNQC_HAS_OPENMP
#pragma omp parallel num_threads(threads)
#endif
  {
#ifdef MODERNQC_HAS_OPENMP
    const int thread = omp_get_thread_num();
#else
    const int thread = 0;
#endif
    const int first = (total_rows * thread) / threads;
    const int last = (total_rows * (thread + 1)) / threads;
    const int rows = last - first;
    if(rows > 0) {
      dgemm_(&no_transpose, &no_transpose, &rows, &columns, &contracted,
             &alpha, left.data() + first, &left_leading, right.data(),
             &right_leading, &beta, result.data() + first,
             &result_leading);
    }
  }
  linalg::require_finite(result, "OpenMP row-blocked CIAH matrix product");
  return result;
}

[[nodiscard]] std::vector<linalg::Matrix> zero_matrices(
    int count, std::size_t rows, std::size_t columns) {
  std::vector<linalg::Matrix> result;
  result.reserve(static_cast<std::size_t>(count));
  for(int index = 0; index < count; ++index) {
    result.emplace_back(rows, columns);
  }
  return result;
}

[[nodiscard]] linalg::Matrix unpack_rotation(const Vector& packed,
                                             std::size_t orbitals) {
  if(packed.size() != parameter_count(orbitals)) {
    throw std::invalid_argument("CIAH packed rotation has wrong length");
  }
  linalg::Matrix result{orbitals, orbitals};
  std::size_t position = 0;
  for(std::size_t row = 1; row < orbitals; ++row) {
    for(std::size_t column = 0; column < row; ++column) {
      result(row, column) = packed[position];
      result(column, row) = -packed[position];
      ++position;
    }
  }
  return result;
}

[[nodiscard]] Vector pack_lower(const linalg::Matrix& matrix) {
  if(matrix.rows() != matrix.columns()) {
    throw std::invalid_argument("CIAH matrix must be square");
  }
  Vector result(parameter_count(matrix.rows()));
  std::size_t position = 0;
  for(std::size_t row = 1; row < matrix.rows(); ++row) {
    for(std::size_t column = 0; column < row; ++column) {
      result[position++] = matrix(row, column);
    }
  }
  return result;
}

[[nodiscard]] linalg::Matrix matrix_linear_combination(
    const std::initializer_list<std::pair<double, const linalg::Matrix*>>&
        terms) {
  if(terms.size() == 0) {
    throw std::invalid_argument("matrix linear combination is empty");
  }
  const linalg::Matrix& first = *terms.begin()->second;
  linalg::Matrix result{first.rows(), first.columns()};
  for(const auto& [factor, matrix] : terms) {
    if(matrix->rows() != result.rows() ||
       matrix->columns() != result.columns()) {
      throw std::invalid_argument(
          "matrix linear-combination dimensions disagree");
    }
    for(std::size_t index = 0; index < result.size(); ++index) {
      result.values()[index] += factor * matrix->values()[index];
    }
  }
  return result;
}

// Higham scaling-and-squaring Padé(13), the same numerical family used by
// scipy.linalg.expm for the small dense real skew matrices generated by CIAH.
[[nodiscard]] linalg::Matrix matrix_exponential(
    const linalg::Matrix& input) {
  if(input.rows() != input.columns()) {
    throw std::invalid_argument("matrix exponential requires a square matrix");
  }
  const std::size_t n = input.rows();
  double one_norm = 0.0;
  for(std::size_t column = 0; column < n; ++column) {
    double sum = 0.0;
    for(std::size_t row = 0; row < n; ++row) {
      sum += std::abs(input(row, column));
    }
    one_norm = std::max(one_norm, sum);
  }
  constexpr double theta13 = 5.371920351148152;
  const int squarings =
      one_norm <= theta13
          ? 0
          : std::max(0, static_cast<int>(
                            std::ceil(std::log2(one_norm / theta13))));
  const double scale = std::ldexp(1.0, -squarings);
  const linalg::Matrix a = linalg::scaled(input, scale);
  const linalg::Matrix a2 = multiply_openmp_rows(a, a);
  const linalg::Matrix a4 = multiply_openmp_rows(a2, a2);
  const linalg::Matrix a6 = multiply_openmp_rows(a4, a2);
  const linalg::Matrix identity = linalg::Matrix::identity(n);
  constexpr std::array<double, 14> b{
      64764752532480000.0, 32382376266240000.0,
      7771770303897600.0, 1187353796428800.0,
      129060195264000.0, 10559470521600.0,
      670442572800.0, 33522128640.0, 1323241920.0,
      40840800.0, 960960.0, 16380.0, 182.0, 1.0};
  const linalg::Matrix inner_u = matrix_linear_combination(
      {{b[13], &a6}, {b[11], &a4}, {b[9], &a2}});
  const linalg::Matrix a6_inner_u =
      multiply_openmp_rows(a6, inner_u);
  const linalg::Matrix u_tail = matrix_linear_combination(
      {{1.0, &a6_inner_u}, {b[7], &a6},
       {b[5], &a4}, {b[3], &a2}, {b[1], &identity}});
  const linalg::Matrix u = multiply_openmp_rows(a, u_tail);
  const linalg::Matrix inner_v = matrix_linear_combination(
      {{b[12], &a6}, {b[10], &a4}, {b[8], &a2}});
  const linalg::Matrix a6_inner_v =
      multiply_openmp_rows(a6, inner_v);
  const linalg::Matrix v = matrix_linear_combination(
      {{1.0, &a6_inner_v}, {b[6], &a6},
       {b[4], &a4}, {b[2], &a2}, {b[0], &identity}});
  linalg::Matrix result = linalg::solve_linear_system(
      linalg::subtract(v, u), linalg::add(v, u));
  for(int iteration = 0; iteration < squarings; ++iteration) {
    result = multiply_openmp_rows(result, result);
  }
  linalg::require_finite(result, "CIAH matrix exponential");
  return result;
}

[[nodiscard]] linalg::Matrix apply_rotation_increment(
    const linalg::Matrix& rotation, const Vector& increment) {
  return multiply_openmp_rows(
      rotation,
      matrix_exponential(unpack_rotation(increment, rotation.rows())));
}

template <typename Operation>
[[nodiscard]] linalg::Matrix root_matrix_operation(
    std::size_t rows, std::size_t columns,
    const LocalizationParallelOptions& parallel, Operation&& operation,
    const char* description) {
  if(parallel.ranks == 1) {
    return operation();
  }
  linalg::Matrix result{rows, columns};
  std::exception_ptr root_error;
  if(parallel.rank == 0) {
    try {
      result = operation();
      if(result.rows() != rows || result.columns() != columns) {
        throw std::logic_error(std::string{description} +
                               " returned incompatible dimensions");
      }
    } catch(...) {
      root_error = std::current_exception();
    }
  }
  Vector failed{root_error ? 1.0 : 0.0};
  sum_parallel(failed, parallel);
  if(failed.front() != 0.0) {
    if(root_error) {
      std::rethrow_exception(root_error);
    }
    throw std::runtime_error(std::string{"rank zero failed during "} +
                             description);
  }
  parallel.broadcast_from_root(result.values());
  linalg::require_finite(result, description);
  return result;
}

[[nodiscard]] linalg::Matrix apply_rotation_increment_parallel(
    const linalg::Matrix& rotation, const Vector& increment,
    const LocalizationParallelOptions& parallel) {
  return root_matrix_operation(
      rotation.rows(), rotation.columns(), parallel,
      [&]() { return apply_rotation_increment(rotation, increment); },
      "CIAH rotation exponential");
}

[[nodiscard]] linalg::Matrix multiply_global_rotation(
    const linalg::Matrix& left, const linalg::Matrix& right,
    const LocalizationParallelOptions& parallel) {
  return root_matrix_operation(
      left.rows(), right.columns(), parallel,
      [&]() { return multiply_openmp_rows(left, right); },
      "CIAH global rotation product");
}

[[nodiscard]] linalg::Matrix sort_global_rotation(
    const linalg::Matrix& rotation,
    const LocalizationParallelOptions& parallel) {
  return root_matrix_operation(
      rotation.rows(), rotation.columns(), parallel,
      [&]() { return sort_like_input(rotation); },
      "CIAH final orbital sorting");
}

[[nodiscard]] std::vector<linalg::Matrix> rotate_populations(
    const std::vector<linalg::Matrix>& canonical,
    const linalg::Matrix& rotation) {
  std::vector<linalg::Matrix> result(
      canonical.size(), linalg::Matrix{rotation.rows(), rotation.columns()});
  const linalg::Matrix transpose = linalg::transpose(rotation);
  if(use_intra_atom_parallelism(rotation.rows(), canonical.size())) {
    for(std::size_t atom = 0; atom < canonical.size(); ++atom) {
      result[atom] = multiply_openmp_rows(
          transpose, multiply_openmp_rows(canonical[atom], rotation));
    }
    return result;
  }

  const int threads = atom_reduction_threads(rotation.rows(), canonical.size());
#ifdef MODERNQC_HAS_OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) if(threads > 1)
#endif
  for(std::ptrdiff_t atom = 0;
      atom < static_cast<std::ptrdiff_t>(canonical.size()); ++atom) {
    const std::size_t index = static_cast<std::size_t>(atom);
    result[index] = linalg::multiply(
        transpose, linalg::multiply(canonical[index], rotation));
  }
  return result;
}

[[nodiscard]] double objective(
    const std::vector<linalg::Matrix>& populations,
    const LocalizationParallelOptions& parallel) {
  long double result = 0.0L;
  for(const linalg::Matrix& atom : populations) {
    for(std::size_t orbital = 0; orbital < atom.rows(); ++orbital) {
      const long double population = atom(orbital, orbital);
      result += population * population;
    }
  }
  Vector reduced{static_cast<double>(result)};
  sum_parallel(reduced, parallel);
  result = reduced.front();
  if(!std::isfinite(static_cast<double>(result))) {
    throw std::runtime_error("CIAH Pipek-Mezey objective became nonfinite");
  }
  return static_cast<double>(result);
}

struct Derivatives {
  Vector gradient;
  Vector hessian_diagonal;
  std::vector<linalg::Matrix> populations;
  linalg::Matrix hessian_g_matrix;
  LocalizationParallelOptions parallel;

  [[nodiscard]] Vector hessian_action(const Vector& direction) const {
    const std::size_t n = populations.front().rows();
    const linalg::Matrix x = unpack_rotation(direction, n);
    if(hessian_g_matrix.rows() != n || hessian_g_matrix.columns() != n) {
      throw std::runtime_error("CIAH cached Hessian G matrix is inconsistent");
    }
    if(use_intra_atom_parallelism(n, populations.size())) {
      linalg::Matrix hx{n, n};
      for(const linalg::Matrix& atom : populations) {
        const linalg::Matrix population_times_x =
            multiply_openmp_rows(atom, x);
#ifdef MODERNQC_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(std::ptrdiff_t position = 0;
            position < static_cast<std::ptrdiff_t>(hx.size()); ++position) {
          const std::size_t index = static_cast<std::size_t>(position);
          const std::size_t i = index % n;
          const std::size_t j = index / n;
          hx.values()[index] -=
              8.0 * population_times_x(i, i) * atom.values()[index];
          hx.values()[index] +=
              4.0 * population_times_x.values()[index] * atom(j, j);
        }
      }
      sum_parallel(hx, parallel);
      hx = linalg::subtract(
          hx, linalg::scaled(
                  multiply_openmp_rows(hessian_g_matrix, x), 2.0));
      const linalg::Matrix raw_hx = hx;
#ifdef MODERNQC_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
      for(std::ptrdiff_t position = 0;
          position < static_cast<std::ptrdiff_t>(hx.size()); ++position) {
        const std::size_t index = static_cast<std::size_t>(position);
        const std::size_t i = index % n;
        const std::size_t j = index / n;
        hx.values()[index] = -(raw_hx(i, j) - raw_hx(j, i));
      }
      return pack_lower(hx);
    }

    const int threads = atom_reduction_threads(n, populations.size());
    std::vector<linalg::Matrix> hx_by_thread =
        zero_matrices(threads, n, n);
#ifdef MODERNQC_HAS_OPENMP
#pragma omp parallel num_threads(threads) if(threads > 1)
#endif
    {
#ifdef MODERNQC_HAS_OPENMP
      const int thread = omp_get_thread_num();
#else
      const int thread = 0;
#endif
      linalg::Matrix& local_hx =
          hx_by_thread[static_cast<std::size_t>(thread)];
      for(std::size_t atom_index = static_cast<std::size_t>(thread);
          atom_index < populations.size();
          atom_index += static_cast<std::size_t>(threads)) {
        const linalg::Matrix& atom = populations[atom_index];
        const linalg::Matrix population_times_x =
            linalg::multiply(atom, x);
        for(std::size_t i = 0; i < n; ++i) {
          for(std::size_t j = 0; j < n; ++j) {
            local_hx(i, j) -=
                8.0 * population_times_x(i, i) * atom(i, j);
            local_hx(i, j) +=
                4.0 * population_times_x(i, j) * atom(j, j);
          }
        }
      }
    }
    linalg::Matrix hx{n, n};
    for(int thread = 0; thread < threads; ++thread) {
      const linalg::Matrix& local_hx =
          hx_by_thread[static_cast<std::size_t>(thread)];
      for(std::size_t index = 0; index < hx.size(); ++index) {
        hx.values()[index] += local_hx.values()[index];
      }
    }
    sum_parallel(hx, parallel);
    hx = linalg::subtract(
        hx, linalg::scaled(linalg::multiply(hessian_g_matrix, x), 2.0));
    const linalg::Matrix raw_hx = hx;
    for(std::size_t i = 0; i < n; ++i) {
      for(std::size_t j = 0; j < n; ++j) {
        hx(i, j) = -(raw_hx(i, j) - raw_hx(j, i));
      }
    }
    return pack_lower(hx);
  }
};

[[nodiscard]] Derivatives derivatives(
    const std::vector<linalg::Matrix>& canonical_populations,
    const linalg::Matrix& rotation,
    const LocalizationParallelOptions& parallel) {
  Derivatives result;
  result.populations = rotate_populations(canonical_populations, rotation);
  const std::size_t n = rotation.rows();
  const int threads = atom_reduction_threads(n, result.populations.size());
  std::vector<linalg::Matrix> g_by_thread =
      zero_matrices(threads, n, n);
  std::vector<linalg::Matrix> h_by_thread =
      zero_matrices(threads, n, n);
#ifdef MODERNQC_HAS_OPENMP
#pragma omp parallel num_threads(threads) if(threads > 1)
#endif
  {
#ifdef MODERNQC_HAS_OPENMP
    const int thread = omp_get_thread_num();
#else
    const int thread = 0;
#endif
    linalg::Matrix& local_g =
        g_by_thread[static_cast<std::size_t>(thread)];
    linalg::Matrix& local_h =
        h_by_thread[static_cast<std::size_t>(thread)];
    for(std::size_t atom_index = static_cast<std::size_t>(thread);
        atom_index < result.populations.size();
        atom_index += static_cast<std::size_t>(threads)) {
      const linalg::Matrix& atom = result.populations[atom_index];
      for(std::size_t i = 0; i < n; ++i) {
        for(std::size_t j = 0; j < n; ++j) {
          local_g(i, j) += atom(j, j) * atom(i, j);
          local_h(i, j) +=
              -4.0 * (atom(i, i) * atom(i, i) -
                      atom(i, i) * atom(j, j)) +
              8.0 * atom(i, j) * atom(i, j);
        }
      }
    }
  }
  linalg::Matrix g_matrix{n, n};
  linalg::Matrix h_matrix{n, n};
  for(int thread = 0; thread < threads; ++thread) {
    const linalg::Matrix& local_g =
        g_by_thread[static_cast<std::size_t>(thread)];
    const linalg::Matrix& local_h =
        h_by_thread[static_cast<std::size_t>(thread)];
    for(std::size_t index = 0; index < g_matrix.size(); ++index) {
      g_matrix.values()[index] += local_g.values()[index];
      h_matrix.values()[index] += local_h.values()[index];
    }
  }
  sum_parallel(g_matrix, parallel);
  sum_parallel(h_matrix, parallel);
  linalg::Matrix gradient_matrix{n, n};
  linalg::Matrix hessian_diagonal_matrix{n, n};
  linalg::Matrix hessian_g_matrix{n, n};
  for(std::size_t i = 0; i < n; ++i) {
    for(std::size_t j = 0; j < n; ++j) {
      gradient_matrix(i, j) = -4.0 * (g_matrix(i, j) - g_matrix(j, i));
      hessian_diagonal_matrix(i, j) = -(h_matrix(i, j) + h_matrix(j, i));
      hessian_g_matrix(i, j) = g_matrix(i, j) + g_matrix(j, i);
    }
  }
  result.gradient = pack_lower(gradient_matrix);
  result.hessian_diagonal = pack_lower(hessian_diagonal_matrix);
  result.hessian_g_matrix = std::move(hessian_g_matrix);
  result.parallel = parallel;
  return result;
}

void require_consistent_hessian_diagonal(const Derivatives& model) {
  if(model.gradient.size() > 120) {
    return;
  }
  double maximum_error = 0.0;
  for(std::size_t parameter = 0; parameter < model.gradient.size();
      ++parameter) {
    Vector direction(model.gradient.size(), 0.0);
    direction[parameter] = 1.0;
    const Vector action = model.hessian_action(direction);
    maximum_error = std::max(
        maximum_error,
        std::abs(action[parameter] -
                 model.hessian_diagonal[parameter]));
  }
  if(maximum_error > 1.0e-9) {
    throw std::runtime_error(
        "CIAH Hessian action and diagonal disagree by " +
        std::to_string(maximum_error));
  }
}

void require_consistent_gradient(
    const std::vector<linalg::Matrix>& canonical_populations,
    const linalg::Matrix& rotation, const Derivatives& model) {
  if(model.gradient.size() > 120) {
    return;
  }
  constexpr double step = 1.0e-6;
  double maximum_error = 0.0;
  for(std::size_t parameter = 0; parameter < model.gradient.size();
      ++parameter) {
    Vector direction(model.gradient.size(), 0.0);
    direction[parameter] = step;
    const double plus = -objective(rotate_populations(
        canonical_populations,
        apply_rotation_increment(rotation, direction)), model.parallel);
    direction[parameter] = -step;
    const double minus = -objective(rotate_populations(
        canonical_populations,
        apply_rotation_increment(rotation, direction)), model.parallel);
    const double numerical = (plus - minus) / (2.0 * step);
    maximum_error = std::max(
        maximum_error, std::abs(numerical - model.gradient[parameter]));
  }
  if(maximum_error > 1.0e-7) {
    throw std::runtime_error(
        "CIAH analytic gradient differs from finite differences by " +
        std::to_string(maximum_error));
  }
}

struct GeneralizedEigenResult {
  std::vector<double> eigenvalues;
  linalg::Matrix eigenvectors;
  std::vector<double> overlap_eigenvalues;
};

[[nodiscard]] GeneralizedEigenResult safe_generalized_eigen(
    const linalg::Matrix& hessian, const linalg::Matrix& overlap,
    double linear_dependency) {
  const auto overlap_eigensystem = linalg::diagonalize_symmetric(overlap);
  std::vector<std::size_t> retained;
  for(std::size_t index = 0;
      index < overlap_eigensystem.eigenvalues.size(); ++index) {
    if(overlap_eigensystem.eigenvalues[index] > linear_dependency) {
      retained.push_back(index);
    }
  }
  if(retained.empty()) {
    throw std::runtime_error("CIAH Davidson overlap is rank deficient");
  }
  linalg::Matrix inverse_sqrt{overlap.rows(), retained.size()};
  for(std::size_t column = 0; column < retained.size(); ++column) {
    const std::size_t source = retained[column];
    const double scale =
        1.0 / std::sqrt(overlap_eigensystem.eigenvalues[source]);
    for(std::size_t row = 0; row < overlap.rows(); ++row) {
      inverse_sqrt(row, column) =
          overlap_eigensystem.eigenvectors(row, source) * scale;
    }
  }
  const linalg::Matrix reduced = linalg::multiply(
      linalg::transpose(inverse_sqrt),
      linalg::multiply(hessian, inverse_sqrt));
  const auto eigensystem = linalg::diagonalize_symmetric(reduced, 1.0e-10);
  return GeneralizedEigenResult{
      .eigenvalues = eigensystem.eigenvalues,
      .eigenvectors = linalg::multiply(inverse_sqrt,
                                       eigensystem.eigenvectors),
      .overlap_eigenvalues = overlap_eigensystem.eigenvalues,
  };
}

struct DavidsonTrial {
  bool converged{false};
  int hessian_actions{0};
  double eigenvalue{0.0};
  Vector step;
  Vector hessian_step;
  Vector residual;
  double smallest_overlap_eigenvalue{0.0};
};

[[nodiscard]] DavidsonTrial davidson_trial(
    const Vector& gradient, const std::vector<Vector>& vectors,
    const std::vector<Vector>& hessian_vectors,
    double previous_eigenvalue, const PipekMezeyOptions& options) {
  const std::size_t count = vectors.size();
  if(count == 0 || hessian_vectors.size() != count) {
    throw std::invalid_argument("CIAH Davidson basis is inconsistent");
  }
  linalg::Matrix effective{count + 1, count + 1};
  linalg::Matrix overlap = linalg::Matrix::identity(count + 1);
  for(std::size_t i = 0; i < count; ++i) {
    effective(0, i + 1) = dot(vectors[i], gradient);
    effective(i + 1, 0) = effective(0, i + 1);
    for(std::size_t j = 0; j <= i; ++j) {
      effective(i + 1, j + 1) = dot(vectors[i], hessian_vectors[j]);
      effective(j + 1, i + 1) = effective(i + 1, j + 1);
      overlap(i + 1, j + 1) = dot(vectors[i], vectors[j]);
      overlap(j + 1, i + 1) = overlap(i + 1, j + 1);
    }
  }
  const auto eigensystem = safe_generalized_eigen(
      effective, overlap,
      options.augmented_hessian_linear_dependency);
  std::size_t selected = eigensystem.eigenvalues.size();
  for(std::size_t root = 0; root < eigensystem.eigenvalues.size(); ++root) {
    if(std::abs(eigensystem.eigenvectors(0, root)) > 0.1) {
      selected = root;
      break;
    }
  }
  if(selected == eigensystem.eigenvalues.size()) {
    throw std::runtime_error(
        "CIAH augmented Hessian has no admissible root");
  }
  const double v0 = eigensystem.eigenvectors(0, selected);
  Vector coefficients(count);
  Vector unscaled_coefficients(count);
  for(std::size_t vector = 0; vector < count; ++vector) {
    unscaled_coefficients[vector] =
        eigensystem.eigenvectors(vector + 1, selected);
    coefficients[vector] = unscaled_coefficients[vector] / v0;
  }
  Vector step = linear_combination(coefficients, vectors);
  Vector unscaled_hessian =
      linear_combination(unscaled_coefficients, hessian_vectors);
  Vector hessian_step = scaled(unscaled_hessian, 1.0 / v0);
  const double eigenvalue = eigensystem.eigenvalues[selected];
  Vector residual = unscaled_hessian;
  add_scaled_in_place(residual, gradient, v0);
  add_scaled_in_place(residual, step, -eigenvalue * v0);
  const bool converged =
      (std::abs(eigenvalue - previous_eigenvalue) <
           options.augmented_hessian_convergence_tolerance &&
       norm(residual) <
           std::sqrt(options.augmented_hessian_convergence_tolerance)) ||
      eigensystem.overlap_eigenvalues.front() <
          options.augmented_hessian_linear_dependency ||
      count == gradient.size();
  return DavidsonTrial{
      .converged = converged,
      .hessian_actions = static_cast<int>(count),
      .eigenvalue = eigenvalue,
      .step = std::move(step),
      .hessian_step = std::move(hessian_step),
      .residual = std::move(residual),
      .smallest_overlap_eigenvalue =
          eigensystem.overlap_eigenvalues.front(),
  };
}

struct MicroResult {
  linalg::Matrix relative_rotation;
  Vector keyframe_gradient;
  Vector estimated_gradient;
  Vector last_step;
  double estimated_gradient_norm{0.0};
  double keyframe_trust_region{0.0};
  int keyframes{0};
  int hessian_actions{0};
};

[[nodiscard]] MicroResult micro_iterations(
    const std::vector<linalg::Matrix>& canonical_populations,
    const linalg::Matrix& base_rotation, const Derivatives& base_model,
    const Vector& initial_guess, double convergence_gradient,
    double initial_keyframe_trust,
    const PipekMezeyOptions& options) {
  Vector orbital_gradient = base_model.gradient;
  Vector keyframe_gradient = base_model.gradient;
  double keyframe_norm = norm(keyframe_gradient);
  double orbital_norm = keyframe_norm;
  double keyframe_trust = initial_keyframe_trust;
  Vector accumulated(orbital_gradient.size(), 0.0);
  linalg::Matrix keyframe_rotation =
      linalg::Matrix::identity(base_rotation.rows());
  int inner = 0;
  int since_keyframe = 0;
  int keyframes = 0;
  int hessian_actions = 0;
  Vector last_step = initial_guess;
  std::vector<Vector> trial_vectors{initial_guess};
  std::vector<Vector> trial_hessian{
      base_model.hessian_action(initial_guess)};
  double previous_eigenvalue = 0.0;

  const int maximum_davidson = std::min(
      options.augmented_hessian_maximum_iterations,
      static_cast<int>(orbital_gradient.size()));
  for(int davidson_cycle = 1; davidson_cycle <= maximum_davidson;
      ++davidson_cycle) {
    const DavidsonTrial trial = davidson_trial(
        orbital_gradient, trial_vectors, trial_hessian,
        previous_eigenvalue, options);
    hessian_actions = trial.hessian_actions;
    const bool accept =
        trial.converged || hessian_actions == maximum_davidson ||
        (norm(trial.residual) <
             options.augmented_hessian_start_tolerance &&
         hessian_actions >= options.augmented_hessian_start_cycle) ||
        trial.smallest_overlap_eigenvalue <
            options.augmented_hessian_linear_dependency;
    bool stop = false;
    if(accept) {
      ++inner;
      ++since_keyframe;
      last_step = trial.step;
      Vector hessian_step = trial.hessian_step;
      const double maximum = maximum_absolute_value(last_step);
      if(maximum > options.maximum_rotation_step) {
        const double factor = options.maximum_rotation_step / maximum;
        for(double& value : last_step) {
          value *= factor;
        }
        for(double& value : hessian_step) {
          value *= factor;
        }
      }
      add_scaled_in_place(accumulated, last_step, 1.0);
      add_scaled_in_place(orbital_gradient, hessian_step, 1.0);
      const double accumulated_norm = norm(accumulated);
      orbital_norm = norm(orbital_gradient);
      const int maximum_micro = std::max(
          options.maximum_micro_iterations,
          options.maximum_micro_iterations -
              static_cast<int>(std::log(keyframe_norm + 1.0e-9) * 2.0));
      if(inner > 3 &&
         orbital_norm >
             keyframe_norm * options.augmented_hessian_trust_region) {
        add_scaled_in_place(orbital_gradient, hessian_step, -1.0);
        add_scaled_in_place(accumulated, last_step, -1.0);
        orbital_norm = norm(orbital_gradient);
        stop = true;
      } else if(inner >= maximum_micro ||
                orbital_norm < convergence_gradient * 0.2) {
        stop = true;
      } else if(since_keyframe > 2 &&
                (since_keyframe >= std::max(
                     options.keyframe_interval,
                     options.keyframe_interval - static_cast<int>(
                         std::log(accumulated_norm + 1.0e-9))) ||
                 orbital_norm < keyframe_norm / keyframe_trust)) {
        since_keyframe = 0;
        keyframe_rotation = apply_rotation_increment_parallel(
            keyframe_rotation, accumulated, base_model.parallel);
        std::fill(accumulated.begin(), accumulated.end(), 0.0);
        const Derivatives refreshed = derivatives(
            canonical_populations,
            multiply_global_rotation(base_rotation, keyframe_rotation,
                                     base_model.parallel),
            base_model.parallel);
        ++keyframes;
        const double refreshed_norm = norm(refreshed.gradient);
        Vector correction = refreshed.gradient;
        add_scaled_in_place(correction, orbital_gradient, -1.0);
        const double correction_norm = norm(correction);
        if(correction_norm <
               orbital_norm * options.augmented_hessian_trust_region ||
           refreshed_norm < convergence_gradient *
                                options.augmented_hessian_trust_region) {
          keyframe_trust = std::min(
              std::max(orbital_norm / (correction_norm + 1.0e-9),
                       options.keyframe_trust_region),
              10.0);
          orbital_gradient = refreshed.gradient;
          keyframe_gradient = refreshed.gradient;
          orbital_norm = refreshed_norm;
          keyframe_norm = refreshed_norm;
        } else {
          add_scaled_in_place(orbital_gradient, hessian_step, -1.0);
          add_scaled_in_place(accumulated, last_step, -1.0);
          orbital_norm = norm(orbital_gradient);
          stop = true;
        }
      }
    }
    if(stop) {
      break;
    }

    previous_eigenvalue = trial.eigenvalue;
    if(trial.converged) {
      if(trial.smallest_overlap_eigenvalue <
             options.augmented_hessian_linear_dependency ||
         norm(trial.residual) <
             options.augmented_hessian_linear_dependency) {
        break;
      }
      // PySCF reuses the same Davidson subspace after a converged AH step;
      // the outer microiteration has updated the gradient through g_op.
      continue;
    }

    Vector preconditioned = trial.residual;
    for(std::size_t index = 0; index < preconditioned.size(); ++index) {
      double denominator = base_model.hessian_diagonal[index] -
                           (trial.eigenvalue -
                            options.augmented_hessian_level_shift);
      if(std::abs(denominator) < 1.0e-8) {
        denominator = 1.0e-8;
      }
      preconditioned[index] /= denominator;
    }
    trial_vectors.push_back(std::move(preconditioned));
    trial_hessian.push_back(
        base_model.hessian_action(trial_vectors.back()));
  }
  return MicroResult{
      .relative_rotation = apply_rotation_increment_parallel(
          keyframe_rotation, accumulated, base_model.parallel),
      .keyframe_gradient = std::move(keyframe_gradient),
      .estimated_gradient = orbital_gradient,
      .last_step = std::move(last_step),
      .estimated_gradient_norm = orbital_norm,
      .keyframe_trust_region = keyframe_trust,
      .keyframes = keyframes,
      .hessian_actions = hessian_actions,
  };
}

void validate_options(const PipekMezeyOptions& options) {
  const double gradient_tolerance =
      options.gradient_tolerance > 0.0
          ? options.gradient_tolerance
          : std::sqrt(options.objective_tolerance * 0.1);
  if(!std::isfinite(gradient_tolerance) || gradient_tolerance <= 0.0 ||
     !std::isfinite(options.maximum_rotation_step) ||
     options.maximum_rotation_step <= 0.0 ||
     options.maximum_micro_iterations <= 0 ||
     options.keyframe_interval <= 0 ||
     !std::isfinite(options.keyframe_trust_region) ||
     options.keyframe_trust_region <= 0.0 ||
     !std::isfinite(options.augmented_hessian_start_tolerance) ||
     options.augmented_hessian_start_tolerance <= 0.0 ||
     options.augmented_hessian_start_cycle <= 0 ||
     !std::isfinite(options.augmented_hessian_level_shift) ||
     !std::isfinite(options.augmented_hessian_convergence_tolerance) ||
     options.augmented_hessian_convergence_tolerance <= 0.0 ||
     !std::isfinite(options.augmented_hessian_linear_dependency) ||
     options.augmented_hessian_linear_dependency <= 0.0 ||
     options.augmented_hessian_maximum_iterations <= 0 ||
     !std::isfinite(options.augmented_hessian_trust_region) ||
     options.augmented_hessian_trust_region <= 0.0 ||
     options.parallel.ranks == 0 ||
     options.parallel.rank >= options.parallel.ranks ||
     (options.parallel.ranks > 1 &&
      (!options.parallel.sum_in_place ||
       !options.parallel.broadcast_from_root))) {
    throw std::invalid_argument("PySCF CIAH options are outside their domains");
  }
}

}  // namespace

PipekMezeyCiahResult optimize_pipek_mezey_ciah(
    const std::vector<linalg::Matrix>& canonical_populations,
    const linalg::Matrix& initial_rotation,
    const PipekMezeyOptions& options) {
  validate_options(options);
  if(options.restart_state) {
    throw std::invalid_argument(
        "PySCF CIAH restart requires optimizer state not present in the Jacobi restart format");
  }
  if(canonical_populations.empty() || initial_rotation.rows() == 0 ||
     initial_rotation.rows() != initial_rotation.columns()) {
    throw std::invalid_argument("PySCF CIAH inputs are empty or nonsquare");
  }
  const std::size_t n = initial_rotation.rows();
  for(const linalg::Matrix& population : canonical_populations) {
    if(population.rows() != n || population.columns() != n) {
      throw std::invalid_argument("CIAH population dimensions disagree");
    }
    linalg::require_finite(population, "CIAH population matrix");
  }
  linalg::Matrix rotation = initial_rotation;
  Derivatives model = derivatives(canonical_populations, rotation,
                                  options.parallel);
  require_consistent_hessian_diagonal(model);
  require_consistent_gradient(canonical_populations, rotation, model);
  const double convergence_gradient =
      options.gradient_tolerance > 0.0
          ? options.gradient_tolerance
          : std::sqrt(options.objective_tolerance * 0.1);

  // PySCF leaves the atomic saddle and starts from a deterministic small
  // rotation based on cos(arange(nparam)).
  if(norm(model.gradient) < 1.0e-5) {
    Vector noise(model.gradient.size());
    for(std::size_t index = 0; index < noise.size(); ++index) {
      noise[index] = std::cos(static_cast<double>(index)) * 1.0e-3;
    }
    rotation = apply_rotation_increment_parallel(
        linalg::Matrix::identity(n), noise, options.parallel);
    model = derivatives(canonical_populations, rotation, options.parallel);
    require_consistent_hessian_diagonal(model);
  }

  std::vector<double> history;
  history.reserve(static_cast<std::size_t>(options.maximum_sweeps) + 1U);
  history.push_back(objective(model.populations, options.parallel));
  Vector initial_guess = model.gradient;
  int total_keyframes = 0;
  int total_hessian_actions = 0;
  double final_gain = 0.0;
  double final_gradient = norm(model.gradient);
  int completed = 0;
  bool converged = n <= 1;
  double dynamic_keyframe_trust = options.keyframe_trust_region;

  const auto report = [&]() {
    if(options.progress) {
      options.progress(
          PipekMezeySweep{
              .completed_sweeps = completed,
              .converged = converged,
              .objective = history.back(),
              .sweep_gain = final_gain,
              .gradient_norm = final_gradient,
              .keyframes = total_keyframes,
              .hessian_actions = total_hessian_actions,
          },
          rotation, history);
    }
  };
  report();

  for(int macro = 1; macro <= options.maximum_sweeps && !converged;
      ++macro) {
    const MicroResult micro = micro_iterations(
        canonical_populations, rotation, model, initial_guess,
        convergence_gradient, dynamic_keyframe_trust, options);
    rotation = multiply_global_rotation(
        rotation, micro.relative_rotation, options.parallel);
    model = derivatives(canonical_populations, rotation, options.parallel);
    require_consistent_hessian_diagonal(model);
    const double next_objective =
        objective(model.populations, options.parallel);
    // PySCF initializes e_last to zero. Preserve the initial objective in the
    // diagnostic history, but use the same first-macro convergence delta.
    final_gain = macro == 1 ? next_objective
                            : next_objective - history.back();
    history.push_back(next_objective);
    completed = macro;
    final_gradient = norm(micro.keyframe_gradient);
    total_keyframes += micro.keyframes;
    total_hessian_actions += micro.hessian_actions;
    converged = final_gradient < convergence_gradient &&
                std::abs(final_gain) < options.objective_tolerance &&
                micro.hessian_actions <
                    options.augmented_hessian_maximum_iterations;
    initial_guess = micro.last_step;
    Vector keyframe_correction = model.gradient;
    add_scaled_in_place(keyframe_correction, micro.estimated_gradient,
                        -1.0);
    dynamic_keyframe_trust = std::min(
        std::max(micro.estimated_gradient_norm /
                     (norm(keyframe_correction) + 1.0e-9),
                 options.keyframe_trust_region),
        10.0);
    report();
  }
  if(!converged) {
    throw std::runtime_error(
        "PySCF CIAH Pipek-Mezey localization did not converge within maximum macro iterations");
  }
  rotation = sort_global_rotation(rotation, options.parallel);
  return PipekMezeyCiahResult{
      .rotation = std::move(rotation),
      .objective_history = std::move(history),
      .completed_macro_iterations = completed,
      .final_objective_gain = final_gain,
      .final_gradient_norm = final_gradient,
      .total_keyframes = total_keyframes,
      .total_hessian_actions = total_hessian_actions,
  };
}

}  // namespace modernqc::localization::detail
