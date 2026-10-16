#include "localized_cc/hamiltonian_rotation.hpp"
#include "localized_cc/parallel.hpp"

#include "modernqc/linalg/matrix.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace localized_cc {
namespace {

extern "C" {
void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb, const double* beta, double* c,
            const int* ldc);
}

std::size_t checked_product(const std::size_t left, const std::size_t right,
                            const char* label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(std::string(label) + " size overflows");
  return left * right;
}

std::size_t checked_sum(const std::size_t left, const std::size_t right,
                        const char* label) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error(std::string(label) + " size overflows");
  return left + right;
}

int checked_blas_dimension(const std::size_t value, const char* label) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error(std::string(label) +
                              " exceeds the BLAS integer range");
  return static_cast<int>(value);
}

modernqc::linalg::Matrix copy_matrix(const Matrix& source) {
  const auto dimensions = source.dimensions();
  modernqc::linalg::Matrix result(dimensions[0], dimensions[1]);
  for (std::size_t row = 0; row < dimensions[0]; ++row)
    for (std::size_t column = 0; column < dimensions[1]; ++column)
      result(row, column) = source(row, column);
  return result;
}

Matrix copy_matrix(const modernqc::linalg::Matrix& source) {
  Matrix result({source.rows(), source.columns()});
  for (std::size_t row = 0; row < source.rows(); ++row)
    for (std::size_t column = 0; column < source.columns(); ++column)
      result(row, column) = source(row, column);
  return result;
}

Matrix rotate_one_body(const Matrix& source,
                       const modernqc::linalg::Matrix& rotation) {
  const auto dimensions = source.dimensions();
  if (dimensions !=
          std::array<std::size_t, 2>{rotation.rows(), rotation.rows()} ||
      rotation.rows() != rotation.columns())
    throw std::invalid_argument(
        "spatial Hamiltonian one-body rotation dimensions mismatch");
  return copy_matrix(modernqc::linalg::multiply(
      modernqc::linalg::transpose(rotation),
      modernqc::linalg::multiply(copy_matrix(source), rotation)));
}

modernqc::linalg::Matrix block_diagonal_rotation(
    const modernqc::linalg::Matrix& occupied,
    const modernqc::linalg::Matrix& virtuals) {
  if (occupied.rows() == 0 || virtuals.rows() == 0 ||
      occupied.rows() != occupied.columns() ||
      virtuals.rows() != virtuals.columns())
    throw std::invalid_argument(
        "spatial Hamiltonian occupied/virtual rotations must be nonempty "
        "square matrices");
  const auto orbitals = checked_sum(occupied.rows(), virtuals.rows(),
                                    "spatial orbital count");
  modernqc::linalg::Matrix result(orbitals, orbitals);
  for (std::size_t row = 0; row < occupied.rows(); ++row)
    for (std::size_t column = 0; column < occupied.columns(); ++column)
      result(row, column) = occupied(row, column);
  for (std::size_t row = 0; row < virtuals.rows(); ++row)
    for (std::size_t column = 0; column < virtuals.columns(); ++column)
      result(occupied.rows() + row, occupied.columns() + column) =
          virtuals(row, column);
  return result;
}

SpatialHamiltonianRotationResult rotate_spatial_hamiltonian_rank_local(
    SpatialRhfHamiltonian source,
    const modernqc::linalg::Matrix& occupied_rotation,
    const modernqc::linalg::Matrix& virtual_rotation,
    const std::size_t maximum_additional_memory_bytes) {
  const auto begin = std::chrono::steady_clock::now();
  const auto rotation =
      block_diagonal_rotation(occupied_rotation, virtual_rotation);
  const auto orbitals = rotation.rows();
  if (source.nocc != occupied_rotation.rows() ||
      source.nvirt != virtual_rotation.rows() ||
      source.fock.dimensions() !=
          std::array<std::size_t, 2>{orbitals, orbitals} ||
      source.hcore.dimensions() !=
          std::array<std::size_t, 2>{orbitals, orbitals} ||
      source.eri_chemist.dimensions() !=
          std::array<std::size_t, 4>{orbitals, orbitals, orbitals, orbitals})
    throw std::invalid_argument(
        "spatial Hamiltonian rotation dimensions are inconsistent");

  const auto square = checked_product(orbitals, orbitals, "orbital pair");
  const auto cube = checked_product(square, orbitals, "orbital cube");
  const auto fourth = checked_product(cube, orbitals, "orbital fourth power");
  const auto logical_bytes =
      checked_product(fourth, sizeof(double), "logical ERI bytes");
  const auto stored_bytes = checked_product(
      source.eri_chemist.size(), sizeof(double), "stored ERI bytes");
  const auto two_dense_bytes =
      checked_product(logical_bytes, 2, "four-pass ERI buffers");
  const auto estimated_peak_additional_bytes = checked_sum(
      two_dense_bytes, stored_bytes, "Hamiltonian rotation peak");
  if (estimated_peak_additional_bytes > maximum_additional_memory_bytes)
    throw std::runtime_error(
        "canonical-to-localized Hamiltonian rotation requires " +
        std::to_string(static_cast<double>(estimated_peak_additional_bytes) /
                       (1024.0 * 1024.0)) +
        " MiB per process, exceeding the configured memory guard");

  auto rotated_fock = rotate_one_body(source.fock, rotation);
  auto rotated_hcore = rotate_one_body(source.hcore, rotation);
  std::vector<double> dense_source(fourth, 0.0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t p_index = 0;
       p_index < static_cast<std::ptrdiff_t>(orbitals); ++p_index) {
    const auto p = static_cast<std::size_t>(p_index);
    for (std::size_t q = 0; q < orbitals; ++q)
      for (std::size_t r = 0; r < orbitals; ++r)
        for (std::size_t s = 0; s < orbitals; ++s)
          dense_source[((p * orbitals + q) * orbitals + r) * orbitals + s] =
              source.eri_chemist(p, q, r, s);
  }
  source.eri_chemist = SymmetricEri4{};

  std::vector<double> first(fourth, 0.0);
  constexpr char no_transpose = 'N';
  constexpr char transpose = 'T';
  constexpr double one = 1.0;
  constexpr double zero = 0.0;
  const int n = checked_blas_dimension(orbitals, "orbital count");
  const int n2 = checked_blas_dimension(square, "orbital-pair count");
  const int n3 = checked_blas_dimension(cube, "orbital-cube count");

  std::size_t workers = 1;
#if defined(_OPENMP)
  workers = std::min(orbitals, static_cast<std::size_t>(omp_get_max_threads()));
#pragma omp parallel num_threads(static_cast<int>(workers))
#endif
  {
    std::size_t worker = 0;
    std::size_t actual_workers = 1;
#if defined(_OPENMP)
    worker = static_cast<std::size_t>(omp_get_thread_num());
    actual_workers = static_cast<std::size_t>(omp_get_num_threads());
#endif
    const auto first_row = cube * worker / actual_workers;
    const auto last_row = cube * (worker + 1) / actual_workers;
    const int rows =
        checked_blas_dimension(last_row - first_row, "first-index row slab");
    if (rows != 0)
      dgemm_(&no_transpose, &no_transpose, &rows, &n, &n, &one,
             dense_source.data() + static_cast<std::ptrdiff_t>(first_row),
             &n3, rotation.data(), &n, &zero,
             first.data() + static_cast<std::ptrdiff_t>(first_row), &n3);
  }
  std::vector<double>().swap(dense_source);

  std::vector<double> second(fourth, 0.0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t p_index = 0;
       p_index < static_cast<std::ptrdiff_t>(orbitals); ++p_index) {
    const auto p = static_cast<std::size_t>(p_index);
    dgemm_(&no_transpose, &no_transpose, &n2, &n, &n, &one,
           first.data() + static_cast<std::ptrdiff_t>(p * cube), &n2,
           rotation.data(), &n, &zero,
           second.data() + static_cast<std::ptrdiff_t>(p * cube), &n2);
  }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t pair_index = 0;
       pair_index < static_cast<std::ptrdiff_t>(square); ++pair_index) {
    const auto pair = static_cast<std::size_t>(pair_index);
    dgemm_(&no_transpose, &no_transpose, &n, &n, &n, &one,
           second.data() + static_cast<std::ptrdiff_t>(pair * square), &n,
           rotation.data(), &n, &zero,
           first.data() + static_cast<std::ptrdiff_t>(pair * square), &n);
  }

#if defined(_OPENMP)
#pragma omp parallel num_threads(static_cast<int>(workers))
#endif
  {
    std::size_t worker = 0;
    std::size_t actual_workers = 1;
#if defined(_OPENMP)
    worker = static_cast<std::size_t>(omp_get_thread_num());
    actual_workers = static_cast<std::size_t>(omp_get_num_threads());
#endif
    const auto first_column = cube * worker / actual_workers;
    const auto last_column = cube * (worker + 1) / actual_workers;
    const int columns = checked_blas_dimension(last_column - first_column,
                                               "final-index column slab");
    if (columns != 0)
      dgemm_(&transpose, &no_transpose, &n, &columns, &n, &one,
             rotation.data(), &n,
             first.data() +
                 static_cast<std::ptrdiff_t>(first_column * orbitals),
             &n, &zero,
             second.data() +
                 static_cast<std::ptrdiff_t>(first_column * orbitals),
             &n);
  }

  SymmetricEri4 rotated_eri(orbitals);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t p_index = 0;
       p_index < static_cast<std::ptrdiff_t>(orbitals); ++p_index) {
    const auto p = static_cast<std::size_t>(p_index);
    for (std::size_t q = 0; q <= p; ++q) {
      const auto first_pair = p * (p + 1) / 2 + q;
      std::size_t second_pair = 0;
      for (std::size_t r = 0; r < orbitals; ++r)
        for (std::size_t s = 0; s <= r; ++s, ++second_pair)
          if (second_pair <= first_pair)
            rotated_eri(p, q, r, s) =
                second[((p * orbitals + q) * orbitals + r) * orbitals + s];
    }
  }
  for (const auto value : rotated_eri.data())
    if (!std::isfinite(value))
      throw std::runtime_error(
          "canonical-to-localized Hamiltonian rotation produced non-finite "
          "integrals");

  source.fock = std::move(rotated_fock);
  source.hcore = std::move(rotated_hcore);
  source.eri_chemist = std::move(rotated_eri);
  const auto end = std::chrono::steady_clock::now();
  return {std::move(source),
          {orbitals,
           logical_bytes,
           stored_bytes,
           estimated_peak_additional_bytes,
           std::chrono::duration<double>(end - begin).count()}};
}

} // namespace

SpatialHamiltonianRotationResult rotate_spatial_hamiltonian(
    SpatialRhfHamiltonian source,
    const modernqc::linalg::Matrix& occupied_rotation,
    const modernqc::linalg::Matrix& virtual_rotation,
    const std::size_t maximum_additional_memory_bytes) {
  const auto collective_begin = std::chrono::steady_clock::now();
  const auto communicator = parallel::world();
  const auto orbitals =
      checked_sum(source.nocc, source.nvirt, "spatial orbital count");
  if (source.nocc != occupied_rotation.rows() ||
      source.nvirt != virtual_rotation.rows() ||
      occupied_rotation.rows() != occupied_rotation.columns() ||
      virtual_rotation.rows() != virtual_rotation.columns() ||
      source.fock.dimensions() !=
          std::array<std::size_t, 2>{orbitals, orbitals} ||
      source.hcore.dimensions() !=
          std::array<std::size_t, 2>{orbitals, orbitals} ||
      source.eri_chemist.dimensions() !=
          std::array<std::size_t, 4>{orbitals, orbitals, orbitals, orbitals})
    throw std::invalid_argument(
        "collective spatial Hamiltonian rotation dimensions are "
        "inconsistent");
  const auto square = checked_product(orbitals, orbitals, "orbital pair");
  const auto fourth = checked_product(square, square, "orbital fourth power");
  const auto logical_bytes =
      checked_product(fourth, sizeof(double), "logical ERI bytes");
  const auto stored_bytes = checked_product(
      source.eri_chemist.size(), sizeof(double), "stored ERI bytes");
  const auto peak_bytes = checked_sum(
      checked_product(logical_bytes, 2, "four-pass ERI buffers"),
      stored_bytes, "Hamiltonian rotation peak");
  if (peak_bytes > maximum_additional_memory_bytes)
    throw std::runtime_error(
        "canonical-to-localized Hamiltonian rotation requires " +
        std::to_string(static_cast<double>(peak_bytes) / (1024.0 * 1024.0)) +
        " MiB per process, exceeding the configured memory guard");

  std::optional<SpatialHamiltonianRotationResult> result;
  std::vector<int> failed(1, 0);
  if (communicator.rank == 0) {
    try {
      result = rotate_spatial_hamiltonian_rank_local(
          std::move(source), occupied_rotation, virtual_rotation,
          maximum_additional_memory_bytes);
    } catch (const std::exception&) {
      failed[0] = 1;
    }
  }
  parallel::broadcast_in_place(failed);
  if (failed[0] != 0)
    throw std::runtime_error(
        "rank-zero canonical-to-localized Hamiltonian rotation failed");
  if (communicator.rank != 0) {
    source.fock = Matrix({orbitals, orbitals});
    source.hcore = Matrix({orbitals, orbitals});
    source.eri_chemist = SymmetricEri4(orbitals);
    result = SpatialHamiltonianRotationResult{
        std::move(source),
        {orbitals, logical_bytes, stored_bytes, peak_bytes, 0.0}};
  }
  parallel::broadcast_in_place(result->hamiltonian.fock.data());
  parallel::broadcast_in_place(result->hamiltonian.hcore.data());
  parallel::broadcast_in_place(result->hamiltonian.eri_chemist.data());
  result->diagnostics.wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    collective_begin)
          .count();
  return std::move(*result);
}

} // namespace localized_cc
