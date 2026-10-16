#include "localized_cc/semicanonical_triples.hpp"

#include "localized_cc/basis_projection.hpp"

#include "modernqc/linalg/eigensolver.hpp"
#include "modernqc/linalg/matrix.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace localized_cc {
namespace {

extern "C" {
void dgemm_(const char *transa, const char *transb, const int *m, const int *n,
            const int *k, const double *alpha, const double *a, const int *lda,
            const double *b, const int *ldb, const double *beta, double *c,
            const int *ldc);
}

std::size_t checked_product(const std::size_t left, const std::size_t right,
                            const char *label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(std::string(label) + " size overflows");
  return left * right;
}

int checked_blas_dimension(const std::size_t value, const char *label) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error(std::string(label) +
                              " exceeds the BLAS integer range");
  return static_cast<int>(value);
}

modernqc::linalg::Matrix copy_matrix(const Matrix &source) {
  const auto dimensions = source.dimensions();
  modernqc::linalg::Matrix result(dimensions[0], dimensions[1]);
  for (std::size_t row = 0; row < dimensions[0]; ++row)
    for (std::size_t column = 0; column < dimensions[1]; ++column)
      result(row, column) = source(row, column);
  return result;
}

Matrix copy_matrix(const modernqc::linalg::Matrix &source) {
  Matrix result({source.rows(), source.columns()});
  for (std::size_t row = 0; row < source.rows(); ++row)
    for (std::size_t column = 0; column < source.columns(); ++column)
      result(row, column) = source(row, column);
  return result;
}

modernqc::linalg::Matrix block(const Matrix &source, const std::size_t first,
                               const std::size_t count) {
  const auto dimensions = source.dimensions();
  if (dimensions[0] != dimensions[1] || first > dimensions[0] ||
      count > dimensions[0] - first || count == 0)
    throw std::invalid_argument("invalid semicanonical Fock block");
  modernqc::linalg::Matrix result(count, count);
  for (std::size_t row = 0; row < count; ++row)
    for (std::size_t column = 0; column < count; ++column)
      result(row, column) = source(first + row, first + column);
  return result;
}

modernqc::linalg::Matrix
block_diagonal_rotation(const modernqc::linalg::Matrix &occupied,
                        const modernqc::linalg::Matrix &virtuals) {
  if (occupied.rows() != occupied.columns() ||
      virtuals.rows() != virtuals.columns())
    throw std::invalid_argument("semicanonical rotations must be square");
  const auto orbitals = occupied.rows() + virtuals.rows();
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

Matrix rotate_one_body(const Matrix &source,
                       const modernqc::linalg::Matrix &rotation) {
  const auto dimensions = source.dimensions();
  if (dimensions !=
          std::array<std::size_t, 2>{rotation.rows(), rotation.rows()} ||
      rotation.rows() != rotation.columns())
    throw std::invalid_argument(
        "semicanonical one-body rotation dimensions mismatch");
  const auto transformed = modernqc::linalg::multiply(
      modernqc::linalg::transpose(rotation),
      modernqc::linalg::multiply(copy_matrix(source), rotation));
  return copy_matrix(transformed);
}

SymmetricEri4 rotate_eri(const SymmetricEri4 &source,
                         const modernqc::linalg::Matrix &rotation) {
  const auto orbitals = rotation.rows();
  if (rotation.columns() != orbitals ||
      source.dimensions() !=
          std::array<std::size_t, 4>{orbitals, orbitals, orbitals, orbitals})
    throw std::invalid_argument(
        "semicanonical ERI rotation dimensions mismatch");
  const auto square = checked_product(orbitals, orbitals, "orbital pair");
  const auto cube = checked_product(square, orbitals, "orbital cube");
  const auto fourth = checked_product(cube, orbitals, "orbital fourth power");
  const int n = checked_blas_dimension(orbitals, "orbital count");
  const int n2 = checked_blas_dimension(square, "orbital-pair count");
  const int n3 = checked_blas_dimension(cube, "orbital-cube count");
  std::vector<double> dense_source(fourth, 0.0);
  for (std::size_t p = 0; p < orbitals; ++p)
    for (std::size_t q = 0; q < orbitals; ++q)
      for (std::size_t r = 0; r < orbitals; ++r)
        for (std::size_t s = 0; s < orbitals; ++s)
          dense_source[((p * orbitals + q) * orbitals + r) * orbitals + s] =
              source(p, q, r, s);
  std::vector<double> first(fourth, 0.0);
  constexpr char no_transpose = 'N';
  constexpr char transpose = 'T';
  constexpr double alpha = 1.0;
  constexpr double beta = 0.0;

  // Tensor storage is row-major. Reinterpreting the trailing indices as the
  // rows of column-major matrices lets BLAS transform one tensor index at a
  // time without transposition copies.
  dgemm_(&no_transpose, &no_transpose, &n3, &n, &n, &alpha,
         dense_source.data(), &n3, rotation.data(), &n, &beta, first.data(),
         &n3);
  std::vector<double>().swap(dense_source);
  std::vector<double> second(fourth, 0.0);
  for (std::size_t p = 0; p < orbitals; ++p)
    dgemm_(&no_transpose, &no_transpose, &n2, &n, &n, &alpha,
           first.data() + p * cube, &n2, rotation.data(), &n, &beta,
           second.data() + p * cube, &n2);
  for (std::size_t pair = 0; pair < square; ++pair)
    dgemm_(&no_transpose, &no_transpose, &n, &n, &n, &alpha,
           second.data() + pair * square, &n, rotation.data(), &n, &beta,
           first.data() + pair * square, &n);
  dgemm_(&transpose, &no_transpose, &n, &n3, &n, &alpha, rotation.data(), &n,
         first.data(), &n, &beta, second.data(), &n);

  SymmetricEri4 result(orbitals);
  std::size_t first_pair = 0;
  for (std::size_t p = 0; p < orbitals; ++p)
    for (std::size_t q = 0; q <= p; ++q, ++first_pair) {
      std::size_t second_pair = 0;
      for (std::size_t r = 0; r < orbitals; ++r)
        for (std::size_t s = 0; s <= r; ++s, ++second_pair)
          if (second_pair <= first_pair)
            result(p, q, r, s) =
                second[((p * orbitals + q) * orbitals + r) * orbitals + s];
    }
  for (const auto value : result.data())
    if (!std::isfinite(value))
      throw std::runtime_error(
          "semicanonical ERI rotation produced non-finite data");
  return result;
}

double maximum_block_offdiagonal(const Matrix &fock,
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

} // namespace

SemicanonicalTriplesResult solve_localized_triples_semicanonical(
    const SpatialRhfHamiltonian &localized_hamiltonian,
    const Matrix &localized_t1, const Tensor4 &localized_t2,
    const CanonicalTriplesOptions &options,
    const std::size_t maximum_additional_memory_bytes) {
  const auto occupied = localized_hamiltonian.nocc;
  const auto virtuals = localized_hamiltonian.nvirt;
  const auto orbitals = occupied + virtuals;
  if (occupied == 0 || virtuals == 0 ||
      localized_hamiltonian.fock.dimensions() !=
          std::array<std::size_t, 2>{orbitals, orbitals} ||
      localized_hamiltonian.hcore.dimensions() !=
          std::array<std::size_t, 2>{orbitals, orbitals} ||
      localized_hamiltonian.eri_chemist.dimensions() !=
          std::array<std::size_t, 4>{orbitals, orbitals, orbitals, orbitals})
    throw std::invalid_argument(
        "localized Hamiltonian dimensions are inconsistent");
  const auto stored_eri_bytes = checked_product(
      localized_hamiltonian.eri_chemist.size(), sizeof(double),
      "semicanonical stored ERI bytes");
  const auto logical_eri_bytes = checked_product(
      localized_hamiltonian.eri_chemist.logical_size(), sizeof(double),
      "semicanonical logical ERI bytes");
  const auto two_eri_bytes = checked_product(
      logical_eri_bytes, 2, "semicanonical ERI work bytes");
  if (two_eri_bytes >
      std::numeric_limits<std::size_t>::max() - stored_eri_bytes)
    throw std::overflow_error("semicanonical rotation peak size overflows");
  const auto rotation_eri_bytes = two_eri_bytes + stored_eri_bytes;
  const auto transformed_t2_bytes = checked_product(
      localized_t2.size(), sizeof(double), "semicanonical T2 bytes");
  if (stored_eri_bytes >
      std::numeric_limits<std::size_t>::max() - transformed_t2_bytes)
    throw std::overflow_error("semicanonical peak memory size overflows");
  const auto estimated_peak_additional_bytes =
      std::max(rotation_eri_bytes,
               stored_eri_bytes + transformed_t2_bytes);
  if (estimated_peak_additional_bytes > maximum_additional_memory_bytes)
    throw std::runtime_error(
        "semicanonical triples rotation requires " +
        std::to_string(static_cast<double>(estimated_peak_additional_bytes) /
                       (1024.0 * 1024.0)) +
        " MiB of additional memory per rank, exceeding the configured "
        "memory guard");

  const auto occupied_eigen = modernqc::linalg::diagonalize_symmetric(
      block(localized_hamiltonian.fock, 0, occupied));
  const auto virtual_eigen = modernqc::linalg::diagonalize_symmetric(
      block(localized_hamiltonian.fock, occupied, virtuals));
  const auto rotation = block_diagonal_rotation(occupied_eigen.eigenvectors,
                                                virtual_eigen.eigenvectors);
  SpatialRhfHamiltonian semicanonical{
      rotate_one_body(localized_hamiltonian.fock, rotation),
      rotate_one_body(localized_hamiltonian.hcore, rotation),
      rotate_eri(localized_hamiltonian.eri_chemist, rotation),
      occupied,
      virtuals,
      localized_hamiltonian.frozen_core,
      localized_hamiltonian.reference_energy,
      localized_hamiltonian.nuclear_energy,
      localized_hamiltonian.orthonormality_error,
      localized_hamiltonian.core_indices,
      localized_hamiltonian.active_occ_indices,
      localized_hamiltonian.virtual_indices};

  // project_ccsd_amplitudes expects target-by-source transformations. The
  // eigenvector matrices are source-by-target, so transpose them here.
  const auto transformed = project_ccsd_amplitudes(
      localized_t1, localized_t2,
      modernqc::linalg::transpose(occupied_eigen.eigenvectors),
      modernqc::linalg::transpose(virtual_eigen.eigenvectors));
  SemicanonicalTriplesResult result;
  result.localized_fock_offdiagonal =
      maximum_block_offdiagonal(localized_hamiltonian.fock, occupied);
  result.semicanonical_fock_offdiagonal =
      maximum_block_offdiagonal(semicanonical.fock, occupied);
  result.transformed_t2_antisymmetry_error = transformed.t2_antisymmetry_error;
  result.rotation_peak_additional_mib =
      static_cast<double>(estimated_peak_additional_bytes) / (1024.0 * 1024.0);
  result.triples = solve_canonical_triples_streamed(
      semicanonical, transformed.t1, transformed.t2, options);
  return result;
}

} // namespace localized_cc
