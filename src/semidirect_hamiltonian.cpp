#include "localized_cc/semidirect_hamiltonian.hpp"

#include "localized_cc/parallel.hpp"
#include "hermitian_one_body.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace localized_cc {
namespace {

extern "C" {
void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb, const double* beta, double* c,
            const int* ldc);
}

std::size_t checked_product(std::initializer_list<std::size_t> factors,
                            const char* label) {
  std::size_t result = 1;
  for (const auto factor : factors) {
    if (factor != 0 && result > std::numeric_limits<std::size_t>::max() / factor)
      throw std::overflow_error(std::string(label) + " size overflows");
    result *= factor;
  }
  return result;
}

std::size_t checked_sum(std::initializer_list<std::size_t> terms,
                        const char* label) {
  std::size_t result = 0;
  for (const auto term : terms) {
    if (term > std::numeric_limits<std::size_t>::max() - result)
      throw std::overflow_error(std::string(label) + " size overflows");
    result += term;
  }
  return result;
}

int checked_blas_dimension(const std::size_t value, const char* label) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error(std::string(label) +
                              " exceeds the BLAS integer range");
  return static_cast<int>(value);
}

std::size_t ceil_divide(const std::size_t numerator,
                        const std::size_t denominator) {
  if (denominator == 0)
    throw std::invalid_argument("semidirect division by zero");
  return numerator == 0 ? 0 : 1 + (numerator - 1) / denominator;
}

modernqc::linalg::Matrix concatenate_coefficients(
    const modernqc::scf::RhfResult& rhf,
    const modernqc::linalg::Matrix& occupied,
    const modernqc::linalg::Matrix& virtuals,
    const std::size_t frozen_core_count) {
  const auto nao = rhf.coefficients.rows();
  if (occupied.rows() != nao || virtuals.rows() != nao ||
      frozen_core_count > rhf.coefficients.columns())
    throw std::invalid_argument("semidirect MO coefficient dimensions mismatch");
  const auto nmo = frozen_core_count + occupied.columns() + virtuals.columns();
  modernqc::linalg::Matrix result(nao, nmo);
  for (std::size_t mu = 0; mu < nao; ++mu) {
    for (std::size_t p = 0; p < frozen_core_count; ++p)
      result(mu, p) = rhf.coefficients(mu, p);
    for (std::size_t p = 0; p < occupied.columns(); ++p)
      result(mu, frozen_core_count + p) = occupied(mu, p);
    for (std::size_t p = 0; p < virtuals.columns(); ++p)
      result(mu, frozen_core_count + occupied.columns() + p) =
          virtuals(mu, p);
  }
  return result;
}

double orthonormality_error(const modernqc::linalg::Matrix& coefficients,
                            const modernqc::linalg::Matrix& overlap) {
  const auto nao = coefficients.rows();
  const auto nmo = coefficients.columns();
  if (overlap.rows() != nao || overlap.columns() != nao)
    throw std::invalid_argument("semidirect AO overlap dimensions mismatch");
  double error = 0.0;
  for (std::size_t p = 0; p < nmo; ++p)
    for (std::size_t q = 0; q < nmo; ++q) {
      double value = 0.0;
      for (std::size_t mu = 0; mu < nao; ++mu)
        for (std::size_t nu = 0; nu < nao; ++nu)
          value += coefficients(mu, p) * overlap(mu, nu) *
                   coefficients(nu, q);
      error = std::max(error, std::abs(value - (p == q ? 1.0 : 0.0)));
    }
  return error;
}

Matrix transform_one_body(const modernqc::linalg::Matrix& ao,
                          const modernqc::linalg::Matrix& coefficients) {
  const auto nao = coefficients.rows();
  if (ao.rows() != nao || ao.columns() != nao)
    throw std::invalid_argument("semidirect one-body AO dimensions mismatch");
  return detail::project_hermitian_one_body(ao, coefficients).value;
}

} // namespace

SemidirectHamiltonianResult build_semidirect_spatial_hamiltonian(
    const modernqc::scf::RhfResult& rhf,
    modernqc::integrals::Libint2IntegralProvider& integrals,
    const modernqc::linalg::Matrix& active_occupied,
    const modernqc::linalg::Matrix& virtuals,
    const std::size_t frozen_core_count,
    const std::size_t maximum_additional_memory_bytes,
    const double orthonormality_tolerance,
    const double reference_tolerance,
    const std::size_t cooperative_slice_cap_bytes) {
  if (maximum_additional_memory_bytes == 0)
    throw std::invalid_argument("semidirect memory limit must be positive");
  if (cooperative_slice_cap_bytes == 0)
    throw std::invalid_argument(
        "semidirect cooperative slice cap must be positive");
  const auto coefficients = concatenate_coefficients(
      rhf, active_occupied, virtuals, frozen_core_count);
  const auto nao = coefficients.rows();
  const auto nmo = coefficients.columns();
  const auto world = parallel::world();
  if (world.size == 0 || world.rank >= world.size)
    throw std::runtime_error("semidirect MPI topology is invalid");
  const double ortho_error = orthonormality_error(coefficients, rhf.overlap);
  if (!std::isfinite(orthonormality_tolerance) ||
      orthonormality_tolerance < 0.0 || ortho_error > orthonormality_tolerance)
    throw std::invalid_argument("semidirect orbitals are not S-orthonormal");

  const auto second_elements = checked_product(
      {nmo, nao, nao}, "semidirect second-index work buffer");
  const auto third_elements = checked_product(
      {nmo, nmo, nao}, "semidirect third-index work buffer");
  const auto fourth_elements = checked_product(
      {nmo, nmo, nmo}, "semidirect final-index work buffer");
  const auto work_elements = checked_sum(
      {second_elements, third_elements, fourth_elements},
      "semidirect work buffers");
  const auto full_mo_elements = SymmetricEri4::stored_elements(nmo);
  const auto active_nmo = active_occupied.columns() + virtuals.columns();
  const auto active_mo_elements = SymmetricEri4::stored_elements(active_nmo);
  const auto work_bytes = checked_product({work_elements, sizeof(double)},
                                          "semidirect work bytes");
  const auto full_mo_bytes = checked_product({full_mo_elements, sizeof(double)},
                                             "full-MO ERI bytes");
  const auto active_mo_bytes = checked_product(
      {active_mo_elements, sizeof(double)}, "active-MO ERI bytes");
  const auto reserved_bytes = checked_sum(
      {work_bytes, full_mo_bytes, active_mo_bytes},
      "semidirect reserved memory");
  if (reserved_bytes >= maximum_additional_memory_bytes)
    throw std::runtime_error(
        "semidirect Hamiltonian fixed tensors exceed the per-rank memory "
        "guard");

  // Repeating every Libint2 shell quartet independently on every rank made
  // the first-index transform slower as MPI ranks increased. Use the fewest
  // cooperating orbital groups allowed by a fixed per-rank slice cap. This
  // minimizes repeated integral evaluation while keeping the partial
  // first-index buffer bounded independently of the user memory ceiling.
  const auto one_first_index_elements =
      checked_product({nao, nao, nao}, "one first-index slice");
  const auto one_first_index_bytes = checked_product(
      {one_first_index_elements, sizeof(double)},
      "one first-index slice bytes");
  const auto allowed_first_bytes = std::min(
      cooperative_slice_cap_bytes,
      maximum_additional_memory_bytes - reserved_bytes);
  if (one_first_index_bytes > allowed_first_bytes)
    throw std::runtime_error(
        "one semidirect first-index slice exceeds the bounded cooperative "
        "buffer limit");
  const auto maximum_group_first_indices =
      std::max<std::size_t>(1, allowed_first_bytes / one_first_index_bytes);
  const auto memory_limited_groups =
      ceil_divide(nmo, maximum_group_first_indices);
  const auto group_count = std::min(
      std::min(world.size, nmo),
      std::max<std::size_t>(1, memory_limited_groups));
  const auto group_color = world.rank % group_count;
  const auto group_rank = world.rank / group_count;
  const auto group_size =
      1 + (world.size - 1 - group_color) / group_count;
  std::vector<std::size_t> group_first_indices;
  for (std::size_t p = group_color; p < nmo; p += group_count)
    group_first_indices.push_back(p);
  std::size_t local_first_index_count = 0;
  if (group_rank < group_first_indices.size())
    local_first_index_count =
        1 + (group_first_indices.size() - 1 - group_rank) / group_size;

  const auto first_elements = checked_product(
      {group_first_indices.size(), one_first_index_elements},
      "cooperative first-index slice");
  const auto first_bytes = checked_product(
      {first_elements, sizeof(double)}, "first-index slice bytes");
  if (first_bytes > allowed_first_bytes)
    throw std::runtime_error(
        "available MPI ranks cannot keep the cooperative first-index slice "
        "within its bounded memory cap");
  const auto estimated_peak = checked_sum(
      {first_bytes, work_bytes, full_mo_bytes, active_mo_bytes},
      "semidirect estimated peak");
  if (estimated_peak > maximum_additional_memory_bytes)
    throw std::runtime_error(
        "semidirect Hamiltonian estimated peak exceeds the per-rank memory "
        "guard");

  auto first = integrals.compute_first_index_transformed_eris(
      coefficients, group_first_indices,
      maximum_additional_memory_bytes - reserved_bytes,
      modernqc::integrals::ParallelBuildOptions{group_rank, group_size, 1});
  parallel::sum_in_place_group(first.values, group_color);
  const auto group_evaluated_shell_quartets =
      parallel::sum_group(first.evaluated_shell_quartets, group_color);
  SymmetricEri4 eri_mo(nmo);
  std::vector<double> second(second_elements, 0.0);
  std::vector<double> third(third_elements, 0.0);
  std::vector<double> fourth(fourth_elements, 0.0);
  const auto nao_squared = checked_product({nao, nao}, "AO-pair extent");
  const auto nmo_squared = checked_product({nmo, nmo}, "MO-pair extent");
  const int blas_nao = checked_blas_dimension(nao, "AO count");
  const int blas_nmo = checked_blas_dimension(nmo, "MO count");
  const int blas_nao_squared =
      checked_blas_dimension(nao_squared, "AO-pair count");
  const int blas_nmo_squared =
      checked_blas_dimension(nmo_squared, "MO-pair count");
  constexpr char no_transpose = 'N';
  constexpr char transpose = 'T';
  constexpr double alpha = 1.0;
  constexpr double beta = 0.0;

  for (std::size_t owned = group_rank; owned < group_first_indices.size();
       owned += group_size) {
    const double* first_slice =
        first.values.data() + owned * nao * nao_squared;
    dgemm_(&no_transpose, &no_transpose, &blas_nao_squared, &blas_nmo,
           &blas_nao, &alpha, first_slice, &blas_nao_squared,
           coefficients.data(), &blas_nao, &beta, second.data(),
           &blas_nao_squared);
    for (std::size_t q = 0; q < nmo; ++q) {
      const double* second_q = second.data() + q * nao_squared;
      double* third_q = third.data() + q * nmo * nao;
      dgemm_(&no_transpose, &no_transpose, &blas_nao, &blas_nmo, &blas_nao,
             &alpha, second_q, &blas_nao, coefficients.data(), &blas_nao,
             &beta, third_q, &blas_nao);
    }
    const auto p = group_first_indices[owned];
    dgemm_(&transpose, &no_transpose, &blas_nmo, &blas_nmo_squared,
           &blas_nao, &alpha, coefficients.data(), &blas_nao, third.data(),
           &blas_nao, &beta, fourth.data(), &blas_nmo);
    std::size_t first_pair = p * (p + 1) / 2;
    for (std::size_t q = 0; q <= p; ++q, ++first_pair) {
      std::size_t second_pair = 0;
      for (std::size_t r = 0; r < nmo; ++r)
        for (std::size_t s = 0; s <= r; ++s, ++second_pair)
          if (second_pair <= first_pair)
            eri_mo(p, q, r, s) = fourth[(q * nmo + r) * nmo + s];
    }
  }
  parallel::sum_in_place(eri_mo.data());
  auto hcore_mo = transform_one_body(rhf.core_hamiltonian, coefficients);
  auto hamiltonian = assemble_frozen_core_spatial_hamiltonian(
      FrozenCoreMoInput{std::move(hcore_mo), std::move(eri_mo),
                        frozen_core_count, active_occupied.columns(),
                        virtuals.columns(), rhf.nuclear_repulsion_energy,
                        rhf.total_energy, ortho_error, true},
      reference_tolerance);
  return SemidirectHamiltonianResult{
      std::move(hamiltonian),
      SemidirectTransformDiagnostics{
          .ao_count = nao,
          .mo_count = nmo,
          .local_first_indices = local_first_index_count,
          .cooperative_first_index_groups = group_count,
          .ranks_in_first_index_group = group_size,
          .first_indices_in_group = group_first_indices.size(),
          .first_index_slice_bytes = first_bytes,
          .transform_work_bytes = work_bytes,
          .full_mo_eri_bytes = full_mo_bytes,
          .active_mo_eri_bytes = active_mo_bytes,
          .estimated_peak_bytes = estimated_peak,
          .unique_shell_quartets = first.unique_shell_quartets,
          .evaluated_shell_quartets = group_evaluated_shell_quartets}};
}

} // namespace localized_cc
