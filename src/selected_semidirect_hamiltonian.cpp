#include "localized_cc/selected_semidirect_hamiltonian.hpp"

#include "localized_cc/parallel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <map>
#include <ranges>
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

std::size_t checked_sum(const std::size_t left, const std::size_t right,
                        const char* label) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error(std::string(label) + " size overflows");
  return left + right;
}

std::size_t ceil_divide(const std::size_t numerator,
                        const std::size_t denominator) {
  if (denominator == 0)
    throw std::invalid_argument("selected semidirect division by zero");
  return numerator == 0 ? 0 : 1 + (numerator - 1) / denominator;
}

int blas_int(const std::size_t value, const char* label) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error(std::string(label) +
                              " exceeds the BLAS integer range");
  return static_cast<int>(value);
}

modernqc::linalg::Matrix concatenate_active_coefficients(
    const modernqc::linalg::Matrix& occupied,
    const modernqc::linalg::Matrix& virtuals) {
  if (occupied.rows() == 0 || occupied.rows() != virtuals.rows() ||
      occupied.columns() == 0 || virtuals.columns() == 0)
    throw std::invalid_argument(
        "selected semidirect orbital coefficient dimensions are invalid");
  modernqc::linalg::Matrix result(
      occupied.rows(), occupied.columns() + virtuals.columns());
  for (std::size_t mu = 0; mu < result.rows(); ++mu) {
    for (std::size_t i = 0; i < occupied.columns(); ++i)
      result(mu, i) = occupied(mu, i);
    for (std::size_t a = 0; a < virtuals.columns(); ++a)
      result(mu, occupied.columns() + a) = virtuals(mu, a);
  }
  return result;
}

double orthonormality_error(const modernqc::linalg::Matrix& coefficients,
                            const modernqc::linalg::Matrix& overlap) {
  if (overlap.rows() != coefficients.rows() ||
      overlap.columns() != coefficients.rows())
    throw std::invalid_argument(
        "selected semidirect AO overlap dimensions mismatch");
  const auto metric = modernqc::linalg::multiply(
      modernqc::linalg::transpose(coefficients),
      modernqc::linalg::multiply(overlap, coefficients));
  double error = 0.0;
  for (std::size_t p = 0; p < metric.rows(); ++p)
    for (std::size_t q = 0; q < metric.columns(); ++q)
      error = std::max(error,
                       std::abs(metric(p, q) - (p == q ? 1.0 : 0.0)));
  return error;
}

Matrix transform_one_body(const modernqc::linalg::Matrix& ao,
                          const modernqc::linalg::Matrix& coefficients) {
  if (ao.rows() != coefficients.rows() ||
      ao.columns() != coefficients.rows())
    throw std::invalid_argument(
        "selected semidirect one-body AO dimensions mismatch");
  const auto transformed = modernqc::linalg::multiply(
      modernqc::linalg::transpose(coefficients),
      modernqc::linalg::multiply(ao, coefficients));
  Matrix result({transformed.rows(), transformed.columns()});
  std::copy(transformed.data(), transformed.data() + transformed.size(),
            result.data().begin());
  return result;
}

std::size_t token(const CcsdOrbitalSpace space, const std::size_t monomer,
                  const std::size_t monomers) {
  if (monomer >= monomers)
    throw std::out_of_range(
        "selected semidirect monomer label is out of range");
  return (space == CcsdOrbitalSpace::virtual_space ? monomers : 0) +
         monomer;
}

const std::vector<std::size_t>& axis_group(
    const CcsdIntegralMonomerBlock& descriptor, const std::size_t axis,
    const std::vector<std::vector<std::size_t>>& occupied_groups,
    const std::vector<std::vector<std::size_t>>& virtual_groups) {
  return descriptor.spaces.at(axis) == CcsdOrbitalSpace::occupied
             ? occupied_groups.at(descriptor.monomers.at(axis))
             : virtual_groups.at(descriptor.monomers.at(axis));
}

std::size_t active_orbital(const CcsdOrbitalSpace space,
                           const std::size_t relative,
                           const std::size_t occupied) {
  return space == CcsdOrbitalSpace::occupied ? relative
                                              : occupied + relative;
}

modernqc::linalg::Matrix gather_columns(
    const modernqc::linalg::Matrix& coefficients,
    const CcsdOrbitalSpace space,
    const std::vector<std::size_t>& relative_orbitals,
    const std::size_t occupied) {
  modernqc::linalg::Matrix result(coefficients.rows(),
                                  relative_orbitals.size());
  for (std::size_t column = 0; column < relative_orbitals.size(); ++column) {
    const auto source = active_orbital(space, relative_orbitals[column],
                                       occupied);
    for (std::size_t mu = 0; mu < coefficients.rows(); ++mu)
      result(mu, column) = coefficients(mu, source);
  }
  return result;
}

} // namespace

SelectedSpatialHamiltonian build_selected_semidirect_spatial_hamiltonian(
    const modernqc::scf::RhfResult& rhf,
    modernqc::integrals::Libint2IntegralProvider& integrals,
    const modernqc::linalg::Matrix& active_occupied,
    const modernqc::linalg::Matrix& virtuals,
    const SelectedCcsdManifoldPlan& plan,
    const std::size_t maximum_additional_memory_bytes,
    SelectedSemidirectTransformDiagnostics& diagnostics,
    const double orthonormality_tolerance) {
  if (maximum_additional_memory_bytes == 0)
    throw std::invalid_argument(
        "selected semidirect memory limit must be positive");
  if (!plan.dependency_status.ready_for_sparse_ccsd())
    throw std::invalid_argument(
        "selected semidirect transform requires a closed dependency plan");
  const auto coefficients =
      concatenate_active_coefficients(active_occupied, virtuals);
  const auto nao = coefficients.rows();
  const auto occupied = active_occupied.columns();
  const auto virtual_count = virtuals.columns();
  const auto active_mos = coefficients.columns();
  const auto ortho_error = orthonormality_error(coefficients, rhf.overlap);
  if (!std::isfinite(orthonormality_tolerance) ||
      orthonormality_tolerance < 0.0 || ortho_error > orthonormality_tolerance)
    throw std::invalid_argument(
        "selected semidirect orbitals are not S-orthonormal");

  std::size_t maximum_work_bytes = 0;
  std::size_t group_count = 0;
  std::size_t group_size = 0;
  std::size_t group_first_count = 0;
  std::size_t first_bytes = 0;
  std::size_t avoided_full_bytes = 0;
  std::uint64_t unique_quartets = 0;
  std::uint64_t evaluated_quartets = 0;

  std::size_t maximum_spatial_group = 0;
  const auto include_spatial_groups =
      [&](const std::vector<std::vector<std::size_t>>& spin_groups,
          const char* space_name) {
        for (const auto& group : spin_groups) {
          if (group.size() % 2 != 0)
            throw std::invalid_argument(
                std::string("selected semidirect ") + space_name +
                " group does not contain complete alpha/beta pairs");
          maximum_spatial_group =
              std::max(maximum_spatial_group, group.size() / 2);
        }
      };
  include_spatial_groups(plan.occupied_orbitals_by_monomer, "occupied");
  include_spatial_groups(plan.virtual_orbitals_by_monomer, "virtual");
  if (maximum_spatial_group == 0)
    throw std::invalid_argument(
        "selected semidirect plan has an empty spatial orbital space");
  const auto maximum_second_elements = checked_product(
      {nao, nao, maximum_spatial_group},
      "selected maximum second-index work");
  const auto maximum_third_elements = checked_product(
      {nao, maximum_spatial_group, maximum_spatial_group},
      "selected maximum third-index work");
  const auto maximum_fourth_elements = checked_product(
      {maximum_spatial_group, maximum_spatial_group,
       maximum_spatial_group},
      "selected maximum final-index work");
  const auto reserved_work_bytes = checked_product(
      {checked_sum(checked_sum(maximum_second_elements,
                               maximum_third_elements,
                               "selected maximum transform work"),
                   maximum_fourth_elements,
                   "selected maximum transform work"),
       sizeof(double)},
      "selected maximum transform work");

  auto selected_integrals = SelectedIntegralStore(
      plan, occupied, virtual_count,
      [&](const std::vector<CcsdIntegralMonomerBlock>& descriptors,
          const std::vector<std::vector<std::size_t>>& occupied_groups,
          const std::vector<std::vector<std::size_t>>& virtual_groups) {
        std::vector<std::vector<double>> blocks;
        blocks.reserve(descriptors.size());
        std::size_t selected_bytes = 0;
        for (const auto& descriptor : descriptors) {
          blocks.emplace_back(descriptor.element_count, 0.0);
          selected_bytes = checked_sum(
              selected_bytes,
              checked_product({descriptor.element_count, sizeof(double)},
                              "selected integral block"),
              "selected integral storage");
        }
        const auto one_first_elements = checked_product(
            {nao, nao, nao}, "selected first-index slice");
        const auto one_first_bytes = checked_product(
            {one_first_elements, sizeof(double)},
            "selected first-index slice");
        const auto fixed_bytes = checked_sum(
            selected_bytes, reserved_work_bytes,
            "selected semidirect fixed memory");
        if (checked_sum(fixed_bytes, one_first_bytes,
                        "selected semidirect minimum memory") >
            maximum_additional_memory_bytes)
          throw std::runtime_error(
              "selected semidirect integral store, transform workspace, "
              "and one first-index slice exceed the per-rank memory guard");
        const auto maximum_first_indices = std::max<std::size_t>(
            1, (maximum_additional_memory_bytes - fixed_bytes) /
                   one_first_bytes);
        const auto world = parallel::world();
        group_count = std::min(
            std::min(world.size, active_mos),
            std::max<std::size_t>(
                1, ceil_divide(active_mos, maximum_first_indices)));
        const auto color = world.rank % group_count;
        const auto group_rank = world.rank / group_count;
        group_size = 1 + (world.size - 1 - color) / group_count;
        std::vector<std::size_t> first_indices;
        for (std::size_t p = color; p < active_mos; p += group_count)
          first_indices.push_back(p);
        group_first_count = first_indices.size();
        first_bytes = checked_product(
            {first_indices.size(), one_first_bytes},
            "selected cooperative first-index storage");
        if (checked_sum(fixed_bytes, first_bytes,
                        "selected semidirect first-index memory") >
            maximum_additional_memory_bytes)
          throw std::runtime_error(
              "selected cooperative first-index storage exceeds the "
              "per-rank memory guard");

        auto first = integrals.compute_first_index_transformed_eris(
            coefficients, first_indices,
            maximum_additional_memory_bytes - selected_bytes -
                reserved_work_bytes,
            modernqc::integrals::ParallelBuildOptions{group_rank,
                                                       group_size, 1});
        parallel::sum_in_place_group(first.values, color);
        unique_quartets = first.unique_shell_quartets;
        const auto group_evaluated =
            parallel::sum_group(first.evaluated_shell_quartets, color);
        evaluated_quartets = parallel::sum_group(
            group_rank == 0 ? group_evaluated : 0, 0);

        constexpr char no_transpose = 'N';
        constexpr char transpose = 'T';
        constexpr double alpha = 1.0;
        constexpr double beta = 0.0;
        const auto nao_squared = checked_product(
            {nao, nao}, "selected AO-pair extent");
        const int blas_nao = blas_int(nao, "AO count");
        const int blas_nao_squared =
            blas_int(nao_squared, "AO-pair count");

        for (std::size_t owned = group_rank; owned < first_indices.size();
             owned += group_size) {
          const auto p = first_indices[owned];
          const double* first_slice =
              first.values.data() + owned * one_first_elements;
          std::vector<std::size_t> p_descriptors;
          for (std::size_t block = 0; block < descriptors.size(); ++block) {
            const auto& descriptor = descriptors[block];
            if ((descriptor.spaces[0] == CcsdOrbitalSpace::occupied) !=
                (p < occupied))
              continue;
            const auto& p_group = axis_group(
                descriptor, 0, occupied_groups, virtual_groups);
            const auto relative =
                descriptor.spaces[0] == CcsdOrbitalSpace::occupied
                    ? p
                    : p - occupied;
            if (std::ranges::find(p_group, relative) == p_group.end())
              continue;
            p_descriptors.push_back(block);
          }
          if (p_descriptors.empty())
            continue;

          std::map<std::size_t, std::vector<std::size_t>> by_q;
          for (const auto block : p_descriptors) {
            const auto& descriptor = descriptors[block];
            by_q[token(descriptor.spaces[1], descriptor.monomers[1],
                       plan.monomer_count)]
                .push_back(block);
          }
          for (const auto& [q_token, q_descriptors] : by_q) {
            static_cast<void>(q_token);
            const auto& q_descriptor = descriptors[q_descriptors.front()];
            const auto& q_group = axis_group(
                q_descriptor, 1, occupied_groups, virtual_groups);
            const auto q_coefficients = gather_columns(
                coefficients, q_descriptor.spaces[1], q_group, occupied);
            const auto second_elements = checked_product(
                {nao_squared, q_group.size()},
                "selected second-index work");
            std::vector<double> second(second_elements);
            const int blas_q = blas_int(q_group.size(), "selected q extent");
            dgemm_(&no_transpose, &no_transpose, &blas_nao_squared, &blas_q,
                   &blas_nao, &alpha, first_slice, &blas_nao_squared,
                   q_coefficients.data(), &blas_nao, &beta, second.data(),
                   &blas_nao_squared);

            std::map<std::size_t, std::vector<std::size_t>> by_r;
            for (const auto block : q_descriptors) {
              const auto& descriptor = descriptors[block];
              by_r[token(descriptor.spaces[2], descriptor.monomers[2],
                         plan.monomer_count)]
                  .push_back(block);
            }
            for (const auto& [r_token, r_descriptors] : by_r) {
              static_cast<void>(r_token);
              const auto& r_descriptor = descriptors[r_descriptors.front()];
              const auto& r_group = axis_group(
                  r_descriptor, 2, occupied_groups, virtual_groups);
              const auto r_coefficients = gather_columns(
                  coefficients, r_descriptor.spaces[2], r_group, occupied);
              const auto third_elements = checked_product(
                  {q_group.size(), r_group.size(), nao},
                  "selected third-index work");
              std::vector<double> third(third_elements);
              const int blas_r =
                  blas_int(r_group.size(), "selected r extent");
              for (std::size_t q = 0; q < q_group.size(); ++q) {
                const double* second_q =
                    second.data() + q * nao_squared;
                double* third_q =
                    third.data() + q * r_group.size() * nao;
                dgemm_(&no_transpose, &no_transpose, &blas_nao, &blas_r,
                       &blas_nao, &alpha, second_q, &blas_nao,
                       r_coefficients.data(), &blas_nao, &beta, third_q,
                       &blas_nao);
              }

              for (const auto block : r_descriptors) {
                const auto& descriptor = descriptors[block];
                const auto& s_group = axis_group(
                    descriptor, 3, occupied_groups, virtual_groups);
                const auto s_coefficients = gather_columns(
                    coefficients, descriptor.spaces[3], s_group, occupied);
                const auto qr = checked_product(
                    {q_group.size(), r_group.size()},
                    "selected q-r pair extent");
                const auto fourth_elements = checked_product(
                    {qr, s_group.size()}, "selected final-index work");
                std::vector<double> fourth(fourth_elements);
                const int blas_s =
                    blas_int(s_group.size(), "selected s extent");
                const int blas_qr = blas_int(qr, "selected q-r extent");
                dgemm_(&transpose, &no_transpose, &blas_s, &blas_qr,
                       &blas_nao, &alpha, s_coefficients.data(), &blas_nao,
                       third.data(), &blas_nao, &beta, fourth.data(), &blas_s);

                const auto& p_group = axis_group(
                    descriptor, 0, occupied_groups, virtual_groups);
                const auto p_relative =
                    descriptor.spaces[0] == CcsdOrbitalSpace::occupied
                        ? p
                        : p - occupied;
                const auto p_position =
                    std::ranges::find(p_group, p_relative);
                if (p_position == p_group.end())
                  throw std::logic_error(
                      "selected first-index owner lost its descriptor");
                const auto p_local = static_cast<std::size_t>(
                    std::distance(p_group.begin(), p_position));
                auto& values = blocks[block];
                for (std::size_t q = 0; q < q_group.size(); ++q)
                  for (std::size_t r = 0; r < r_group.size(); ++r)
                    for (std::size_t s = 0; s < s_group.size(); ++s) {
                      const auto destination =
                          (((p_local * q_group.size()) + q) *
                               r_group.size() +
                           r) *
                              s_group.size() +
                          s;
                      values[destination] =
                          fourth[(q * r_group.size() + r) *
                                     s_group.size() +
                                 s];
                    }
                const auto work_bytes = checked_product(
                    {checked_sum(
                         checked_sum(second.size(), third.size(),
                                     "selected transform work"),
                         fourth.size(), "selected transform work"),
                     sizeof(double)},
                    "selected transform work");
                maximum_work_bytes = std::max(maximum_work_bytes, work_bytes);
              }
            }
          }
        }
        for (auto& block : blocks)
          parallel::sum_in_place(block);
        avoided_full_bytes = checked_product(
            {SymmetricEri4::stored_elements(active_mos), sizeof(double)},
            "avoided full active MO ERI storage");
        return blocks;
      });

  diagnostics = SelectedSemidirectTransformDiagnostics{
      .ao_count = nao,
      .active_mo_count = active_mos,
      .cooperative_first_index_groups = group_count,
      .ranks_in_first_index_group = group_size,
      .first_indices_in_group = group_first_count,
      .first_index_slice_bytes = first_bytes,
      .maximum_selected_transform_work_bytes = maximum_work_bytes,
      .requested_spatial_integral_blocks =
          selected_integrals.spatial_component_request_count(),
      .symmetry_unique_spatial_integral_blocks =
          selected_integrals.block_count(),
      .selected_integral_bytes = selected_integrals.selected_bytes(),
      .selected_integral_tuple_lookup_bytes =
          selected_integrals.tuple_lookup_bytes(),
      .avoided_full_active_mo_eri_bytes = avoided_full_bytes,
      .estimated_peak_bytes = std::max(
          checked_sum(
              selected_integrals.selected_bytes(),
              checked_sum(first_bytes, maximum_work_bytes,
                          "selected semidirect transform peak"),
              "selected semidirect transform peak"),
          checked_sum(selected_integrals.selected_bytes(),
                      selected_integrals.tuple_lookup_bytes(),
                      "selected semidirect lookup peak")),
      .unique_shell_quartets = unique_quartets,
      .evaluated_shell_quartets_across_groups = evaluated_quartets};
  if (diagnostics.estimated_peak_bytes > maximum_additional_memory_bytes)
    throw std::runtime_error(
        "selected semidirect measured workspace exceeds the per-rank "
        "memory guard");

  return SelectedSpatialHamiltonian{
      .fock = transform_one_body(rhf.fock, coefficients),
      .hcore = transform_one_body(rhf.core_hamiltonian, coefficients),
      .nocc = occupied,
      .nvirt = virtual_count,
      .reference_energy = rhf.total_energy,
      .integrals = std::move(selected_integrals)};
}

} // namespace localized_cc
