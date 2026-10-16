#include "localized_cc/ccsd.hpp"
#include "localized_cc/cuda_triples.hpp"
#include "localized_cc/diis.hpp"
#include "localized_cc/distributed_pair_tensor.hpp"
#include "localized_cc/parallel.hpp"
#include "localized_cc/pm_symmetry.hpp"
#include "localized_cc/selected_ccsd_storage.hpp"
#include "spatial_pair_terms.hpp"
#include "pm_orbit_plan.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

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

using SteadyClock = std::chrono::steady_clock;

std::size_t checked_memory_product(std::size_t left, std::size_t right);
std::size_t checked_memory_sum(std::size_t left, std::size_t right);

int checked_blas_dimension(const std::size_t value, const char* const label) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error(std::string("CCSD ") + label +
                              " exceeds the BLAS integer range");
  return static_cast<int>(value);
}

double elapsed_seconds(const SteadyClock::time_point begin,
                       const SteadyClock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

void add_timing(CcsdIterationTiming& total,
                const CcsdIterationTiming& iteration) {
  total.intermediates_seconds += iteration.intermediates_seconds;
  total.intermediate_mixed_pair_tasks +=
      iteration.intermediate_mixed_pair_tasks;
  total.intermediate_blas_contractions +=
      iteration.intermediate_blas_contractions;
  total.singles_seconds += iteration.singles_seconds;
  total.doubles_seconds += iteration.doubles_seconds;
  total.doubles_occupied_pair_tasks +=
      iteration.doubles_occupied_pair_tasks;
  total.doubles_blas_contractions += iteration.doubles_blas_contractions;
  total.wabef_seconds += iteration.wabef_seconds;
  total.wabef_build_seconds += iteration.wabef_build_seconds;
  total.wabef_contraction_seconds += iteration.wabef_contraction_seconds;
  total.wabef_accumulation_seconds +=
      iteration.wabef_accumulation_seconds;
  total.wabef_external_tiles += iteration.wabef_external_tiles;
  total.wabef_internal_panels += iteration.wabef_internal_panels;
  total.wabef_output_tasks += iteration.wabef_output_tasks;
  total.wabef_blas_contractions += iteration.wabef_blas_contractions;
  total.wabef_factorized_cached_evaluations +=
      iteration.wabef_factorized_cached_evaluations;
  total.residual_seconds += iteration.residual_seconds;
  total.diis_seconds += iteration.diis_seconds;
  total.diis_error_dot_products += iteration.diis_error_dot_products;
  total.diis_history_vector_accumulations +=
      iteration.diis_history_vector_accumulations;
  total.energy_seconds += iteration.energy_seconds;
  total.iteration_seconds += iteration.iteration_seconds;
}

void validate_projection_mask(const CcsdProjectionMask& mask,
                              const std::size_t occupied,
                              const std::size_t virtuals) {
  const auto singles = checked_memory_product(occupied, virtuals);
  const auto doubles = checked_memory_product(singles, singles);
  if (mask.singles.size() != singles || mask.doubles.size() != doubles)
    throw std::invalid_argument("CCSD projection-mask dimensions mismatch");
  const auto validate_binary = [](const std::vector<unsigned char>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](const unsigned char value) { return value <= 1U; });
  };
  if (!validate_binary(mask.singles) || !validate_binary(mask.doubles))
    throw std::invalid_argument("CCSD projection mask must be binary");
  const auto selected_singles = static_cast<std::size_t>(
      std::count(mask.singles.begin(), mask.singles.end(), 1U));
  const auto selected_doubles = static_cast<std::size_t>(
      std::count(mask.doubles.begin(), mask.doubles.end(), 1U));
  if (selected_singles != mask.selected_singles ||
      selected_doubles != mask.selected_doubles)
    throw std::invalid_argument("CCSD projection-mask counts mismatch");
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t j = 0; j < occupied; ++j)
      for (std::size_t a = 0; a < virtuals; ++a)
        for (std::size_t b = 0; b < virtuals; ++b) {
          const auto at = [&](const std::size_t first_hole,
                              const std::size_t second_hole,
                              const std::size_t first_particle,
                              const std::size_t second_particle) {
            return mask.doubles[((first_hole * occupied + second_hole) *
                                     virtuals +
                                 first_particle) *
                                    virtuals +
                                second_particle];
          };
          const auto value = at(i, j, a, b);
          if (value != at(j, i, a, b) || value != at(i, j, b, a) ||
              value != at(j, i, b, a))
            throw std::invalid_argument(
                "CCSD projection mask is not antisymmetry closed");
        }
}

void apply_projection_mask(const CcsdProjectionMask& mask, Matrix& t1,
                           Tensor4& t2) {
  for (std::size_t index = 0; index < t1.size(); ++index)
    if (mask.singles[index] == 0U) t1.data()[index] = 0.0;
  for (std::size_t index = 0; index < t2.size(); ++index)
    if (mask.doubles[index] == 0U) t2.data()[index] = 0.0;
}

double projection_excluded_maximum(const CcsdProjectionMask& mask,
                                   const Matrix& t1, const Tensor4& t2) {
  double maximum = 0.0;
  for (std::size_t index = 0; index < t1.size(); ++index)
    if (mask.singles[index] == 0U)
      maximum = std::max(maximum, std::abs(t1.data()[index]));
  for (std::size_t index = 0; index < t2.size(); ++index)
    if (mask.doubles[index] == 0U)
      maximum = std::max(maximum, std::abs(t2.data()[index]));
  return maximum;
}

bool projected_single(const CcsdProjectionMask* const mask,
                      const std::size_t virtuals, const std::size_t i,
                      const std::size_t a) {
  return mask == nullptr || mask->singles[i * virtuals + a] != 0U;
}

bool projected_double(const CcsdProjectionMask* const mask,
                      const std::size_t occupied,
                      const std::size_t virtuals, const std::size_t i,
                      const std::size_t j, const std::size_t a,
                      const std::size_t b) {
  return mask == nullptr ||
         mask->doubles[((i * occupied + j) * virtuals + a) * virtuals + b] !=
             0U;
}

bool projected_tau(const CcsdProjectionMask* const mask,
                   const std::size_t occupied, const std::size_t virtuals,
                   const std::size_t i, const std::size_t j,
                   const std::size_t a, const std::size_t b) {
  return projected_double(mask, occupied, virtuals, i, j, a, b) ||
         (projected_single(mask, virtuals, i, a) &&
          projected_single(mask, virtuals, j, b)) ||
         (projected_single(mask, virtuals, i, b) &&
          projected_single(mask, virtuals, j, a));
}

struct Work {
  Tensor4 tau;
  Tensor4 tt;
  Matrix fae;
  Matrix fmi;
  Matrix fme;
  Tensor4 wmnij;
  Tensor4 wmbej;
};

struct DistributedWork {
  Matrix fae;
  Matrix fmi;
  Matrix fme;
};

template <typename Function>
void distributed_for(const std::size_t count, Function&& function) {
  const auto communicator = parallel::world();
  if (count == 0 || communicator.rank >= count)
    return;
  const auto owned_count =
      1 + (count - 1 - communicator.rank) / communicator.size;
  if (owned_count >
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::overflow_error("distributed loop extent exceeds ptrdiff_t");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t owned = 0;
       owned < static_cast<std::ptrdiff_t>(owned_count); ++owned) {
    const auto index = communicator.rank +
                       static_cast<std::size_t>(owned) * communicator.size;
    function(index);
  }
}

Tensor4 antisym(const Tensor4& input) {
  const auto dimensions = input.dimensions();
  Tensor4 result(dimensions);
  const auto count = input.size();
  if (count >
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::overflow_error("antisymmetrization extent exceeds ptrdiff_t");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t flat = 0; flat < static_cast<std::ptrdiff_t>(count);
       ++flat) {
    auto index = static_cast<std::size_t>(flat);
    const auto b = index % dimensions[3];
    index /= dimensions[3];
    const auto a = index % dimensions[2];
    index /= dimensions[2];
    const auto j = index % dimensions[1];
    const auto i = index / dimensions[1];
    result(i, j, a, b) =
        0.25 * (input(i, j, a, b) - input(j, i, a, b) -
                input(i, j, b, a) + input(j, i, b, a));
  }
  return result;
}

Work make_work(const std::size_t occupied, const std::size_t virtuals) {
  return Work{Tensor4({occupied, occupied, virtuals, virtuals}),
              Tensor4({occupied, occupied, virtuals, virtuals}),
              Matrix({virtuals, virtuals}),
              Matrix({occupied, occupied}),
              Matrix({occupied, virtuals}),
              Tensor4({occupied, occupied, occupied, occupied}),
              Tensor4({occupied, virtuals, virtuals, occupied})};
}

double t2_from_work(const Work& work, const Matrix& t1,
                    const std::size_t i, const std::size_t j,
                    const std::size_t a, const std::size_t b) {
  const auto cross = t1(i, a) * t1(j, b) - t1(i, b) * t1(j, a);
  return work.tau(i, j, a, b) - cross;
}

template <typename Eri>
double selected_panel_eri(
    const Eri& eri, const SelectedIntegralPanelCache* const panel_cache,
    const std::size_t p, const std::size_t q, const std::size_t r,
    const std::size_t s) {
  return panel_cache ? (*panel_cache)(p, q, r, s) : eri(p, q, r, s);
}

template <typename Eri>
Work finish_intermediates(const Matrix& fock, const Eri& eri,
                          const Matrix& t1, Work work,
                          const Tensor4* direct_t2,
                          const std::size_t occupied,
                          const bool use_tiled_intermediates = false,
                          CcsdIterationTiming* const timing = nullptr,
                          const SelectedIntegralPanelCache* const
                              selected_panel_cache = nullptr,
                          const CcsdProjectionMask* const
                              projection_mask = nullptr) {
  const auto virtuals = t1.dimensions()[1];
  const bool truncated_projection =
      projection_mask != nullptr &&
      projection_mask->selected_doubles != work.tau.size();
  const auto eri_value = [&](const std::size_t p, const std::size_t q,
                             const std::size_t r, const std::size_t s) {
    return selected_panel_eri(eri, selected_panel_cache, p, q, r, s);
  };
  const auto eri_declared = [&](const std::size_t p, const std::size_t q,
                                const std::size_t r, const std::size_t s) {
    if constexpr (requires { eri.contains(p, q, r, s); })
      return static_cast<bool>(eri.contains(p, q, r, s));
    return true;
  };

  distributed_for(virtuals * virtuals, [&](const std::size_t flat) {
    const auto a = flat / virtuals;
    const auto e = flat % virtuals;
    double value = a == e ? 0.0 : fock(occupied + a, occupied + e);
    for (std::size_t m = 0; m < occupied; ++m)
      value -= 0.5 * fock(m, occupied + e) * t1(m, a);
    for (std::size_t m = 0; m < occupied; ++m)
      for (std::size_t q = 0; q < virtuals; ++q)
        if (projected_single(projection_mask, virtuals, m, q))
          value += t1(m, q) * eri_value(m, occupied + a, occupied + q,
                                        occupied + e);
    for (std::size_t m = 0; m < occupied; ++m)
      for (std::size_t n = 0; n < occupied; ++n)
        for (std::size_t q = 0; q < virtuals; ++q)
          if (projected_tau(projection_mask, occupied, virtuals, m, n, a,
                            q))
            value -= 0.5 * work.tt(m, n, a, q) *
                     eri_value(m, n, occupied + e, occupied + q);
    work.fae(a, e) = value;
  });
  parallel::sum_in_place(work.fae.data());

  distributed_for(occupied * occupied, [&](const std::size_t flat) {
    const auto m = flat / occupied;
    const auto i = flat % occupied;
    double value = m == i ? 0.0 : fock(m, i);
    for (std::size_t e = 0; e < virtuals; ++e)
      value += 0.5 * t1(i, e) * fock(m, occupied + e);
    for (std::size_t n = 0; n < occupied; ++n)
      for (std::size_t e = 0; e < virtuals; ++e)
        if (projected_single(projection_mask, virtuals, n, e))
          value += t1(n, e) * eri_value(m, n, i, occupied + e);
    for (std::size_t n = 0; n < occupied; ++n)
      for (std::size_t e = 0; e < virtuals; ++e)
        for (std::size_t q = 0; q < virtuals; ++q)
          if (projected_tau(projection_mask, occupied, virtuals, i, n, e,
                            q))
            value += 0.5 * work.tt(i, n, e, q) *
                     eri_value(m, n, occupied + e, occupied + q);
    work.fmi(m, i) = value;
  });
  parallel::sum_in_place(work.fmi.data());

  distributed_for(occupied * virtuals, [&](const std::size_t flat) {
    const auto m = flat / virtuals;
    const auto e = flat % virtuals;
    double value = fock(m, occupied + e);
    for (std::size_t n = 0; n < occupied; ++n)
      for (std::size_t q = 0; q < virtuals; ++q)
        if (projected_single(projection_mask, virtuals, n, q))
          value += t1(n, q) *
                   eri_value(m, n, occupied + e, occupied + q);
    work.fme(m, e) = value;
  });
  parallel::sum_in_place(work.fme.data());

  const auto occupied_fourth = work.wmnij.size();
  distributed_for(occupied_fourth, [&](const std::size_t flat) {
    auto index = flat;
    const auto j = index % occupied;
    index /= occupied;
    const auto i = index % occupied;
    index /= occupied;
    const auto n = index % occupied;
    const auto m = index / occupied;
    if (truncated_projection && !eri_declared(m, n, i, j))
      return;
    double value = eri_value(m, n, i, j);
    for (std::size_t e = 0; e < virtuals; ++e)
      value +=
          (projected_single(projection_mask, virtuals, j, e)
               ? t1(j, e) * eri_value(m, n, i, occupied + e)
               : 0.0) -
          (projected_single(projection_mask, virtuals, i, e)
               ? t1(i, e) * eri_value(m, n, j, occupied + e)
               : 0.0);
    for (std::size_t e = 0; e < virtuals; ++e)
      for (std::size_t q = 0; q < virtuals; ++q)
        if (projected_tau(projection_mask, occupied, virtuals, i, j, e, q))
          value += 0.25 * work.tau(i, j, e, q) *
                   eri_value(m, n, occupied + e, occupied + q);
    work.wmnij(m, n, i, j) = value;
  });
  parallel::sum_in_place(work.wmnij.data());

  if (use_tiled_intermediates) {
    const auto mixed_pair_count = checked_memory_product(occupied, occupied);
    const auto contracted = checked_memory_product(occupied, virtuals);
    const auto virtual_pairs = checked_memory_product(virtuals, virtuals);
    const int blas_virtuals =
        checked_blas_dimension(virtuals, "intermediate virtual tile");
    const int blas_occupied =
        checked_blas_dimension(occupied, "intermediate occupied tile");
    const int blas_contracted =
        checked_blas_dimension(contracted, "intermediate contracted tile");
    constexpr char no_transpose = 'N';
    constexpr char transpose = 'T';
    constexpr double one = 1.0;
    constexpr double zero = 0.0;
    const auto bit_words = (virtuals + 63U) / 64U;
    std::vector<double> projected_t1 = t1.data();
    std::vector<std::uint64_t> projected_single_bits(
        checked_memory_product(occupied, bit_words));
    for (std::size_t n = 0; n < occupied; ++n)
      for (std::size_t b = 0; b < virtuals; ++b)
        if (projected_single(projection_mask, virtuals, n, b)) {
          projected_single_bits[n * bit_words + b / 64U] |=
              std::uint64_t{1} << (b % 64U);
        } else {
          projected_t1[n * virtuals + b] = 0.0;
        }
    distributed_for(mixed_pair_count, [&](const std::size_t pair) {
      const auto m = pair / occupied;
      const auto j = pair % occupied;
      std::vector<double> values(virtual_pairs);
      std::vector<double> linear_eri(
          checked_memory_product(occupied, virtuals));
      std::vector<double> linear_contraction(virtual_pairs);
      std::vector<double> nonlinear_amplitudes(
          checked_memory_product(contracted, virtuals));
      std::vector<double> nonlinear_eri(
          checked_memory_product(contracted, virtuals));
      std::vector<double> nonlinear_contraction(virtual_pairs);
      std::vector<std::uint8_t> active_output(virtual_pairs);
      std::vector<std::uint64_t> active_output_bits(
          checked_memory_product(virtuals, bit_words));
      std::vector<std::uint64_t> nonlinear_amplitude_bits(
          checked_memory_product(contracted, bit_words));

      for (std::size_t b = 0; b < virtuals; ++b)
        for (std::size_t e = 0; e < virtuals; ++e) {
          const auto output = b * virtuals + e;
          if (truncated_projection &&
              !eri_declared(m, occupied + b, occupied + e, j))
            continue;
          active_output[output] = 1U;
          active_output_bits[e * bit_words + b / 64U] |=
              std::uint64_t{1} << (b % 64U);
          double value = eri_value(m, occupied + b, occupied + e, j);
          for (std::size_t q = 0; q < virtuals; ++q)
            if (projected_single(projection_mask, virtuals, j, q))
              value += t1(j, q) *
                       eri_value(m, occupied + b, occupied + e,
                                 occupied + q);
          values[output] = value;
        }
      for (std::size_t n = 0; n < occupied; ++n)
        for (std::size_t e = 0; e < virtuals; ++e) {
          bool required = !truncated_projection;
          for (std::size_t word = 0;
               truncated_projection && word < bit_words; ++word)
            required = required ||
                       (active_output_bits[e * bit_words + word] &
                        projected_single_bits[n * bit_words + word]) != 0U;
          if (required)
            linear_eri[n * virtuals + e] =
                eri_value(m, n, occupied + e, j);
        }
      // Row-major E(n,e) and T1(n,b) are viewed by column-major BLAS as
      // E^T(e,n) and T1^T(b,n). The output memory is row-major C(b,e).
      dgemm_(&no_transpose, &transpose, &blas_virtuals, &blas_virtuals,
             &blas_occupied, &one, linear_eri.data(), &blas_virtuals,
             projected_t1.data(), &blas_virtuals, &zero,
             linear_contraction.data(), &blas_virtuals);

      for (std::size_t n = 0; n < occupied; ++n)
        for (std::size_t q = 0; q < virtuals; ++q) {
          const auto nq = n * virtuals + q;
          for (std::size_t b = 0; b < virtuals; ++b) {
            const bool active_amplitude =
                projected_double(projection_mask, occupied, virtuals, j, n,
                                 q, b) ||
                (projected_single(projection_mask, virtuals, j, q) &&
                 projected_single(projection_mask, virtuals, n, b));
            if (!active_amplitude) continue;
            nonlinear_amplitudes[nq * virtuals + b] =
                0.5 * (direct_t2 ? (*direct_t2)(j, n, q, b)
                                 : t2_from_work(work, t1, j, n, q, b)) +
                t1(j, q) * t1(n, b);
            nonlinear_amplitude_bits[nq * bit_words + b / 64U] |=
                std::uint64_t{1} << (b % 64U);
          }
          for (std::size_t e = 0; e < virtuals; ++e) {
            bool required = !truncated_projection;
            for (std::size_t word = 0;
                 truncated_projection && word < bit_words; ++word)
              required = required ||
                         (active_output_bits[e * bit_words + word] &
                          nonlinear_amplitude_bits[nq * bit_words + word]) !=
                             0U;
            if (required)
              nonlinear_eri[nq * virtuals + e] =
                  eri_value(m, n, occupied + e, occupied + q);
          }
        }
      dgemm_(&no_transpose, &transpose, &blas_virtuals, &blas_virtuals,
             &blas_contracted, &one, nonlinear_eri.data(), &blas_virtuals,
             nonlinear_amplitudes.data(), &blas_virtuals, &zero,
             nonlinear_contraction.data(), &blas_virtuals);
      for (std::size_t b = 0; b < virtuals; ++b)
        for (std::size_t e = 0; e < virtuals; ++e)
          if (active_output[b * virtuals + e] != 0U)
            work.wmbej(m, b, e, j) =
                values[b * virtuals + e] -
                linear_contraction[b * virtuals + e] -
                nonlinear_contraction[b * virtuals + e];
    });
    if (timing) {
      const auto communicator = parallel::world();
      const auto owned_tasks = communicator.rank < mixed_pair_count
          ? 1 + (mixed_pair_count - 1 - communicator.rank) /
                    communicator.size
          : 0;
      timing->intermediate_mixed_pair_tasks += owned_tasks;
      timing->intermediate_blas_contractions +=
          checked_memory_product(2, owned_tasks);
    }
  } else {
    const auto mixed_count = work.wmbej.size();
    distributed_for(mixed_count, [&](const std::size_t flat) {
      auto index = flat;
      const auto j = index % occupied;
      index /= occupied;
      const auto e = index % virtuals;
      index /= virtuals;
      const auto b = index % virtuals;
      const auto m = index / virtuals;
      if (truncated_projection &&
          !eri_declared(m, occupied + b, occupied + e, j))
        return;
      double value = eri_value(m, occupied + b, occupied + e, j);
      for (std::size_t q = 0; q < virtuals; ++q)
        if (projected_single(projection_mask, virtuals, j, q))
          value += t1(j, q) *
                   eri_value(m, occupied + b, occupied + e, occupied + q);
      for (std::size_t n = 0; n < occupied; ++n)
        if (projected_single(projection_mask, virtuals, n, b))
          value -= t1(n, b) * eri_value(m, n, occupied + e, j);
      for (std::size_t n = 0; n < occupied; ++n)
        for (std::size_t q = 0; q < virtuals; ++q)
          if (projected_double(projection_mask, occupied, virtuals, j, n, q,
                               b) ||
              (projected_single(projection_mask, virtuals, j, q) &&
               projected_single(projection_mask, virtuals, n, b)))
            value -=
                (0.5 * (direct_t2 ? (*direct_t2)(j, n, q, b)
                                  : t2_from_work(work, t1, j, n, q, b)) +
                 t1(j, q) * t1(n, b)) *
                eri_value(m, n, occupied + e, occupied + q);
      work.wmbej(m, b, e, j) = value;
    });
  }
  parallel::sum_in_place(work.wmbej.data());
  return work;
}

template <typename Eri>
Work intermediates(const Matrix& fock, const Eri& eri, const Matrix& t1,
                   const Tensor4& t2, const std::size_t occupied,
                   const bool use_tiled_intermediates = false,
                   CcsdIterationTiming* const timing = nullptr,
                   const SelectedIntegralPanelCache* const
                       selected_panel_cache = nullptr,
                   const CcsdProjectionMask* const
                       projection_mask = nullptr) {
  const auto virtuals = t1.dimensions()[1];
  auto work = make_work(occupied, virtuals);
  const auto doubles_count = work.tau.size();
  if (doubles_count >
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::overflow_error("CCSD doubles extent exceeds ptrdiff_t");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t flat = 0;
       flat < static_cast<std::ptrdiff_t>(doubles_count); ++flat) {
    auto index = static_cast<std::size_t>(flat);
    const auto b = index % virtuals;
    index /= virtuals;
    const auto a = index % virtuals;
    index /= virtuals;
    const auto j = index % occupied;
    const auto i = index / occupied;
    const auto cross = t1(i, a) * t1(j, b) - t1(i, b) * t1(j, a);
    work.tau(i, j, a, b) = t2(i, j, a, b) + cross;
    work.tt(i, j, a, b) = t2(i, j, a, b) + 0.5 * cross;
  }
  return finish_intermediates(fock, eri, t1, std::move(work), &t2,
                              occupied, use_tiled_intermediates, timing,
                              selected_panel_cache, projection_mask);
}

template <typename Eri>
DistributedWork distributed_intermediates(
    const Matrix& fock, const Eri& eri, const Matrix& t1,
    const GlobalArraysPairTensor& t2, const std::size_t occupied,
    GlobalArraysPairTensor& tau, GlobalArraysPairTensor& tilde_tau,
    GlobalArraysPairTensor& wmbej, GlobalArraysPairTensor& wmnij) {
  const auto virtuals = t1.dimensions()[1];
  const auto communicator = parallel::world();
  const auto pair_count = occupied * occupied;
  const auto block_size = virtuals * virtuals;
  DistributedWork work{Matrix({virtuals, virtuals}),
                       Matrix({occupied, occupied}),
                       Matrix({occupied, virtuals})};

  tau.zero();
  tilde_tau.zero();
  for (std::size_t pair = communicator.rank; pair < pair_count;
       pair += communicator.size) {
    const auto i = pair / occupied;
    const auto j = pair % occupied;
    const auto current = t2.get_pair(i, j);
    std::vector<double> tau_block(block_size);
    std::vector<double> tilde_block(block_size);
    if (block_size >
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
      throw std::overflow_error("distributed tau block exceeds ptrdiff_t");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t flat = 0;
         flat < static_cast<std::ptrdiff_t>(block_size); ++flat) {
      const auto a = static_cast<std::size_t>(flat) / virtuals;
      const auto b = static_cast<std::size_t>(flat) % virtuals;
      const auto cross =
          t1(i, a) * t1(j, b) - t1(i, b) * t1(j, a);
      tau_block[static_cast<std::size_t>(flat)] =
          current[static_cast<std::size_t>(flat)] + cross;
      tilde_block[static_cast<std::size_t>(flat)] =
          current[static_cast<std::size_t>(flat)] + 0.5 * cross;
    }
    tau.put_pair(i, j, tau_block);
    tilde_tau.put_pair(i, j, tilde_block);
  }
  tau.synchronize();
  tilde_tau.synchronize();

  // Base terms are divided by output element. T2-sized source terms are
  // divided by occupied pair, so every GA block is fetched once per phase.
  for (std::size_t flat = communicator.rank; flat < virtuals * virtuals;
       flat += communicator.size) {
    const auto a = flat / virtuals;
    const auto e = flat % virtuals;
    double value = a == e ? 0.0 : fock(occupied + a, occupied + e);
    for (std::size_t m = 0; m < occupied; ++m)
      value -= 0.5 * fock(m, occupied + e) * t1(m, a);
    for (std::size_t m = 0; m < occupied; ++m)
      for (std::size_t q = 0; q < virtuals; ++q)
        value += t1(m, q) *
                 eri(m, occupied + a, occupied + q, occupied + e);
    work.fae(a, e) = value;
  }
  for (std::size_t pair = communicator.rank; pair < pair_count;
       pair += communicator.size) {
    const auto m = pair / occupied;
    const auto n = pair % occupied;
    const auto block = tilde_tau.get_pair(m, n);
    for (std::size_t a = 0; a < virtuals; ++a)
      for (std::size_t e = 0; e < virtuals; ++e)
        for (std::size_t q = 0; q < virtuals; ++q)
          work.fae(a, e) -=
              0.5 * block[a * virtuals + q] *
              eri(m, n, occupied + e, occupied + q);
  }
  parallel::sum_in_place(work.fae.data());

  for (std::size_t flat = communicator.rank; flat < occupied * occupied;
       flat += communicator.size) {
    const auto m = flat / occupied;
    const auto i = flat % occupied;
    double value = m == i ? 0.0 : fock(m, i);
    for (std::size_t e = 0; e < virtuals; ++e)
      value += 0.5 * t1(i, e) * fock(m, occupied + e);
    for (std::size_t n = 0; n < occupied; ++n)
      for (std::size_t e = 0; e < virtuals; ++e)
        value += t1(n, e) * eri(m, n, i, occupied + e);
    work.fmi(m, i) = value;
  }
  for (std::size_t pair = communicator.rank; pair < pair_count;
       pair += communicator.size) {
    const auto i = pair / occupied;
    const auto n = pair % occupied;
    const auto block = tilde_tau.get_pair(i, n);
    for (std::size_t m = 0; m < occupied; ++m)
      for (std::size_t e = 0; e < virtuals; ++e)
        for (std::size_t q = 0; q < virtuals; ++q)
          work.fmi(m, i) +=
              0.5 * block[e * virtuals + q] *
              eri(m, n, occupied + e, occupied + q);
  }
  parallel::sum_in_place(work.fmi.data());

  for (std::size_t flat = communicator.rank; flat < occupied * virtuals;
       flat += communicator.size) {
    const auto m = flat / virtuals;
    const auto e = flat % virtuals;
    double value = fock(m, occupied + e);
    for (std::size_t n = 0; n < occupied; ++n)
      for (std::size_t q = 0; q < virtuals; ++q)
        value += t1(n, q) * eri(m, n, occupied + e, occupied + q);
    work.fme(m, e) = value;
  }
  parallel::sum_in_place(work.fme.data());

  wmnij.zero();
  for (std::size_t pair = communicator.rank; pair < pair_count;
       pair += communicator.size) {
    const auto i = pair / occupied;
    const auto j = pair % occupied;
    const auto tau_block = tau.get_pair(i, j);
    std::vector<double> block(pair_count);
    if (pair_count >
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
      throw std::overflow_error("distributed W_mnij block exceeds ptrdiff_t");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t flat = 0;
         flat < static_cast<std::ptrdiff_t>(pair_count); ++flat) {
      const auto m = static_cast<std::size_t>(flat) / occupied;
      const auto n = static_cast<std::size_t>(flat) % occupied;
      double value = eri(m, n, i, j);
      for (std::size_t e = 0; e < virtuals; ++e)
        value += t1(j, e) * eri(m, n, i, occupied + e) -
                 t1(i, e) * eri(m, n, j, occupied + e);
      for (std::size_t e = 0; e < virtuals; ++e)
        for (std::size_t q = 0; q < virtuals; ++q)
          value +=
              0.25 * tau_block[e * virtuals + q] *
              eri(m, n, occupied + e, occupied + q);
      block[static_cast<std::size_t>(flat)] = value;
    }
    wmnij.put_pair(i, j, block);
  }
  wmnij.synchronize();

  wmbej.zero();
  for (std::size_t j = 0; j < occupied; ++j) {
    bool owns_output = false;
    for (std::size_t m = 0; m < occupied; ++m)
      owns_output = owns_output ||
                    ((m * occupied + j) % communicator.size ==
                     communicator.rank);
    if (!owns_output)
      continue;
    std::vector<std::vector<double>> t2_jn;
    t2_jn.reserve(occupied);
    for (std::size_t n = 0; n < occupied; ++n)
      t2_jn.push_back(t2.get_pair(j, n));
    for (std::size_t m = 0; m < occupied; ++m) {
      if ((m * occupied + j) % communicator.size != communicator.rank)
        continue;
      std::vector<double> block(block_size);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (std::ptrdiff_t flat = 0;
           flat < static_cast<std::ptrdiff_t>(block_size); ++flat) {
        const auto b = static_cast<std::size_t>(flat) / virtuals;
        const auto e = static_cast<std::size_t>(flat) % virtuals;
        double value = eri(m, occupied + b, occupied + e, j);
        for (std::size_t q = 0; q < virtuals; ++q)
          value +=
              t1(j, q) *
              eri(m, occupied + b, occupied + e, occupied + q);
        for (std::size_t n = 0; n < occupied; ++n)
          value -= t1(n, b) * eri(m, n, occupied + e, j);
        for (std::size_t n = 0; n < occupied; ++n)
          for (std::size_t q = 0; q < virtuals; ++q)
            value -=
                (0.5 * t2_jn[n][q * virtuals + b] +
                 t1(j, q) * t1(n, b)) *
                eri(m, n, occupied + e, occupied + q);
        block[static_cast<std::size_t>(flat)] = value;
      }
      wmbej.put_pair(m, j, block);
    }
  }
  wmbej.synchronize();
  return work;
}

struct TiledWabefEvaluation {
  CcsdWabefResult result;
  double build_seconds{};
  double contraction_seconds{};
  std::size_t virtual_pair_panels{};
  std::size_t blas_contractions{};
  bool factorized_cached{};
};

struct TiledWabefInvariantCache {
  std::size_t occupied{};
  std::size_t virtuals{};
  bool pair_symmetry{};
  std::shared_ptr<const PmSymmetry> pm_symmetry;
  std::vector<std::size_t> tasks;
  std::vector<std::size_t> task_to_local;
  // Row-major logical views. Each final internal virtual-pair index is
  // contiguous so a panel can also be consumed as a column-major BLAS matrix
  // with virtual_pairs as its leading dimension.
  std::vector<double> bare_vvvv;
  std::vector<double> ovvv;
  std::vector<double> oovv;
  std::size_t allocated_bytes{};
};

void validate_wabef_pair_symmetry_options(const CcsdOptions& options) {
  if (options.pm_symmetry &&
      (options.use_wabef_pair_symmetry || options.projection_mask ||
       options.wabef_backend != CcsdWabefBackend::tiled_cpu ||
       options.wabef_packing_backend != CcsdWabefPackingBackend::cached_invariants ||
       options.amplitude_backend != CcsdAmplitudeBackend::replicated ||
       (options.use_direct_spatial_doubles && !options.use_tiled_spatial_pairs)))
    throw std::invalid_argument("PM symmetry requires full replicated cached tiled-CPU spatial CCSD, without a projection or pair-symmetry flag");
  if (options.use_wabef_pair_symmetry &&
      (options.wabef_backend != CcsdWabefBackend::tiled_cpu ||
       options.wabef_packing_backend !=
           CcsdWabefPackingBackend::cached_invariants ||
       options.projection_mask || options.use_direct_spatial_doubles ||
       options.amplitude_backend == CcsdAmplitudeBackend::global_arrays))
    throw std::invalid_argument(
        "W_abef pair symmetry requires unprojected cached tiled-CPU "
        "replicated contractions");
}

std::vector<std::size_t> owned_wabef_tasks(
    const std::size_t occupied, const std::size_t virtuals,
    const CcsdProjectionMask* const projection_mask) {
  const auto communicator = parallel::world();
  const auto virtual_pairs = checked_memory_product(virtuals, virtuals);
  std::vector<std::size_t> tasks;
  for (std::size_t task = communicator.rank; task < virtual_pairs;
       task += communicator.size) {
    if (projection_mask) {
      const auto a = task / virtuals;
      const auto b = task % virtuals;
      bool required = false;
      for (std::size_t i = 0; i < occupied && !required; ++i)
        for (std::size_t j = 0; j < occupied; ++j) {
          const auto flat = ((i * occupied + j) * virtuals + a) * virtuals + b;
          if (projection_mask->doubles[flat] != 0U) {
            required = true;
            break;
          }
        }
      if (!required) continue;
    }
    tasks.push_back(task);
  }
  return tasks;
}

template <typename Eri>
TiledWabefInvariantCache build_tiled_wabef_invariant_cache(
    const Eri& eri, const std::size_t occupied, const std::size_t virtuals,
    const CcsdProjectionMask* const projection_mask,
    const std::size_t maximum_cache_bytes,
    const bool pair_symmetry = false,
    std::shared_ptr<const PmSymmetry> pm_symmetry = {}) {
  if (maximum_cache_bytes == 0)
    throw std::invalid_argument(
        "CCSD W_abef invariant-cache memory must be nonzero");
  TiledWabefInvariantCache cache;
  cache.occupied = occupied;
  cache.virtuals = virtuals;
  const auto virtual_pairs = checked_memory_product(virtuals, virtuals);
  cache.pair_symmetry = pair_symmetry;
  cache.pm_symmetry = std::move(pm_symmetry);
  if (cache.pm_symmetry) {
    const auto& symmetry = *cache.pm_symmetry;
    if (occupied != 2*symmetry.occupied || virtuals != 2*symmetry.virtuals || projection_mask || pair_symmetry)
      throw std::invalid_argument("PM symmetry W_abef dimensions/options mismatch");
    const auto world = parallel::world();
    std::size_t ordinal = 0;
    for (std::size_t task=0; task<virtual_pairs; ++task) {
      auto representative=task;
      for (const auto& g:symmetry.operations)
        representative=std::min(representative,
            pm_spin_image(g,task/virtuals,symmetry.occupied)*virtuals+
            pm_spin_image(g,task%virtuals,symmetry.occupied));
      if (representative != task) continue;
      if (ordinal++ % world.size == world.rank) cache.tasks.push_back(task);
    }
  } else if (pair_symmetry) {
    if (projection_mask)
      throw std::invalid_argument("W_abef pair symmetry rejects projection masks");
    // Balance the triangular task ordinal, not a*v+b: rectangular ownership
    // would cluster the surviving tasks unevenly across MPI ranks.
    const auto communicator = parallel::world();
    std::size_t ordinal = 0;
    for (std::size_t a = 0; a < virtuals; ++a)
      for (std::size_t b = a + 1; b < virtuals; ++b, ++ordinal)
        if (ordinal % communicator.size == communicator.rank)
          cache.tasks.push_back(a * virtuals + b);
  } else {
    cache.tasks = owned_wabef_tasks(occupied, virtuals, projection_mask);
  }
  const auto occupied_pairs = checked_memory_product(occupied, occupied);
  const auto bare_elements =
      checked_memory_product(cache.tasks.size(), virtual_pairs);
  const auto ovvv_elements = checked_memory_product(
      checked_memory_product(virtuals, occupied), virtual_pairs);
  const auto oovv_elements =
      checked_memory_product(occupied_pairs, virtual_pairs);
  const auto double_elements = checked_memory_sum(
      checked_memory_sum(bare_elements, ovvv_elements), oovv_elements);
  const auto data_bytes =
      checked_memory_product(double_elements, sizeof(double));
  const auto metadata_elements =
      checked_memory_sum(cache.tasks.size(), virtual_pairs);
  cache.allocated_bytes = checked_memory_sum(
      data_bytes,
      checked_memory_product(metadata_elements, sizeof(std::size_t)));
  if (cache.allocated_bytes > maximum_cache_bytes)
    throw std::runtime_error(
        "exact tiled CCSD W_abef invariant cache requires " +
        std::to_string(cache.allocated_bytes) +
        " bytes per rank, exceeding CCSD_WABEF_INVARIANT_CACHE_MIB");

  cache.task_to_local.assign(virtual_pairs,
                             std::numeric_limits<std::size_t>::max());
  for (std::size_t local = 0; local < cache.tasks.size(); ++local)
    cache.task_to_local[cache.tasks[local]] = local;
  cache.bare_vvvv.resize(bare_elements);
  cache.ovvv.resize(ovvv_elements);
  cache.oovv.resize(oovv_elements);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t local_index = 0;
       local_index < static_cast<std::ptrdiff_t>(cache.tasks.size());
       ++local_index) {
    const auto local = static_cast<std::size_t>(local_index);
    const auto a = cache.tasks[local] / virtuals;
    const auto b = cache.tasks[local] % virtuals;
    auto* const output = cache.bare_vvvv.data() +
                         static_cast<std::ptrdiff_t>(local * virtual_pairs);
    for (std::size_t ef = 0; ef < virtual_pairs; ++ef)
      output[ef] = eri(occupied + a, occupied + b,
                       occupied + ef / virtuals,
                       occupied + ef % virtuals);
  }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t row_index = 0;
       row_index < static_cast<std::ptrdiff_t>(virtuals * occupied);
       ++row_index) {
    const auto row = static_cast<std::size_t>(row_index);
    const auto a = row / occupied;
    const auto m = row % occupied;
    auto* const output = cache.ovvv.data() +
                         static_cast<std::ptrdiff_t>(row * virtual_pairs);
    for (std::size_t ef = 0; ef < virtual_pairs; ++ef)
      output[ef] = eri(occupied + a, m, occupied + ef / virtuals,
                       occupied + ef % virtuals);
  }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t pair_index = 0;
       pair_index < static_cast<std::ptrdiff_t>(occupied_pairs);
       ++pair_index) {
    const auto pair = static_cast<std::size_t>(pair_index);
    const auto m = pair / occupied;
    const auto n = pair % occupied;
    auto* const output = cache.oovv.data() +
                         static_cast<std::ptrdiff_t>(pair * virtual_pairs);
    for (std::size_t ef = 0; ef < virtual_pairs; ++ef)
      output[ef] = eri(m, n, occupied + ef / virtuals,
                       occupied + ef % virtuals);
  }
  return cache;
}

std::size_t factorized_cached_wabef_workspace_bytes(
    const std::size_t occupied, const std::size_t virtuals,
    const std::size_t task_count) {
  const auto occupied_pairs = checked_memory_product(occupied, occupied);
  const auto occupied_virtual = checked_memory_product(occupied, virtuals);
  const auto elements = checked_memory_sum(
      checked_memory_product(task_count, occupied_pairs),
      checked_memory_sum(
          checked_memory_product(occupied_pairs, occupied_virtual),
          checked_memory_product(occupied_pairs, occupied_pairs)));
  return checked_memory_product(elements, sizeof(double));
}

struct TiledDoublesEvaluation {
  std::size_t occupied_pair_tasks{};
  std::size_t blas_contractions{};
};

template <typename Eri>
TiledDoublesEvaluation evaluate_tiled_doubles_without_wabef(
    const Eri& eri, const Matrix& t1, const Tensor4& t2,
    const Work& work, const std::size_t occupied, const Tensor4& d2,
    Tensor4& next_t2,
    const CcsdProjectionMask* const projection_mask = nullptr,
    const SelectedIntegralPanelCache* const selected_panel_cache = nullptr) {
  const auto virtuals = t1.dimensions()[1];
  const auto eri_value = [&](const std::size_t p, const std::size_t q,
                             const std::size_t r, const std::size_t s) {
    return selected_panel_eri(eri, selected_panel_cache, p, q, r, s);
  };
  const auto virtual_pairs = checked_memory_product(virtuals, virtuals);
  const auto occupied_pairs = checked_memory_product(occupied, occupied);
  if (t2.dimensions() !=
          std::array<std::size_t, 4>{occupied, occupied, virtuals,
                                     virtuals} ||
      work.tau.dimensions() != t2.dimensions() ||
      work.wmnij.dimensions() !=
          std::array<std::size_t, 4>{occupied, occupied, occupied,
                                     occupied} ||
      work.wmbej.dimensions() !=
          std::array<std::size_t, 4>{occupied, virtuals, virtuals,
                                     occupied} ||
      d2.dimensions() != t2.dimensions() ||
      next_t2.dimensions() != t2.dimensions())
    throw std::invalid_argument(
        "tiled CCSD doubles input dimensions are inconsistent");

  Matrix dressed_fae({virtuals, virtuals});
  for (std::size_t b = 0; b < virtuals; ++b)
    for (std::size_t e = 0; e < virtuals; ++e) {
      dressed_fae(b, e) = work.fae(b, e);
      for (std::size_t m = 0; m < occupied; ++m)
        dressed_fae(b, e) -= 0.5 * t1(m, b) * work.fme(m, e);
    }
  Matrix dressed_fmi({occupied, occupied});
  for (std::size_t m = 0; m < occupied; ++m)
    for (std::size_t j = 0; j < occupied; ++j) {
      dressed_fmi(m, j) = work.fmi(m, j);
      for (std::size_t e = 0; e < virtuals; ++e)
        dressed_fmi(m, j) += 0.5 * t1(j, e) * work.fme(m, e);
    }

  const auto communicator = parallel::world();
  std::vector<std::size_t> tasks;
  if (communicator.rank < occupied_pairs)
    for (std::size_t pair = communicator.rank; pair < occupied_pairs;
         pair += communicator.size)
      tasks.push_back(pair);
  if (tasks.size() >
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::overflow_error(
        "tiled CCSD doubles task count exceeds ptrdiff_t");

  const int blas_virtuals =
      checked_blas_dimension(virtuals, "doubles virtual tile");
  const int blas_virtual_pairs =
      checked_blas_dimension(virtual_pairs, "doubles virtual-pair tile");
  const int blas_occupied =
      checked_blas_dimension(occupied, "doubles occupied tile");
  const int blas_occupied_pairs =
      checked_blas_dimension(occupied_pairs, "doubles occupied-pair tile");
  constexpr char no_transpose = 'N';
  constexpr char transpose = 'T';
  constexpr double zero = 0.0;
  constexpr double one = 1.0;
  const int one_column = 1;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t task_index = 0;
       task_index < static_cast<std::ptrdiff_t>(tasks.size());
       ++task_index) {
    const auto pair = tasks[static_cast<std::size_t>(task_index)];
    const auto i = pair / occupied;
    const auto j = pair % occupied;
    const auto* const direct_t2 =
        t2.data().data() +
        static_cast<std::ptrdiff_t>(pair * virtual_pairs);

    std::vector<double> block(virtual_pairs);
    std::vector<double> fae(virtual_pairs);
    std::vector<double> fmi_direct(virtual_pairs);
    std::vector<double> fmi_exchange(virtual_pairs);
    std::vector<double> wmnij(virtual_pairs);
    std::vector<double> wm_direct(virtual_pairs);
    std::vector<double> wm_exchange(virtual_pairs);
    std::vector<double> wm_direct_correction(virtual_pairs);
    std::vector<double> wm_exchange_correction(virtual_pairs);
    std::vector<double> mbij(virtual_pairs);
    std::vector<double> coefficients(std::max(occupied, occupied_pairs));
    std::vector<double> wm_matrix(std::max(
        virtual_pairs, checked_memory_product(occupied, virtuals)));
    std::vector<double> contracted_ov(occupied * virtuals);
    std::vector<std::uint8_t> active_second_virtual(virtuals, 0U);
    for (std::size_t b = 0; b < virtuals; ++b)
      for (std::size_t a = 0; a < virtuals; ++a) {
        const auto flat = pair * virtual_pairs + a * virtuals + b;
        if (!projection_mask || projection_mask->doubles[flat] != 0U) {
          active_second_virtual[b] = 1U;
          break;
        }
      }

    // C_ab = sum_e T_ij^ae F_be. Row-major operands are presented to
    // column-major BLAS in transposed logical order.
    dgemm_(&transpose, &no_transpose, &blas_virtuals, &blas_virtuals,
           &blas_virtuals, &one, dressed_fae.data().data(), &blas_virtuals,
           direct_t2, &blas_virtuals, &zero, fae.data(), &blas_virtuals);

    for (std::size_t m = 0; m < occupied; ++m)
      coefficients[m] = dressed_fmi(m, j);
    const auto* const t2_i =
        t2.data().data() +
        static_cast<std::ptrdiff_t>(i * occupied * virtual_pairs);
    dgemm_(&no_transpose, &no_transpose, &blas_virtual_pairs, &one_column,
           &blas_occupied, &one, t2_i, &blas_virtual_pairs,
           coefficients.data(), &blas_occupied, &zero, fmi_direct.data(),
           &blas_virtual_pairs);
    for (std::size_t m = 0; m < occupied; ++m)
      coefficients[m] = dressed_fmi(m, i);
    const auto* const t2_j =
        t2.data().data() +
        static_cast<std::ptrdiff_t>(j * occupied * virtual_pairs);
    dgemm_(&no_transpose, &no_transpose, &blas_virtual_pairs, &one_column,
           &blas_occupied, &one, t2_j, &blas_virtual_pairs,
           coefficients.data(), &blas_occupied, &zero, fmi_exchange.data(),
           &blas_virtual_pairs);

    for (std::size_t m = 0; m < occupied; ++m)
      for (std::size_t n = 0; n < occupied; ++n)
        coefficients[m * occupied + n] = work.wmnij(m, n, i, j);
    dgemm_(&no_transpose, &no_transpose, &blas_virtual_pairs, &one_column,
           &blas_occupied_pairs, &one, work.tau.data().data(),
           &blas_virtual_pairs, coefficients.data(), &blas_occupied_pairs,
           &zero, wmnij.data(), &blas_virtual_pairs);

    // C_ab = sum_me T_im^ae W_mbej. Packing one W_mj virtual matrix at a
    // time keeps the workspace bounded while turning the dominant e
    // contraction into level-3 BLAS.
    for (std::size_t m = 0; m < occupied; ++m) {
      for (std::size_t b = 0; b < virtuals; ++b)
        for (std::size_t e = 0; e < virtuals; ++e)
          wm_matrix[b * virtuals + e] = work.wmbej(m, b, e, j);
      const auto* const t2_im =
          t2.data().data() +
          static_cast<std::ptrdiff_t>((i * occupied + m) * virtual_pairs);
      dgemm_(&transpose, &no_transpose, &blas_virtuals, &blas_virtuals,
             &blas_virtuals, &one, wm_matrix.data(), &blas_virtuals, t2_im,
             &blas_virtuals, &one, wm_direct.data(), &blas_virtuals);

      for (std::size_t b = 0; b < virtuals; ++b)
        for (std::size_t e = 0; e < virtuals; ++e)
          wm_matrix[b * virtuals + e] = work.wmbej(m, b, e, i);
      const auto* const t2_jm =
          t2.data().data() +
          static_cast<std::ptrdiff_t>((j * occupied + m) * virtual_pairs);
      dgemm_(&transpose, &no_transpose, &blas_virtuals, &blas_virtuals,
             &blas_virtuals, &one, wm_matrix.data(), &blas_virtuals, t2_jm,
             &blas_virtuals, &one, wm_exchange.data(), &blas_virtuals);
    }

    // Factor the T1*T1*ERI part of W_mbej as R_mb = sum_e t_i^e (mb|ej),
    // followed by C_ab = sum_m t_m^a R_mb.
    std::fill(contracted_ov.begin(), contracted_ov.end(), 0.0);
    std::fill(wm_matrix.begin(), wm_matrix.end(), 0.0);
    for (std::size_t m = 0; m < occupied; ++m)
      for (std::size_t b = 0; b < virtuals; ++b) {
        if (active_second_virtual[b] == 0U) continue;
        double direct_value = 0.0;
        double exchange_value = 0.0;
        for (std::size_t e = 0; e < virtuals; ++e) {
          direct_value +=
              t1(i, e) * eri_value(m, occupied + b, occupied + e, j);
          exchange_value +=
              t1(j, e) * eri_value(m, occupied + b, occupied + e, i);
        }
        contracted_ov[m * virtuals + b] = direct_value;
        wm_matrix[m * virtuals + b] = exchange_value;
      }
    dgemm_(&no_transpose, &transpose, &blas_virtuals, &blas_virtuals,
           &blas_occupied, &one, contracted_ov.data(), &blas_virtuals,
           t1.data().data(), &blas_virtuals, &zero,
           wm_direct_correction.data(), &blas_virtuals);
    dgemm_(&no_transpose, &transpose, &blas_virtuals, &blas_virtuals,
           &blas_occupied, &one, wm_matrix.data(), &blas_virtuals,
           t1.data().data(), &blas_virtuals, &zero,
           wm_exchange_correction.data(), &blas_virtuals);

    // M_bij^a = sum_m t_m^a (mb|ij).
    std::fill(contracted_ov.begin(), contracted_ov.end(), 0.0);
    for (std::size_t m = 0; m < occupied; ++m)
      for (std::size_t b = 0; b < virtuals; ++b)
        if (active_second_virtual[b] != 0U)
          contracted_ov[m * virtuals + b] =
              eri_value(m, occupied + b, i, j);
    dgemm_(&no_transpose, &transpose, &blas_virtuals, &blas_virtuals,
           &blas_occupied, &one, contracted_ov.data(), &blas_virtuals,
           t1.data().data(), &blas_virtuals, &zero, mbij.data(),
           &blas_virtuals);

    for (std::size_t a = 0; a < virtuals; ++a)
      for (std::size_t b = 0; b < virtuals; ++b) {
        const auto ab = a * virtuals + b;
        const auto ba = b * virtuals + a;
        const auto flat = pair * virtual_pairs + ab;
        if (projection_mask && projection_mask->doubles[flat] == 0U)
          continue;
        double value =
            eri_value(i, j, occupied + a, occupied + b) + fae[ab] -
            fae[ba] - fmi_direct[ab] + fmi_exchange[ab] +
            0.5 * wmnij[ab];
        const auto direct = wm_direct[ab] - wm_direct_correction[ab];
        const auto exchange = wm_exchange[ab] - wm_exchange_correction[ab];
        const auto direct_transposed =
            wm_direct[ba] - wm_direct_correction[ba];
        const auto exchange_transposed =
            wm_exchange[ba] - wm_exchange_correction[ba];
        value += direct - exchange - direct_transposed + exchange_transposed;
        for (std::size_t e = 0; e < virtuals; ++e)
          value +=
              t1(i, e) * eri_value(occupied + a, occupied + b,
                                   occupied + e, j) -
              t1(j, e) * eri_value(occupied + a, occupied + b,
                                   occupied + e, i);
        value += -mbij[ab] + mbij[ba];
        block[ab] = value / d2(i, j, a, b);
      }
    std::copy(block.begin(), block.end(),
              next_t2.data().begin() +
                  static_cast<std::ptrdiff_t>(pair * virtual_pairs));
  }

  TiledDoublesEvaluation evaluation;
  evaluation.occupied_pair_tasks = tasks.size();
  const auto contractions_per_task =
      checked_memory_product(2, occupied);
  if (contractions_per_task >
      std::numeric_limits<std::size_t>::max() - 7)
    throw std::overflow_error(
        "tiled CCSD doubles contraction count overflows");
  evaluation.blas_contractions = checked_memory_product(
      tasks.size(), contractions_per_task + 7);
  return evaluation;
}

template <typename Eri>
TiledWabefEvaluation evaluate_tiled_wabef(
    const Eri& eri, const Matrix& t1, const std::size_t occupied,
    const Tensor4& tau,
    const std::vector<std::size_t>& tasks,
    const std::vector<double>& tau_ab_scalars,
    const std::size_t maximum_panel_bytes,
    const TiledWabefInvariantCache* const invariant_cache = nullptr) {
  const auto virtuals = t1.dimensions()[1];
  const auto occupied_pairs = checked_memory_product(occupied, occupied);
  const auto virtual_pairs = checked_memory_product(virtuals, virtuals);
  const auto task_count = tasks.size();
  if (task_count == 0 ||
      tau_ab_scalars.size() !=
          checked_memory_product(task_count, occupied_pairs) ||
      tau.dimensions() !=
          std::array<std::size_t, 4>{occupied, occupied, virtuals, virtuals} ||
      maximum_panel_bytes == 0)
    throw std::invalid_argument(
        "tiled CCSD W_abef input dimensions are inconsistent");
  if (invariant_cache &&
      (invariant_cache->occupied != occupied ||
       invariant_cache->virtuals != virtuals ||
       invariant_cache->task_to_local.size() != virtual_pairs))
    throw std::invalid_argument(
        "tiled CCSD W_abef invariant-cache dimensions are inconsistent");
  if (invariant_cache)
    for (const auto task : tasks)
      if (task >= invariant_cache->task_to_local.size() ||
          invariant_cache->task_to_local[task] ==
              std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument(
            "tiled CCSD W_abef task is absent from invariant cache");

  // Reassociate the cached exact W_abef contractions so the iteration never
  // materializes the dressed W_abef(a,b,e,f) panel. For the complete owned
  // task set this evaluates
  //   tau * bare_vvvv,
  //   (tau * ovvv^T) * t1, and
  //   (tau * oovv^T) * tau
  // separately. The algebra and accumulation domain are unchanged, while the
  // expensive OOVV term is reduced from an external-pair-by-internal-pair
  // panel contraction to two occupied-pair contractions.
  const auto factorized_workspace_bytes =
      factorized_cached_wabef_workspace_bytes(
          occupied, virtuals, task_count);
  if (invariant_cache && tasks == invariant_cache->tasks &&
      factorized_workspace_bytes <= maximum_panel_bytes) {
    TiledWabefEvaluation evaluation;
    evaluation.virtual_pair_panels = 1;
    evaluation.blas_contractions = 4;
    evaluation.factorized_cached = true;
    evaluation.result.workspace_bytes = factorized_workspace_bytes;
    evaluation.result.contractions.assign(
        checked_memory_product(task_count, occupied_pairs), 0.0);
    const auto occupied_virtual =
        checked_memory_product(occupied, virtuals);
    std::vector<double> tau_ovvv(
        checked_memory_product(occupied_pairs, occupied_virtual));
    std::vector<double> tau_oovv(
        checked_memory_product(occupied_pairs, occupied_pairs));
    constexpr char no_transpose = 'N';
    constexpr char transpose = 'T';
    constexpr double zero = 0.0;
    constexpr double one = 1.0;
    constexpr double quarter = 0.25;
    const int blas_occupied_pairs =
        checked_blas_dimension(occupied_pairs, "factorized occupied pairs");
    const int blas_occupied_virtual = checked_blas_dimension(
        occupied_virtual, "factorized occupied-virtual rows");
    const int blas_virtual_pairs =
        checked_blas_dimension(virtual_pairs, "factorized virtual pairs");
#if defined(_OPENMP)
    const auto task_workers = std::min(
        std::max(task_count, occupied_pairs),
        static_cast<std::size_t>(omp_get_max_threads()));
    const auto ovvv_workers = std::min(
        occupied_virtual,
        static_cast<std::size_t>(omp_get_max_threads()));
    // The OOVV product has only o^2 output rows.  Requiring a small row slab
    // per worker avoids launching the serial-BLAS interface on many extremely
    // narrow products when a node exposes substantially more threads than
    // occupied pairs.
    constexpr std::size_t minimum_oovv_rows_per_worker = 8;
    const auto oovv_workers = std::min(
        (occupied_pairs + minimum_oovv_rows_per_worker - 1) /
            minimum_oovv_rows_per_worker,
        static_cast<std::size_t>(omp_get_max_threads()));
#endif

    const auto contraction_begin = SteadyClock::now();
#if defined(_OPENMP)
#pragma omp parallel num_threads(static_cast<int>(task_workers))
#endif
    {
      std::size_t worker = 0;
      std::size_t workers = 1;
#if defined(_OPENMP)
      worker = static_cast<std::size_t>(omp_get_thread_num());
      workers = static_cast<std::size_t>(omp_get_num_threads());
#endif
      const auto first = task_count * worker / workers;
      const auto last = task_count * (worker + 1) / workers;
      const int columns = checked_blas_dimension(
          last - first, "factorized external-pair slab");
      if (columns != 0) {
        const auto cached_first =
            invariant_cache->task_to_local[tasks[first]];
        dgemm_(&transpose, &no_transpose, &blas_occupied_pairs, &columns,
               &blas_virtual_pairs, &one, tau.data().data(),
               &blas_virtual_pairs,
               invariant_cache->bare_vvvv.data() +
                   static_cast<std::ptrdiff_t>(cached_first * virtual_pairs),
               &blas_virtual_pairs, &zero,
               evaluation.result.contractions.data() +
                   static_cast<std::ptrdiff_t>(first * occupied_pairs),
               &blas_occupied_pairs);
      }
    }
    evaluation.contraction_seconds +=
        elapsed_seconds(contraction_begin, SteadyClock::now());

    const auto build_begin = SteadyClock::now();
    // Split the output rows rather than the occupied-pair columns.  Each
    // serial-BLAS call then retains the complete occupied-pair dimension,
    // increasing arithmetic intensity without replicating an output element.
#if defined(_OPENMP)
#pragma omp parallel num_threads(static_cast<int>(ovvv_workers))
#endif
    {
      std::size_t worker = 0;
      std::size_t workers = 1;
#if defined(_OPENMP)
      worker = static_cast<std::size_t>(omp_get_thread_num());
      workers = static_cast<std::size_t>(omp_get_num_threads());
#endif
      const auto first = occupied_virtual * worker / workers;
      const auto last = occupied_virtual * (worker + 1) / workers;
      const int rows = checked_blas_dimension(
          last - first, "factorized occupied-virtual slab");
      if (rows != 0)
        dgemm_(&transpose, &no_transpose, &rows, &blas_occupied_pairs,
               &blas_virtual_pairs, &one,
               invariant_cache->ovvv.data() +
                   static_cast<std::ptrdiff_t>(first * virtual_pairs),
               &blas_virtual_pairs, tau.data().data(),
               &blas_virtual_pairs, &zero,
               tau_ovvv.data() + static_cast<std::ptrdiff_t>(first),
               &blas_occupied_virtual);
    }
#if defined(_OPENMP)
#pragma omp parallel num_threads(static_cast<int>(oovv_workers))
#endif
    {
      std::size_t worker = 0;
      std::size_t workers = 1;
#if defined(_OPENMP)
      worker = static_cast<std::size_t>(omp_get_thread_num());
      workers = static_cast<std::size_t>(omp_get_num_threads());
#endif
      const auto first = occupied_pairs * worker / workers;
      const auto last = occupied_pairs * (worker + 1) / workers;
      const int rows = checked_blas_dimension(
          last - first, "factorized OOVV row slab");
      if (rows != 0)
        dgemm_(&transpose, &no_transpose, &rows, &blas_occupied_pairs,
               &blas_virtual_pairs, &one,
               invariant_cache->oovv.data() +
                   static_cast<std::ptrdiff_t>(first * virtual_pairs),
               &blas_virtual_pairs, tau.data().data(),
               &blas_virtual_pairs, &zero,
               tau_oovv.data() + static_cast<std::ptrdiff_t>(first),
               &blas_occupied_pairs);
    }
    evaluation.build_seconds +=
        elapsed_seconds(build_begin, SteadyClock::now());

    const auto accumulation_begin = SteadyClock::now();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t local_index = 0;
         local_index < static_cast<std::ptrdiff_t>(task_count);
         ++local_index) {
      const auto local = static_cast<std::size_t>(local_index);
      const auto a = tasks[local] / virtuals;
      const auto b = tasks[local] % virtuals;
      auto* const output = evaluation.result.contractions.data() +
          static_cast<std::ptrdiff_t>(local * occupied_pairs);
      for (std::size_t ij = 0; ij < occupied_pairs; ++ij) {
        const auto* const contracted_ovvv = tau_ovvv.data() +
            static_cast<std::ptrdiff_t>(ij * occupied_virtual);
        double singles_dressing = 0.0;
        for (std::size_t m = 0; m < occupied; ++m)
          singles_dressing +=
              -contracted_ovvv[a * occupied + m] * t1(m, b) +
              contracted_ovvv[b * occupied + m] * t1(m, a);
        output[ij] += singles_dressing;
      }
    }

#if defined(_OPENMP)
#pragma omp parallel num_threads(static_cast<int>(task_workers))
#endif
    {
      std::size_t worker = 0;
      std::size_t workers = 1;
#if defined(_OPENMP)
      worker = static_cast<std::size_t>(omp_get_thread_num());
      workers = static_cast<std::size_t>(omp_get_num_threads());
#endif
      const auto first = task_count * worker / workers;
      const auto last = task_count * (worker + 1) / workers;
      const int columns = checked_blas_dimension(
          last - first, "factorized external-pair slab");
      if (columns != 0)
        dgemm_(&transpose, &no_transpose, &blas_occupied_pairs, &columns,
               &blas_occupied_pairs, &quarter, tau_oovv.data(),
               &blas_occupied_pairs,
               tau_ab_scalars.data() +
                   static_cast<std::ptrdiff_t>(first * occupied_pairs),
               &blas_occupied_pairs, &one,
               evaluation.result.contractions.data() +
                   static_cast<std::ptrdiff_t>(first * occupied_pairs),
               &blas_occupied_pairs);
    }
    evaluation.contraction_seconds +=
        elapsed_seconds(accumulation_begin, SteadyClock::now());
    for (const double value : evaluation.result.contractions)
      if (!std::isfinite(value))
        throw std::runtime_error(
            "factorized cached CCSD W_abef contraction produced a "
            "non-finite value");
    return evaluation;
  }

  if (occupied_pairs >
      (std::numeric_limits<std::size_t>::max() - task_count) / 2)
    throw std::overflow_error("tiled CCSD W_abef panel size overflows");
  const auto values_per_virtual_pair = invariant_cache
      ? task_count
      : task_count + 2 * occupied_pairs;
  const auto bytes_per_virtual_pair =
      checked_memory_product(values_per_virtual_pair, sizeof(double));
  const auto panel_capacity = std::max<std::size_t>(
      1, maximum_panel_bytes /
             std::max<std::size_t>(1, bytes_per_virtual_pair));

  TiledWabefEvaluation evaluation;
  evaluation.result.contractions.assign(
      checked_memory_product(task_count, occupied_pairs), 0.0);
  constexpr char no_transpose = 'N';
  constexpr char transpose = 'T';
  constexpr double quarter = 0.25;
  constexpr double one = 1.0;
#if defined(_OPENMP)
  const auto requested_workers = std::min(
      task_count, static_cast<std::size_t>(omp_get_max_threads()));
#endif

  for (std::size_t panel_first = 0; panel_first < virtual_pairs;
       panel_first += panel_capacity) {
    const auto build_begin = SteadyClock::now();
    const auto panel_count =
        std::min(panel_capacity, virtual_pairs - panel_first);
    ++evaluation.virtual_pair_panels;
    evaluation.blas_contractions += 2;
    std::vector<double> wabef_panel(
        checked_memory_product(task_count, panel_count));
    std::vector<double> oovv_panel;
    std::vector<double> tau_panel;
    if (!invariant_cache) {
      oovv_panel.resize(
          checked_memory_product(occupied_pairs, panel_count));
      tau_panel.resize(
          checked_memory_product(occupied_pairs, panel_count));
    }

    // Form the bare VVVV and singles-dressed OVVV pieces directly into the
    // bounded internal-(e,f) panel of uniquely owned external-(a,b) rows.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t local_index = 0;
         local_index < static_cast<std::ptrdiff_t>(task_count);
         ++local_index) {
      const auto local = static_cast<std::size_t>(local_index);
      const auto a = tasks[local] / virtuals;
      const auto b = tasks[local] % virtuals;
      auto* const output =
          wabef_panel.data() +
          static_cast<std::ptrdiff_t>(local * panel_count);
      if (invariant_cache) {
        const auto cached_local =
            invariant_cache->task_to_local[tasks[local]];
        const auto* const bare =
            invariant_cache->bare_vvvv.data() +
            static_cast<std::ptrdiff_t>(cached_local * virtual_pairs +
                                        panel_first);
        std::copy_n(bare, panel_count, output);
        // Traverse each cached OVVV row contiguously. The former ef-major
        // loop jumped by virtual_pairs for every occupied index, defeating
        // cache-line reuse and SIMD on diffuse-basis panels. This reordered
        // accumulation evaluates the same singles dressing while retaining
        // one bounded external-pair row as the only live output workspace.
        for (std::size_t m = 0; m < occupied; ++m) {
          const double left_coefficient = -t1(m, b);
          const double right_coefficient = t1(m, a);
          const auto* const left =
              invariant_cache->ovvv.data() +
              static_cast<std::ptrdiff_t>(
                  (a * occupied + m) * virtual_pairs + panel_first);
          const auto* const right =
              invariant_cache->ovvv.data() +
              static_cast<std::ptrdiff_t>(
                  (b * occupied + m) * virtual_pairs + panel_first);
#if defined(_OPENMP)
#pragma omp simd
#endif
          for (std::size_t local_ef = 0; local_ef < panel_count; ++local_ef)
            output[local_ef] += left_coefficient * left[local_ef] +
                                right_coefficient * right[local_ef];
        }
      } else {
        for (std::size_t local_ef = 0; local_ef < panel_count; ++local_ef) {
          const auto ef = panel_first + local_ef;
          const auto e = ef / virtuals;
          const auto q = ef % virtuals;
          double value =
              eri(occupied + a, occupied + b, occupied + e, occupied + q);
          for (std::size_t m = 0; m < occupied; ++m)
            value -=
                t1(m, b) *
                    eri(occupied + a, m, occupied + e, occupied + q) -
                t1(m, a) *
                    eri(occupied + b, m, occupied + e, occupied + q);
          output[local_ef] = value;
        }
      }
    }

    if (!invariant_cache) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t pair_index = 0;
         pair_index < static_cast<std::ptrdiff_t>(occupied_pairs);
         ++pair_index) {
      const auto pair = static_cast<std::size_t>(pair_index);
      const auto m = pair / occupied;
      const auto n = pair % occupied;
      auto* const integral_output =
          oovv_panel.data() +
          static_cast<std::ptrdiff_t>(pair * panel_count);
      auto* const amplitude_output =
          tau_panel.data() +
          static_cast<std::ptrdiff_t>(pair * panel_count);
      for (std::size_t local_ef = 0; local_ef < panel_count; ++local_ef) {
        const auto ef = panel_first + local_ef;
        const auto e = ef / virtuals;
        const auto q = ef % virtuals;
        integral_output[local_ef] =
            eri(m, n, occupied + e, occupied + q);
        amplitude_output[local_ef] = tau(m, n, e, q);
      }
    }
    }

    const int rows =
        checked_blas_dimension(panel_count, "internal virtual-pair panel");
    const int panel_leading_dimension = invariant_cache
        ? checked_blas_dimension(virtual_pairs,
                                 "cached internal virtual-pair extent")
        : rows;
    const int occupied_contraction =
        checked_blas_dimension(occupied_pairs, "occupied-pair tile");
#if defined(_OPENMP)
#pragma omp parallel num_threads(static_cast<int>(requested_workers))
#endif
    {
      std::size_t worker = 0;
      std::size_t workers = 1;
#if defined(_OPENMP)
      worker = static_cast<std::size_t>(omp_get_thread_num());
      workers = static_cast<std::size_t>(omp_get_num_threads());
#endif
      const auto first = task_count * worker / workers;
      const auto last = task_count * (worker + 1) / workers;
      const int columns = checked_blas_dimension(
          last - first, "external virtual-pair slab");
      if (columns != 0)
        dgemm_(&no_transpose, &no_transpose, &rows, &columns,
               &occupied_contraction, &quarter,
               invariant_cache
                   ? invariant_cache->oovv.data() +
                         static_cast<std::ptrdiff_t>(panel_first)
                   : oovv_panel.data(),
               &panel_leading_dimension,
               tau_ab_scalars.data() +
                   static_cast<std::ptrdiff_t>(first * occupied_pairs),
               &occupied_contraction, &one,
               wabef_panel.data() +
                   static_cast<std::ptrdiff_t>(first * panel_count),
               &rows);
    }
    evaluation.build_seconds +=
        elapsed_seconds(build_begin, SteadyClock::now());

    const auto contraction_begin = SteadyClock::now();
    const int output_rows =
        checked_blas_dimension(occupied_pairs, "occupied-pair output");
#if defined(_OPENMP)
#pragma omp parallel num_threads(static_cast<int>(requested_workers))
#endif
    {
      std::size_t worker = 0;
      std::size_t workers = 1;
#if defined(_OPENMP)
      worker = static_cast<std::size_t>(omp_get_thread_num());
      workers = static_cast<std::size_t>(omp_get_num_threads());
#endif
      const auto first = task_count * worker / workers;
      const auto last = task_count * (worker + 1) / workers;
      const int columns = checked_blas_dimension(
          last - first, "external virtual-pair slab");
      if (columns != 0)
        dgemm_(&transpose, &no_transpose, &output_rows, &columns, &rows,
               &one,
               invariant_cache
                   ? tau.data().data() +
                         static_cast<std::ptrdiff_t>(panel_first)
                   : tau_panel.data(),
               &panel_leading_dimension,
               wabef_panel.data() +
                   static_cast<std::ptrdiff_t>(first * panel_count),
               &rows, &one,
               evaluation.result.contractions.data() +
                   static_cast<std::ptrdiff_t>(first * occupied_pairs),
               &output_rows);
    }
    evaluation.contraction_seconds +=
        elapsed_seconds(contraction_begin, SteadyClock::now());
  }
  evaluation.result.workspace_bytes = checked_memory_product(
      evaluation.result.contractions.size(), sizeof(double));
  for (const double value : evaluation.result.contractions)
    if (!std::isfinite(value))
      throw std::runtime_error(
          "tiled CPU CCSD W_abef contraction produced a non-finite value");
  return evaluation;
}

template <typename Eri>
std::pair<Matrix, Tensor4> fixed_point(
    const Matrix& fock, const Eri& eri, const Matrix& t1,
    const Tensor4& t2, const std::size_t occupied, const Matrix& d1,
    const Tensor4& d2, CcsdIterationTiming* const timing = nullptr,
    const CcsdWabefBackend wabef_backend = CcsdWabefBackend::cpu,
    const bool use_tiled_intermediates = false,
    const bool use_tiled_doubles = false,
    const std::size_t wabef_tile_max_bytes = 64U * 1024U * 1024U,
    const CcsdProjectionMask* const projection_mask = nullptr,
    const TiledWabefInvariantCache* const wabef_invariant_cache = nullptr,
    const SelectedIntegralPanelCache* const selected_panel_cache = nullptr,
    const bool skip_singles = false) {
  const auto virtuals = t1.dimensions()[1];
  if (wabef_invariant_cache && wabef_invariant_cache->pm_symmetry)
    validate_pm_symmetry_amplitudes(*wabef_invariant_cache->pm_symmetry, t1, t2);
  const auto intermediates_begin = SteadyClock::now();
  const auto work = intermediates(fock, eri, t1, t2, occupied,
                                  use_tiled_intermediates, timing,
                                  selected_panel_cache, projection_mask);
  const auto intermediates_end = SteadyClock::now();
  if (timing)
    timing->intermediates_seconds +=
        elapsed_seconds(intermediates_begin, intermediates_end);
  Matrix next_t1({occupied, virtuals});
  Tensor4 next_t2({occupied, occupied, virtuals, virtuals});

  const auto singles_begin = SteadyClock::now();
  if (!skip_singles) {
    distributed_for(occupied * virtuals, [&](const std::size_t flat) {
      if (projection_mask && projection_mask->singles[flat] == 0U)
        return;
      const auto i = flat / virtuals;
      const auto a = flat % virtuals;
      double value = fock(i, occupied + a);
      for (std::size_t e = 0; e < virtuals; ++e)
        value += t1(i, e) * work.fae(a, e);
      for (std::size_t m = 0; m < occupied; ++m)
        value -= t1(m, a) * work.fmi(m, i);
      for (std::size_t m = 0; m < occupied; ++m)
        for (std::size_t e = 0; e < virtuals; ++e)
          value += t2(i, m, a, e) * work.fme(m, e) -
                   t1(m, e) * eri(m, occupied + a, i, occupied + e);
      for (std::size_t m = 0; m < occupied; ++m)
        for (std::size_t e = 0; e < virtuals; ++e)
          for (std::size_t q = 0; q < virtuals; ++q)
            value -= 0.5 * t2(i, m, e, q) *
                     eri(m, occupied + a, occupied + e, occupied + q);
      for (std::size_t m = 0; m < occupied; ++m)
        for (std::size_t n = 0; n < occupied; ++n)
          for (std::size_t e = 0; e < virtuals; ++e)
            value -= 0.5 * t2(m, n, a, e) *
                     eri(n, m, occupied + e, i);
      next_t1(i, a) = value / d1(i, a);
    });
    parallel::sum_in_place(next_t1.data());
  }
  const auto singles_end = SteadyClock::now();
  if (timing)
    timing->singles_seconds += elapsed_seconds(singles_begin, singles_end);

  const auto fae_term = [&](const std::size_t i, const std::size_t j,
                            const std::size_t a, const std::size_t b) {
    double value = 0.0;
    for (std::size_t e = 0; e < virtuals; ++e) {
      double intermediate = work.fae(b, e);
      for (std::size_t m = 0; m < occupied; ++m)
        intermediate -= 0.5 * t1(m, b) * work.fme(m, e);
      value += t2(i, j, a, e) * intermediate;
    }
    return value;
  };
  const auto fmi_term = [&](const std::size_t i, const std::size_t j,
                            const std::size_t a, const std::size_t b) {
    double value = 0.0;
    for (std::size_t m = 0; m < occupied; ++m) {
      double intermediate = work.fmi(m, j);
      for (std::size_t e = 0; e < virtuals; ++e)
        intermediate += 0.5 * t1(j, e) * work.fme(m, e);
      value += t2(i, m, a, b) * intermediate;
    }
    return value;
  };
  const auto wm_term = [&](const std::size_t i, const std::size_t j,
                           const std::size_t a, const std::size_t b) {
    double value = 0.0;
    for (std::size_t m = 0; m < occupied; ++m)
      for (std::size_t e = 0; e < virtuals; ++e)
        value += t2(i, m, a, e) * work.wmbej(m, b, e, j) -
                 t1(i, e) * t1(m, a) * eri(m, occupied + b, occupied + e, j);
    return value;
  };
  const auto abej = [&](const std::size_t i, const std::size_t j,
                        const std::size_t a, const std::size_t b) {
    double value = 0.0;
    for (std::size_t e = 0; e < virtuals; ++e)
      value += t1(i, e) * eri(occupied + a, occupied + b, occupied + e, j);
    return value;
  };
  const auto mbij = [&](const std::size_t i, const std::size_t j,
                        const std::size_t a, const std::size_t b) {
    double value = 0.0;
    for (std::size_t m = 0; m < occupied; ++m)
      value += t1(m, a) * eri(m, occupied + b, i, j);
    return value;
  };

  const auto doubles_begin = SteadyClock::now();
  const auto doubles_count = next_t2.size();
  if (use_tiled_doubles) {
    const auto evaluation = evaluate_tiled_doubles_without_wabef(
        eri, t1, t2, work, occupied, d2, next_t2, projection_mask,
        selected_panel_cache);
    if (timing) {
      timing->doubles_occupied_pair_tasks +=
          evaluation.occupied_pair_tasks;
      timing->doubles_blas_contractions += evaluation.blas_contractions;
    }
  } else {
    distributed_for(doubles_count, [&](const std::size_t flat) {
      if (projection_mask && projection_mask->doubles[flat] == 0U)
        return;
      auto index = flat;
      const auto b = index % virtuals;
      index /= virtuals;
      const auto a = index % virtuals;
      index /= virtuals;
      const auto j = index % occupied;
      const auto i = index / occupied;
      double value = eri(i, j, occupied + a, occupied + b) +
                     fae_term(i, j, a, b) - fae_term(i, j, b, a) -
                     fmi_term(i, j, a, b) + fmi_term(j, i, a, b);
      for (std::size_t m = 0; m < occupied; ++m)
        for (std::size_t n = 0; n < occupied; ++n)
          value += 0.5 * work.tau(m, n, a, b) * work.wmnij(m, n, i, j);
      value += wm_term(i, j, a, b) - wm_term(j, i, a, b) -
               wm_term(i, j, b, a) + wm_term(j, i, b, a);
      value += abej(i, j, a, b) - abej(j, i, a, b) -
               mbij(i, j, a, b) + mbij(i, j, b, a);
      next_t2(i, j, a, b) = value / d2(i, j, a, b);
    });
  }
  const auto doubles_end = SteadyClock::now();
  if (timing)
    timing->doubles_seconds += elapsed_seconds(doubles_begin, doubles_end);

  // Stream the dominant W_abef contribution in bounded virtual-pair tiles.
  // Building remains on the host because the production ERI object is a
  // zero-copy spatial view. The final dense dot products may be sent to CUDA.
  const auto wabef_begin = SteadyClock::now();
  if (wabef_tile_max_bytes == 0)
    throw std::invalid_argument("CCSD W_abef tile memory must be nonzero");
  const auto virtual_pair_count = virtuals * virtuals;
  const auto owned_virtual_pair_tasks = wabef_invariant_cache
      ? wabef_invariant_cache->tasks
      : owned_wabef_tasks(occupied, virtuals, projection_mask);
  const auto owned_virtual_pairs = owned_virtual_pair_tasks.size();
  const auto block_size = virtual_pair_count;
  const auto bytes_per_block =
      checked_memory_product(block_size, sizeof(double));
  auto tile_capacity = std::max<std::size_t>(
      1, wabef_tile_max_bytes /
             std::max<std::size_t>(1, bytes_per_block));
  if (wabef_invariant_cache &&
      factorized_cached_wabef_workspace_bytes(
          occupied, virtuals, owned_virtual_pairs) <= wabef_tile_max_bytes)
    tile_capacity = std::max<std::size_t>(1, owned_virtual_pairs);
  for (std::size_t tile_begin = 0; tile_begin < owned_virtual_pairs;
       tile_begin += tile_capacity) {
    const auto tile_build_begin = SteadyClock::now();
    const auto tile_count =
        std::min(tile_capacity, owned_virtual_pairs - tile_begin);
    std::vector<std::size_t> tasks(tile_count);
    for (std::size_t local = 0; local < tile_count; ++local)
      tasks[local] = owned_virtual_pair_tasks[tile_begin + local];
    if (timing) {
      ++timing->wabef_external_tiles;
      timing->wabef_output_tasks += tile_count;
    }

    CcsdWabefResult contractions;
    if (wabef_backend == CcsdWabefBackend::tiled_cpu) {
      std::vector<double> tau_ab_scalars(checked_memory_product(
          tile_count, checked_memory_product(occupied, occupied)));
      for (std::size_t local = 0; local < tile_count; ++local) {
        const auto a = tasks[local] / virtuals;
        const auto b = tasks[local] % virtuals;
        for (std::size_t m = 0; m < occupied; ++m)
          for (std::size_t n = 0; n < occupied; ++n)
            tau_ab_scalars[(local * occupied + m) * occupied + n] =
                work.tau(m, n, a, b);
      }
      const auto evaluation_begin = SteadyClock::now();
      auto evaluation = evaluate_tiled_wabef(
          eri, t1, occupied, work.tau, tasks, tau_ab_scalars,
          wabef_tile_max_bytes, wabef_invariant_cache);
      contractions = std::move(evaluation.result);
      if (timing) {
        timing->wabef_build_seconds +=
            elapsed_seconds(tile_build_begin, evaluation_begin) +
            evaluation.build_seconds;
        timing->wabef_contraction_seconds +=
            evaluation.contraction_seconds;
        timing->wabef_internal_panels += evaluation.virtual_pair_panels;
        timing->wabef_blas_contractions += evaluation.blas_contractions;
        timing->wabef_factorized_cached_evaluations +=
            evaluation.factorized_cached ? 1U : 0U;
      }
    } else {
      CcsdWabefTile contraction_tile;
      contraction_tile.occupied_pairs = occupied * occupied;
      contraction_tile.virtual_pair_elements = block_size;
      contraction_tile.virtual_pair_tasks = tile_count;
      contraction_tile.tau = work.tau.data();
      std::vector<Matrix> wabef_blocks;
      wabef_blocks.reserve(tile_count);
      for (std::size_t local = 0; local < tile_count; ++local)
        wabef_blocks.emplace_back(
            std::array<std::size_t, 2>{virtuals, virtuals});

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (std::ptrdiff_t local_index = 0;
           local_index < static_cast<std::ptrdiff_t>(tile_count);
           ++local_index) {
        const auto local = static_cast<std::size_t>(local_index);
        const auto a = tasks[local] / virtuals;
        const auto b = tasks[local] % virtuals;
        auto& wabef_block = wabef_blocks[local];
        for (std::size_t e = 0; e < virtuals; ++e)
          for (std::size_t q = 0; q < virtuals; ++q) {
            double value = eri(occupied + a, occupied + b, occupied + e,
                               occupied + q);
            for (std::size_t m = 0; m < occupied; ++m)
              value -=
                  t1(m, b) *
                      eri(occupied + a, m, occupied + e, occupied + q) -
                  t1(m, a) *
                      eri(occupied + b, m, occupied + e, occupied + q);
            for (std::size_t m = 0; m < occupied; ++m)
              for (std::size_t n = 0; n < occupied; ++n)
                value += 0.25 * work.tau(m, n, a, b) *
                         eri(m, n, occupied + e, occupied + q);
            wabef_block(e, q) = value;
          }
      }
      contraction_tile.wabef.resize(tile_count * block_size);
      for (std::size_t local = 0; local < tile_count; ++local)
        std::copy(wabef_blocks[local].data().begin(),
                  wabef_blocks[local].data().end(),
                  contraction_tile.wabef.begin() +
                      static_cast<std::ptrdiff_t>(local * block_size));
      const auto tile_build_end = SteadyClock::now();
      if (timing)
        timing->wabef_build_seconds +=
            elapsed_seconds(tile_build_begin, tile_build_end);
      const auto contraction_begin = SteadyClock::now();
      contractions = wabef_backend == CcsdWabefBackend::cuda
          ? contract_ccsd_wabef_cuda(contraction_tile)
          : contract_ccsd_wabef_cpu(contraction_tile);
      const auto contraction_end = SteadyClock::now();
      if (timing)
        timing->wabef_contraction_seconds +=
            elapsed_seconds(contraction_begin, contraction_end);
      if (timing) {
        ++timing->wabef_internal_panels;
        ++timing->wabef_blas_contractions;
      }
    }
    const auto accumulation_begin = SteadyClock::now();
    for (std::size_t local = 0; local < tile_count; ++local) {
      const auto task = tasks[local];
      const auto a = task / virtuals;
      const auto b = task % virtuals;
      for (std::size_t pair = 0; pair < occupied * occupied; ++pair) {
        const auto i = pair / occupied;
        const auto j = pair % occupied;
        if (projection_mask) {
          const auto flat = ((i * occupied + j) * virtuals + a) * virtuals + b;
          if (projection_mask->doubles[flat] == 0U)
            continue;
        }
        if (wabef_invariant_cache && wabef_invariant_cache->pm_symmetry) {
          const auto& symmetry = *wabef_invariant_cache->pm_symmetry;
          std::array<std::size_t, 4> visited{};
          std::size_t count=0;
          for (const auto& g:symmetry.operations) {
            const auto A=pm_spin_image(g,a,symmetry.occupied);
            const auto B=pm_spin_image(g,b,symmetry.occupied);
            const auto mapped=A*virtuals+B;
            if (std::find(visited.begin(),visited.begin()+static_cast<std::ptrdiff_t>(count),mapped)!=visited.begin()+static_cast<std::ptrdiff_t>(count)) continue;
            visited[count++]=mapped;
            const auto I=pm_spin_image(g,i,0),J=pm_spin_image(g,j,0);
            const int sign=pm_spin_sign(g,i,0)*pm_spin_sign(g,j,0)*
                pm_spin_sign(g,a,symmetry.occupied)*pm_spin_sign(g,b,symmetry.occupied);
            next_t2(I,J,A,B) += sign*0.5*
                contractions.contractions[local*occupied*occupied+pair]/d2(I,J,A,B);
          }
          continue;
        }
        next_t2(i, j, a, b) +=
            0.5 * contractions.contractions[local * occupied * occupied + pair] /
            d2(i, j, a, b);
        if (wabef_invariant_cache && wabef_invariant_cache->pair_symmetry) {
          // W_baef = -W_abef, including singles and tau dressing. The
          // diagonal a=b is zero. Restore the exchanged residual on the same
          // owner before the existing collective; no second contraction.
          next_t2(i, j, b, a) -=
              0.5 * contractions.contractions[local * occupied * occupied + pair] /
              d2(i, j, b, a);
        }
      }
    }
    if (timing)
      timing->wabef_accumulation_seconds +=
          elapsed_seconds(accumulation_begin, SteadyClock::now());
  }
  parallel::sum_in_place(next_t2.data());
  const auto wabef_end = SteadyClock::now();
  if (timing)
    timing->wabef_seconds += elapsed_seconds(wabef_begin, wabef_end);
  return {std::move(next_t1), antisym(next_t2)};
}

void antisymmetrize_distributed(const GlobalArraysPairTensor& input,
                               GlobalArraysPairTensor& output) {
  const auto& layout = input.layout();
  if (output.layout().occupied() != layout.occupied() ||
      output.layout().virtuals() != layout.virtuals())
    throw std::invalid_argument("distributed antisymmetrization layout mismatch");
  output.zero();
  const auto communicator = parallel::world();
  const auto virtuals = layout.virtuals();
  for (std::size_t pair = communicator.rank; pair < layout.pair_count();
       pair += communicator.size) {
    const auto [i, j] = layout.pair_from_flat(pair);
    const auto direct = input.get_pair(i, j);
    const auto exchanged = input.get_pair(j, i);
    std::vector<double> block(virtuals * virtuals);
    for (std::size_t a = 0; a < virtuals; ++a)
      for (std::size_t b = 0; b < virtuals; ++b)
        block[a * virtuals + b] =
            0.25 * (direct[a * virtuals + b] -
                    exchanged[a * virtuals + b] -
                    direct[b * virtuals + a] +
                    exchanged[b * virtuals + a]);
    output.put_pair(i, j, block);
  }
  output.synchronize();
}

template <typename Eri>
Matrix fixed_point_distributed(
    const Matrix& fock, const Eri& eri, const Matrix& t1,
    const GlobalArraysPairTensor& t2, const std::size_t occupied,
    const Matrix& d1, const Tensor4& d2, GlobalArraysPairTensor& raw_t2,
    GlobalArraysPairTensor& next_t2, GlobalArraysPairTensor& tau,
    GlobalArraysPairTensor& tilde_tau, GlobalArraysPairTensor& wmbej,
    GlobalArraysPairTensor& wmnij,
    CcsdIterationTiming* const timing = nullptr,
    const std::size_t wabef_tile_max_bytes = 64U * 1024U * 1024U) {
  const auto virtuals = t1.dimensions()[1];
  const auto intermediates_begin = SteadyClock::now();
  const auto work = distributed_intermediates(
      fock, eri, t1, t2, occupied, tau, tilde_tau, wmbej, wmnij);
  const auto intermediates_end = SteadyClock::now();
  if (timing)
    timing->intermediates_seconds +=
        elapsed_seconds(intermediates_begin, intermediates_end);
  const auto communicator = parallel::world();
  const auto pair_count = occupied * occupied;
  const auto block_size = virtuals * virtuals;
  Matrix next_t1({occupied, virtuals});

  const auto singles_begin = SteadyClock::now();
  for (std::size_t flat = communicator.rank; flat < occupied * virtuals;
       flat += communicator.size) {
    const auto i = flat / virtuals;
    const auto a = flat % virtuals;
    double value = fock(i, occupied + a);
    for (std::size_t e = 0; e < virtuals; ++e)
      value += t1(i, e) * work.fae(a, e);
    for (std::size_t m = 0; m < occupied; ++m)
      value -= t1(m, a) * work.fmi(m, i);
    for (std::size_t m = 0; m < occupied; ++m)
      for (std::size_t e = 0; e < virtuals; ++e)
        value -= t1(m, e) * eri(m, occupied + a, i, occupied + e);
    next_t1(i, a) = value;
  }
  for (std::size_t pair = communicator.rank; pair < pair_count;
       pair += communicator.size) {
    const auto p = pair / occupied;
    const auto q = pair % occupied;
    const auto block = t2.get_pair(p, q);
    for (std::size_t a = 0; a < virtuals; ++a) {
      for (std::size_t e = 0; e < virtuals; ++e)
        next_t1(p, a) +=
            block[a * virtuals + e] * work.fme(q, e);
      for (std::size_t e = 0; e < virtuals; ++e)
        for (std::size_t r = 0; r < virtuals; ++r)
          next_t1(p, a) -=
              0.5 * block[e * virtuals + r] *
              eri(q, occupied + a, occupied + e, occupied + r);
    }
    for (std::size_t i = 0; i < occupied; ++i)
      for (std::size_t a = 0; a < virtuals; ++a)
        for (std::size_t e = 0; e < virtuals; ++e)
          next_t1(i, a) -=
              0.5 * block[a * virtuals + e] *
              eri(q, p, occupied + e, i);
  }
  parallel::sum_in_place(next_t1.data());
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t a = 0; a < virtuals; ++a)
      next_t1(i, a) /= d1(i, a);
  const auto singles_end = SteadyClock::now();
  if (timing)
    timing->singles_seconds += elapsed_seconds(singles_begin, singles_end);

  const auto doubles_begin = SteadyClock::now();
  raw_t2.zero();
  for (std::size_t pair = communicator.rank; pair < pair_count;
       pair += communicator.size) {
    const auto i = pair / occupied;
    const auto j = pair % occupied;
    std::vector<std::vector<double>> t2_im;
    std::vector<std::vector<double>> t2_jm;
    std::vector<std::vector<double>> wm_mj;
    std::vector<std::vector<double>> wm_mi;
    t2_im.reserve(occupied);
    t2_jm.reserve(occupied);
    wm_mj.reserve(occupied);
    wm_mi.reserve(occupied);
    for (std::size_t m = 0; m < occupied; ++m) {
      t2_im.push_back(t2.get_pair(i, m));
      t2_jm.push_back(t2.get_pair(j, m));
      wm_mj.push_back(wmbej.get_pair(m, j));
      wm_mi.push_back(wmbej.get_pair(m, i));
    }
    const auto& direct = t2_im[j];
    std::vector<double> block(block_size);
    if (block_size >
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
      throw std::overflow_error("distributed CCSD pair extent exceeds ptrdiff_t");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t flat = 0;
         flat < static_cast<std::ptrdiff_t>(block_size); ++flat) {
      const auto a = static_cast<std::size_t>(flat) / virtuals;
      const auto b = static_cast<std::size_t>(flat) % virtuals;
      const auto fae_term = [&](const std::size_t left,
                                const std::size_t right) {
        double term = 0.0;
        for (std::size_t e = 0; e < virtuals; ++e) {
          double intermediate = work.fae(right, e);
          for (std::size_t m = 0; m < occupied; ++m)
            intermediate -= 0.5 * t1(m, right) * work.fme(m, e);
          term += direct[left * virtuals + e] * intermediate;
        }
        return term;
      };
      const auto fmi_term = [&](const bool exchanged) {
        const auto& cache = exchanged ? t2_jm : t2_im;
        const auto right_hole = exchanged ? i : j;
        double term = 0.0;
        for (std::size_t m = 0; m < occupied; ++m) {
          double intermediate = work.fmi(m, right_hole);
          for (std::size_t e = 0; e < virtuals; ++e)
            intermediate +=
                0.5 * t1(right_hole, e) * work.fme(m, e);
          term += cache[m][a * virtuals + b] * intermediate;
        }
        return term;
      };
      const auto wm_term = [&](const bool exchanged_holes,
                               const std::size_t left_virtual,
                               const std::size_t right_virtual) {
        const auto left_hole = exchanged_holes ? j : i;
        const auto& amplitude_cache = exchanged_holes ? t2_jm : t2_im;
        const auto& wm_cache = exchanged_holes ? wm_mi : wm_mj;
        double term = 0.0;
        for (std::size_t m = 0; m < occupied; ++m)
          for (std::size_t e = 0; e < virtuals; ++e)
            term +=
                amplitude_cache[m][left_virtual * virtuals + e] *
                    wm_cache[m][right_virtual * virtuals + e] -
                t1(left_hole, e) * t1(m, left_virtual) *
                    eri(m, occupied + right_virtual, occupied + e,
                        exchanged_holes ? i : j);
        return term;
      };
      double value =
          eri(i, j, occupied + a, occupied + b) +
          fae_term(a, b) - fae_term(b, a) - fmi_term(false) +
          fmi_term(true);
      value += wm_term(false, a, b) - wm_term(true, a, b) -
               wm_term(false, b, a) + wm_term(true, b, a);
      for (std::size_t e = 0; e < virtuals; ++e)
        value +=
            t1(i, e) *
                eri(occupied + a, occupied + b, occupied + e, j) -
            t1(j, e) *
                eri(occupied + a, occupied + b, occupied + e, i);
      for (std::size_t m = 0; m < occupied; ++m)
        value +=
            -t1(m, a) * eri(m, occupied + b, i, j) +
            t1(m, b) * eri(m, occupied + a, i, j);
      block[a * virtuals + b] = value / d2(i, j, a, b);
    }
    const auto wmnij_block = wmnij.get_pair(i, j);
    for (std::size_t m = 0; m < occupied; ++m)
      for (std::size_t n = 0; n < occupied; ++n) {
        const auto tau_block = tau.get_pair(m, n);
        const auto coefficient =
            0.5 * wmnij_block[m * occupied + n];
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t flat = 0;
             flat < static_cast<std::ptrdiff_t>(block_size); ++flat) {
          const auto a = static_cast<std::size_t>(flat) / virtuals;
          const auto b = static_cast<std::size_t>(flat) % virtuals;
          block[static_cast<std::size_t>(flat)] +=
              coefficient * tau_block[static_cast<std::size_t>(flat)] /
              d2(i, j, a, b);
        }
      }
    raw_t2.put_pair(i, j, block);
  }
  raw_t2.synchronize();
  const auto doubles_end = SteadyClock::now();
  if (timing)
    timing->doubles_seconds += elapsed_seconds(doubles_begin, doubles_end);

  // W_abef remains streamed by (a,b), but owned virtual pairs are processed
  // in bounded coarse tiles. Each tau occupied-pair block is fetched twice
  // per tile instead of twice for every virtual pair.
  const auto wabef_begin = SteadyClock::now();
  const auto virtual_pair_count = virtuals * virtuals;
  const auto owned_virtual_pairs = communicator.rank < virtual_pair_count
      ? 1 + (virtual_pair_count - 1 - communicator.rank) / communicator.size
      : 0;
  if (owned_virtual_pairs != 0 &&
      pair_count > std::numeric_limits<std::size_t>::max() /
                       owned_virtual_pairs)
    throw std::overflow_error("streamed W_abef contribution storage overflows");
  if (wabef_tile_max_bytes == 0)
    throw std::invalid_argument("CCSD W_abef tile memory must be nonzero");
  const auto bytes_per_wabef_block =
      checked_memory_product(block_size, sizeof(double));
  const auto tile_capacity = std::max<std::size_t>(
      1, wabef_tile_max_bytes /
             std::max<std::size_t>(1, bytes_per_wabef_block));
  std::vector<double> wabef_contributions(owned_virtual_pairs * pair_count);
  for (std::size_t tile_begin = 0; tile_begin < owned_virtual_pairs;
       tile_begin += tile_capacity) {
    const auto tile_count =
        std::min(tile_capacity, owned_virtual_pairs - tile_begin);
    std::vector<std::size_t> tasks(tile_count);
    for (std::size_t local = 0; local < tile_count; ++local)
      tasks[local] = communicator.rank +
                     (tile_begin + local) * communicator.size;

    std::vector<double> tau_ab_scalars(tile_count * pair_count);
    for (std::size_t pair = 0; pair < pair_count; ++pair) {
      const auto tau_block =
          tau.get_pair(pair / occupied, pair % occupied);
      for (std::size_t local = 0; local < tile_count; ++local)
        tau_ab_scalars[local * pair_count + pair] = tau_block[tasks[local]];
    }

    std::vector<Matrix> wabef_blocks;
    wabef_blocks.reserve(tile_count);
    for (std::size_t local = 0; local < tile_count; ++local)
      wabef_blocks.emplace_back(
          std::array<std::size_t, 2>{virtuals, virtuals});

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t local_index = 0;
         local_index < static_cast<std::ptrdiff_t>(tile_count);
         ++local_index) {
      const auto local = static_cast<std::size_t>(local_index);
      const auto a = tasks[local] / virtuals;
      const auto b = tasks[local] % virtuals;
      auto& wabef_block = wabef_blocks[local];
      for (std::size_t e = 0; e < virtuals; ++e)
        for (std::size_t q = 0; q < virtuals; ++q) {
          double value = eri(occupied + a, occupied + b, occupied + e,
                             occupied + q);
          for (std::size_t m = 0; m < occupied; ++m)
            value -=
                t1(m, b) *
                    eri(occupied + a, m, occupied + e, occupied + q) -
                t1(m, a) *
                    eri(occupied + b, m, occupied + e, occupied + q);
          for (std::size_t m = 0; m < occupied; ++m)
            for (std::size_t n = 0; n < occupied; ++n)
              value +=
                  0.25 *
                  tau_ab_scalars[local * pair_count + m * occupied + n] *
                  eri(m, n, occupied + e, occupied + q);
          wabef_block(e, q) = value;
        }
    }

    for (std::size_t pair = 0; pair < pair_count; ++pair) {
      const auto i = pair / occupied;
      const auto j = pair % occupied;
      const auto tau_block = tau.get_pair(i, j);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (std::ptrdiff_t local_index = 0;
           local_index < static_cast<std::ptrdiff_t>(tile_count);
           ++local_index) {
        const auto local = static_cast<std::size_t>(local_index);
        double contribution = 0.0;
        const auto& wabef_block = wabef_blocks[local];
        for (std::size_t element = 0; element < block_size; ++element)
          contribution += tau_block[element] * wabef_block.data()[element];
        const auto task = tasks[local];
        wabef_contributions[(tile_begin + local) * pair_count + pair] =
            0.5 * contribution /
            d2(i, j, task / virtuals, task % virtuals);
      }
    }
  }
  if (owned_virtual_pairs != 0) {
    std::vector<double> block(virtual_pair_count);
    for (std::size_t pair = 0; pair < pair_count; ++pair) {
      std::fill(block.begin(), block.end(), 0.0);
      for (std::size_t owned = 0; owned < owned_virtual_pairs; ++owned) {
        const auto task = communicator.rank + owned * communicator.size;
        block[task] = wabef_contributions[owned * pair_count + pair];
      }
      raw_t2.accumulate_pair(pair / occupied, pair % occupied, block);
    }
  }
  raw_t2.synchronize();
  antisymmetrize_distributed(raw_t2, next_t2);
  const auto wabef_end = SteadyClock::now();
  if (timing)
    timing->wabef_seconds += elapsed_seconds(wabef_begin, wabef_end);
  return next_t1;
}

void form_distributed_residual(
    const GlobalArraysPairTensor& candidate,
    const GlobalArraysPairTensor& current, const Tensor4& denominator,
    GlobalArraysPairTensor& raw_residual,
    GlobalArraysPairTensor& residual) {
  const auto& layout = current.layout();
  raw_residual.zero();
  const auto communicator = parallel::world();
  const auto virtuals = layout.virtuals();
  for (std::size_t pair = communicator.rank; pair < layout.pair_count();
       pair += communicator.size) {
    const auto [i, j] = layout.pair_from_flat(pair);
    auto block = candidate.get_pair(i, j);
    const auto old_block = current.get_pair(i, j);
    for (std::size_t a = 0; a < virtuals; ++a)
      for (std::size_t b = 0; b < virtuals; ++b)
        block[a * virtuals + b] =
            denominator(i, j, a, b) *
            (block[a * virtuals + b] - old_block[a * virtuals + b]);
    raw_residual.put_pair(i, j, block);
  }
  raw_residual.synchronize();
  antisymmetrize_distributed(raw_residual, residual);
}

void damp_distributed(GlobalArraysPairTensor& candidate,
                      const GlobalArraysPairTensor& current,
                      const double damping) {
  const auto& layout = current.layout();
  const auto communicator = parallel::world();
  for (std::size_t pair = communicator.rank; pair < layout.pair_count();
       pair += communicator.size) {
    const auto [i, j] = layout.pair_from_flat(pair);
    auto block = candidate.get_pair(i, j);
    const auto old_block = current.get_pair(i, j);
    for (std::size_t index = 0; index < block.size(); ++index)
      block[index] = old_block[index] +
                     damping * (block[index] - old_block[index]);
    candidate.put_pair(i, j, block);
  }
  candidate.synchronize();
}

template <typename Eri>
double distributed_ccsd_energy(const Matrix& fock, const Eri& eri,
                               const Matrix& t1,
                               const GlobalArraysPairTensor& t2,
                               const std::size_t occupied) {
  const auto communicator = parallel::world();
  const auto virtuals = t1.dimensions()[1];
  double local_energy = 0.0;
  if (communicator.rank == 0)
    for (std::size_t i = 0; i < occupied; ++i)
      for (std::size_t a = 0; a < virtuals; ++a)
        local_energy += fock(i, occupied + a) * t1(i, a);
  for (std::size_t pair = communicator.rank; pair < t2.layout().pair_count();
       pair += communicator.size) {
    const auto [i, j] = t2.layout().pair_from_flat(pair);
    const auto block = t2.get_pair(i, j);
    for (std::size_t a = 0; a < virtuals; ++a)
      for (std::size_t b = 0; b < virtuals; ++b)
        local_energy +=
            0.25 * eri(i, j, occupied + a, occupied + b) *
                block[a * virtuals + b] +
            0.5 * eri(i, j, occupied + a, occupied + b) * t1(i, a) *
                t1(j, b);
  }
  std::vector<double> reduced{local_energy};
  parallel::sum_in_place(reduced);
  return reduced.front();
}

std::size_t checked_memory_sum(const std::size_t left,
                               const std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error("CCSD distributed memory count overflows");
  return left + right;
}

std::size_t checked_memory_product(const std::size_t left,
                                   const std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error("CCSD distributed memory product overflows");
  return left * right;
}

template <typename Eri>
CcsdResult solve_ccsd_distributed(
    const Matrix& fock, const Eri& eri, const std::size_t occupied,
    const Matrix& d1, const Tensor4& d2, const CcsdOptions& options) {
  const auto virtuals = fock.dimensions()[0] - occupied;
  const OccupiedPairLayout layout(occupied, virtuals);
  const OccupiedPairLayout occupied_layout(occupied, occupied);
  GlobalArraysPairTensor current(layout, "ccsd_current_t2");
  GlobalArraysPairTensor raw(layout, "ccsd_raw_t2");
  GlobalArraysPairTensor candidate(layout, "ccsd_candidate_t2");
  GlobalArraysPairTensor residual(layout, "ccsd_residual_t2");
  GlobalArraysPairTensor tau(layout, "ccsd_tau");
  GlobalArraysPairTensor tilde_tau(layout, "ccsd_tilde_tau");
  GlobalArraysPairTensor wmbej(layout, "ccsd_wmbej");
  GlobalArraysPairTensor wmnij(occupied_layout, "ccsd_wmnij");
  Matrix t1({occupied, virtuals});

  if (options.initial_t1.has_value() != options.initial_t2.has_value())
    throw std::invalid_argument(
        "CCSD restart requires both T1 and T2 amplitudes");
  if (options.initial_t1) {
    if (options.initial_t1->dimensions() !=
            std::array<std::size_t, 2>{occupied, virtuals} ||
        options.initial_t2->dimensions() !=
            std::array<std::size_t, 4>{occupied, occupied, virtuals,
                                       virtuals})
      throw std::invalid_argument("CCSD restart amplitude dimensions mismatch");
    t1 = *options.initial_t1;
    for (const auto value : t1.data())
      if (!std::isfinite(value))
        throw std::invalid_argument("non-finite restarted T1 amplitude");
    for (const auto value : options.initial_t2->data())
      if (!std::isfinite(value))
        throw std::invalid_argument("non-finite restarted T2 amplitude");
    if (t2_antisymmetry_error(*options.initial_t2) > 1.0e-10)
      throw std::invalid_argument("restarted T2 amplitudes are not antisymmetric");
    current.load_replicated(*options.initial_t2);
  } else {
    const auto communicator = parallel::world();
    for (std::size_t pair = communicator.rank; pair < layout.pair_count();
         pair += communicator.size) {
      const auto [i, j] = layout.pair_from_flat(pair);
      std::vector<double> block(virtuals * virtuals);
      for (std::size_t a = 0; a < virtuals; ++a)
        for (std::size_t b = 0; b < virtuals; ++b)
          block[a * virtuals + b] =
              eri(i, j, occupied + a, occupied + b) / d2(i, j, a, b);
      raw.put_pair(i, j, block);
    }
    raw.synchronize();
    antisymmetrize_distributed(raw, current);
  }

  GlobalArraysPairDiis diis(options.diis_size, layout, "ccsd_diis");
  CcsdResult result{};
  result.amplitude_backend = CcsdAmplitudeBackend::global_arrays;
  result.wabef_backend = CcsdWabefBackend::cpu;
  result.history = options.initial_history;
  if (!result.history.empty() &&
      result.history.back().iteration != options.completed_iterations)
    throw std::invalid_argument("CCSD restart history/iteration mismatch");
  double old_energy =
      distributed_ccsd_energy(fock, eri, t1, current, occupied);
  for (std::size_t iteration = options.completed_iterations + 1;
       iteration <= options.max_iterations; ++iteration) {
    CcsdIterationTiming iteration_timing;
    const auto iteration_begin = SteadyClock::now();
    auto next_t1 = fixed_point_distributed(
        fock, eri, t1, current, occupied, d1, d2, raw, candidate, tau,
        tilde_tau, wmbej, wmnij, &iteration_timing,
        options.wabef_tile_max_bytes);
    const auto residual_begin = SteadyClock::now();
    Matrix r1({occupied, virtuals});
    for (std::size_t i = 0; i < occupied; ++i)
      for (std::size_t a = 0; a < virtuals; ++a)
        r1(i, a) = d1(i, a) * (next_t1(i, a) - t1(i, a));
    form_distributed_residual(candidate, current, d2, raw, residual);
    result.r1_norm = norm(r1.data());
    result.r2_norm = std::sqrt(std::max(0.0, residual.dot(residual)));
    const auto r1_convergence_measure = ccsd_residual_measure(
        result.r1_norm, r1.size(), options.residual_metric);
    const auto r2_convergence_measure = ccsd_residual_measure(
        result.r2_norm, d2.size(), options.residual_metric);
    const auto residual_end = SteadyClock::now();
    iteration_timing.residual_seconds +=
        elapsed_seconds(residual_begin, residual_end);
    if (!result.history.empty() &&
        std::abs(result.history.back().delta_energy) <
            options.energy_tolerance &&
        std::max(r1_convergence_measure, r2_convergence_measure) <
            options.residual_tolerance) {
      result.converged = true;
      result.history.back().r1_norm = result.r1_norm;
      result.history.back().r2_norm = result.r2_norm;
      iteration_timing.iteration_seconds =
          elapsed_seconds(iteration_begin, SteadyClock::now());
      add_timing(result.cumulative_timing, iteration_timing);
      if (options.iteration_observer)
        options.iteration_observer(CcsdIterationReport{
            result.history.back().iteration, old_energy,
            result.history.back().delta_energy,
            result.r1_norm, result.r2_norm, r1_convergence_measure,
            r2_convergence_measure, true,
            CcsdAmplitudeBackend::global_arrays, CcsdWabefBackend::cpu,
            iteration_timing});
      if (options.observer) {
        const auto checkpoint_t2 = current.replicate();
        options.observer(CcsdProgress{
            result.history.back().iteration, old_energy, result.r1_norm,
            result.r2_norm, true, t1, checkpoint_t2, result.history,
            CcsdAmplitudeBackend::global_arrays, CcsdWabefBackend::cpu,
            iteration_timing});
      }
      break;
    }
    for (std::size_t index = 0; index < t1.size(); ++index)
      next_t1.data()[index] =
          t1.data()[index] +
          options.damping * (next_t1.data()[index] - t1.data()[index]);
    damp_distributed(candidate, current, options.damping);
    const auto diis_begin = SteadyClock::now();
    if (iteration >= options.diis_start) {
      next_t1 = diis.extrapolate_distributed(next_t1, candidate, r1,
                                             residual, raw);
      antisymmetrize_distributed(raw, candidate);
    }
    const auto diis_end = SteadyClock::now();
    iteration_timing.diis_seconds += elapsed_seconds(diis_begin, diis_end);
    const auto energy_begin = SteadyClock::now();
    const auto new_energy =
        distributed_ccsd_energy(fock, eri, next_t1, candidate, occupied);
    const auto energy_end = SteadyClock::now();
    iteration_timing.energy_seconds +=
        elapsed_seconds(energy_begin, energy_end);
    result.history.push_back({iteration, new_energy, new_energy - old_energy,
                              result.r1_norm, result.r2_norm});
    t1 = std::move(next_t1);
    current.copy_from(candidate);
    old_energy = new_energy;
    iteration_timing.iteration_seconds =
        elapsed_seconds(iteration_begin, SteadyClock::now());
    add_timing(result.cumulative_timing, iteration_timing);
    if (options.iteration_observer)
      options.iteration_observer(CcsdIterationReport{
          iteration, old_energy, result.history.back().delta_energy,
          result.r1_norm, result.r2_norm, r1_convergence_measure,
          r2_convergence_measure, result.converged,
          CcsdAmplitudeBackend::global_arrays, CcsdWabefBackend::cpu,
          iteration_timing});
    if (options.observer &&
        (result.converged || options.checkpoint_interval == 0 ||
         iteration % options.checkpoint_interval == 0)) {
      const auto checkpoint_t2 = current.replicate();
      options.observer(CcsdProgress{iteration, old_energy, result.r1_norm,
                                    result.r2_norm, result.converged, t1,
                                    checkpoint_t2, result.history,
                                    CcsdAmplitudeBackend::global_arrays,
                                    CcsdWabefBackend::cpu, iteration_timing});
    }
    if (result.converged)
      break;
  }
  result.correlation_energy = old_energy;
  const auto diis_memory = diis.memory();
  result.diis_memory = {true,
                        diis_memory.history_entries,
                        diis_memory.doubles_history_global_bytes,
                        diis_memory.doubles_history_local_bytes,
                        diis_memory.doubles_history_replicated_bytes_per_rank,
                        diis_memory.singles_history_bytes_per_rank};
  const std::array<DistributedTensorMemory, 4> memories{
      current.memory(), raw.memory(), candidate.memory(), residual.memory()};
  CcsdDistributedMemory distributed_memory;
  distributed_memory.global_arrays = true;
  distributed_memory.solver_pair_arrays = memories.size();
  for (const auto& memory : memories) {
    distributed_memory.solver_pair_global_bytes = checked_memory_sum(
        distributed_memory.solver_pair_global_bytes, memory.global_bytes);
    distributed_memory.solver_pair_local_bytes = checked_memory_sum(
        distributed_memory.solver_pair_local_bytes, memory.local_bytes);
    distributed_memory.solver_pair_replicated_bytes_per_rank =
        checked_memory_sum(
            distributed_memory.solver_pair_replicated_bytes_per_rank,
            memory.replicated_bytes);
  }
  const std::array<DistributedTensorMemory, 3> intermediate_memories{
      tau.memory(), tilde_tau.memory(), wmbej.memory()};
  distributed_memory.intermediate_pair_arrays =
      intermediate_memories.size();
  for (const auto& memory : intermediate_memories) {
    distributed_memory.intermediate_pair_global_bytes = checked_memory_sum(
        distributed_memory.intermediate_pair_global_bytes,
        memory.global_bytes);
    distributed_memory.intermediate_pair_local_bytes = checked_memory_sum(
        distributed_memory.intermediate_pair_local_bytes,
        memory.local_bytes);
    distributed_memory.intermediate_pair_replicated_bytes_per_rank =
        checked_memory_sum(
            distributed_memory.intermediate_pair_replicated_bytes_per_rank,
            memory.replicated_bytes);
  }
  const auto occupied_intermediate_memory = wmnij.memory();
  distributed_memory.occupied_intermediate_pair_arrays = 1;
  distributed_memory.occupied_intermediate_pair_global_bytes =
      occupied_intermediate_memory.global_bytes;
  distributed_memory.occupied_intermediate_pair_local_bytes =
      occupied_intermediate_memory.local_bytes;
  distributed_memory.occupied_intermediate_pair_replicated_bytes_per_rank =
      occupied_intermediate_memory.replicated_bytes;
  const auto cached_pair_blocks =
      checked_memory_sum(checked_memory_product(4, occupied), 2);
  distributed_memory.bounded_pair_cache_bytes_per_rank =
      checked_memory_sum(
          checked_memory_product(
              cached_pair_blocks,
              checked_memory_product(
                  checked_memory_product(virtuals, virtuals),
                  sizeof(double))),
          checked_memory_product(
              checked_memory_product(occupied, occupied), sizeof(double)));
  auto small_intermediate_elements =
      checked_memory_sum(checked_memory_product(virtuals, virtuals),
                         checked_memory_product(occupied, occupied));
  small_intermediate_elements =
      checked_memory_sum(small_intermediate_elements,
                         checked_memory_product(occupied, virtuals));
  const auto occupied_squared =
      checked_memory_product(occupied, occupied);
  distributed_memory.replicated_small_intermediate_bytes_per_rank =
      checked_memory_product(small_intermediate_elements, sizeof(double));
  const auto communicator = parallel::world();
  const auto virtual_pair_count =
      checked_memory_product(virtuals, virtuals);
  const auto owned_virtual_pairs =
      communicator.rank < virtual_pair_count
          ? 1 + (virtual_pair_count - 1 - communicator.rank) /
                    communicator.size
          : 0;
  distributed_memory.streamed_wabef_accumulator_bytes_per_rank =
      checked_memory_product(
          checked_memory_product(owned_virtual_pairs, occupied_squared),
          sizeof(double));
  result.distributed_memory = distributed_memory;
  result.t1 = std::move(t1);
  result.t2 = current.replicate();
  return result;
}

} // namespace

std::string_view ccsd_amplitude_backend_name(
    const CcsdAmplitudeBackend backend) {
  switch (backend) {
  case CcsdAmplitudeBackend::automatic:
    return "automatic";
  case CcsdAmplitudeBackend::replicated:
    return "replicated exact";
  case CcsdAmplitudeBackend::global_arrays:
    return "Global Arrays distributed";
  }
  throw std::invalid_argument("invalid CCSD amplitude backend");
}

std::string_view ccsd_residual_metric_name(
    const CcsdResidualMetric metric) {
  switch (metric) {
  case CcsdResidualMetric::l2:
    return "unnormalized L2";
  case CcsdResidualMetric::rms:
    return "RMS over stored amplitude elements";
  }
  throw std::invalid_argument("invalid CCSD residual metric");
}

double ccsd_residual_measure(const double l2_norm,
                             const std::size_t element_count,
                             const CcsdResidualMetric metric) {
  if (!std::isfinite(l2_norm) || l2_norm < 0.0 || element_count == 0)
    throw std::invalid_argument(
        "CCSD residual measure requires a finite nonnegative norm and "
        "positive element count");
  if (metric == CcsdResidualMetric::l2) return l2_norm;
  if (metric == CcsdResidualMetric::rms)
    return l2_norm / std::sqrt(static_cast<double>(element_count));
  throw std::invalid_argument("invalid CCSD residual metric");
}

std::string_view ccsd_wabef_backend_name(const CcsdWabefBackend backend) {
  switch (backend) {
  case CcsdWabefBackend::cpu:
    return "CPU";
  case CcsdWabefBackend::tiled_cpu:
    return "tiled CPU BLAS";
  case CcsdWabefBackend::cuda:
    return "CUDA contraction tiles";
  }
  throw std::invalid_argument("invalid CCSD W_abef backend");
}

std::string_view ccsd_wabef_packing_backend_name(
    const CcsdWabefPackingBackend backend) {
  switch (backend) {
  case CcsdWabefPackingBackend::direct:
    return "direct ERI access";
  case CcsdWabefPackingBackend::cached_invariants:
    return "cached exact invariants";
  }
  throw std::invalid_argument("invalid CCSD W_abef packing backend");
}

template <typename Eri>
double ccsd_energy_impl(const Matrix& fock, const Eri& eri, const Matrix& t1,
                        const Tensor4& t2, const std::size_t occupied) {
  const auto virtuals = t1.dimensions()[1];
  double energy = 0.0;
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t a = 0; a < virtuals; ++a)
      energy += fock(i, occupied + a) * t1(i, a);
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t j = 0; j < occupied; ++j)
      for (std::size_t a = 0; a < virtuals; ++a)
        for (std::size_t b = 0; b < virtuals; ++b)
          energy += 0.25 * eri(i, j, occupied + a, occupied + b) *
                        t2(i, j, a, b) +
                    0.5 * eri(i, j, occupied + a, occupied + b) * t1(i, a) *
                        t1(j, b);
  return energy;
}

template <typename Eri>
double distributed_openmp_ccsd_energy_impl(
    const Matrix& fock, const Eri& eri, const Matrix& t1,
    const Tensor4& t2, const std::size_t occupied) {
  const auto communicator = parallel::world();
  const auto virtuals = t1.dimensions()[1];
  const auto singles_count = checked_memory_product(occupied, virtuals);
  const auto doubles_count = checked_memory_product(
      checked_memory_product(occupied, occupied),
      checked_memory_product(virtuals, virtuals));
  const auto term_count = checked_memory_sum(singles_count, doubles_count);
  const auto owned_count = communicator.rank < term_count
      ? 1 + (term_count - 1 - communicator.rank) / communicator.size
      : 0;
  if (owned_count >
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::overflow_error(
        "distributed CCSD energy extent exceeds ptrdiff_t");
  double local_energy = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : local_energy)
#endif
  for (std::ptrdiff_t owned = 0;
       owned < static_cast<std::ptrdiff_t>(owned_count); ++owned) {
    const auto term = communicator.rank +
                      static_cast<std::size_t>(owned) * communicator.size;
    if (term < singles_count) {
      const auto i = term / virtuals;
      const auto a = term % virtuals;
      local_energy += fock(i, occupied + a) * t1(i, a);
      continue;
    }
    auto index = term - singles_count;
    const auto b = index % virtuals;
    index /= virtuals;
    const auto a = index % virtuals;
    index /= virtuals;
    const auto j = index % occupied;
    const auto i = index / occupied;
    const auto integral = eri(i, j, occupied + a, occupied + b);
    local_energy += 0.25 * integral * t2(i, j, a, b) +
                    0.5 * integral * t1(i, a) * t1(j, b);
  }
  std::vector<double> reduced{local_energy};
  parallel::sum_in_place(reduced);
  return reduced.front();
}

double t2_antisymmetry_error(const Tensor4& t2) {
  const auto dimensions = t2.dimensions();
  double error = 0.0;
  for (std::size_t i = 0; i < dimensions[0]; ++i)
    for (std::size_t j = 0; j < dimensions[1]; ++j)
      for (std::size_t a = 0; a < dimensions[2]; ++a)
        for (std::size_t b = 0; b < dimensions[3]; ++b)
          error = std::max({error, std::abs(t2(i, j, a, b) + t2(j, i, a, b)),
                            std::abs(t2(i, j, a, b) + t2(i, j, b, a))});
  return error;
}

CcsdProjectionMask make_closed_shell_spin_projection_mask(
    const std::size_t spatial_occupied,
    const std::size_t spatial_virtuals) {
  if (spatial_occupied == 0 || spatial_virtuals == 0 ||
      spatial_occupied > std::numeric_limits<std::size_t>::max() / 2 ||
      spatial_virtuals > std::numeric_limits<std::size_t>::max() / 2)
    throw std::invalid_argument(
        "closed-shell spin projection requires nonzero spatial dimensions");
  const auto occupied = 2 * spatial_occupied;
  const auto virtuals = 2 * spatial_virtuals;
  const auto singles_count = checked_memory_product(occupied, virtuals);
  const auto doubles_count =
      checked_memory_product(singles_count, singles_count);
  CcsdProjectionMask mask;
  mask.singles.assign(singles_count, 0U);
  mask.doubles.assign(doubles_count, 0U);
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t a = 0; a < virtuals; ++a) {
      const auto flat = i * virtuals + a;
      if (i % 2 == a % 2) {
        mask.singles[flat] = 1U;
        ++mask.selected_singles;
      }
    }
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t j = 0; j < occupied; ++j)
      for (std::size_t a = 0; a < virtuals; ++a)
        for (std::size_t b = 0; b < virtuals; ++b) {
          const auto hole_alpha_count = (i % 2 == 0 ? 1U : 0U) +
                                        (j % 2 == 0 ? 1U : 0U);
          const auto particle_alpha_count = (a % 2 == 0 ? 1U : 0U) +
                                            (b % 2 == 0 ? 1U : 0U);
          if (hole_alpha_count != particle_alpha_count)
            continue;
          const auto flat =
              ((i * occupied + j) * virtuals + a) * virtuals + b;
          mask.doubles[flat] = 1U;
          ++mask.selected_doubles;
        }
  validate_projection_mask(mask, occupied, virtuals);
  return mask;
}

CcsdProjectionMask intersect_ccsd_projection_masks(
    const CcsdProjectionMask& first, const CcsdProjectionMask& second) {
  if (first.singles.size() != second.singles.size() ||
      first.doubles.size() != second.doubles.size())
    throw std::invalid_argument(
        "cannot intersect CCSD projection masks with different dimensions");
  const auto valid_mask = [](const CcsdProjectionMask& mask) {
    const auto binary = [](const unsigned char value) { return value <= 1U; };
    return std::all_of(mask.singles.begin(), mask.singles.end(), binary) &&
           std::all_of(mask.doubles.begin(), mask.doubles.end(), binary) &&
           static_cast<std::size_t>(
               std::count(mask.singles.begin(), mask.singles.end(), 1U)) ==
               mask.selected_singles &&
           static_cast<std::size_t>(
               std::count(mask.doubles.begin(), mask.doubles.end(), 1U)) ==
               mask.selected_doubles;
  };
  if (!valid_mask(first) || !valid_mask(second))
    throw std::invalid_argument(
        "cannot intersect an invalid CCSD projection mask");
  CcsdProjectionMask result;
  result.singles.resize(first.singles.size());
  result.doubles.resize(first.doubles.size());
  for (std::size_t index = 0; index < result.singles.size(); ++index) {
    const auto value = static_cast<unsigned char>(
        first.singles[index] != 0U && second.singles[index] != 0U);
    result.singles[index] = value;
    result.selected_singles += value;
  }
  for (std::size_t index = 0; index < result.doubles.size(); ++index) {
    const auto value = static_cast<unsigned char>(
        first.doubles[index] != 0U && second.doubles[index] != 0U);
    result.doubles[index] = value;
    result.selected_doubles += value;
  }
  return result;
}

template <typename Eri>
CcsdResult solve_ccsd_impl(const Matrix& fock, const Eri& eri,
                           const std::size_t occupied,
                           const CcsdOptions& options) {
  if (options.spatial_diis_metric != SpatialDiisMetric::raw)
    throw std::invalid_argument("weighted spatial DIIS requires the spatial solver");
  validate_wabef_pair_symmetry_options(options);
  if (options.audit_compact_spatial_doubles || options.use_compact_spatial_doubles)
    throw std::invalid_argument("compact spatial doubles requires the spatial solver");
  if (options.pm_symmetry)
    throw std::invalid_argument("PM symmetry admission requires the spatial CCSD solver");
  const auto orbitals = fock.dimensions()[0];
  if (fock.dimensions()[1] != orbitals ||
      eri.dimensions() !=
          std::array<std::size_t, 4>{orbitals, orbitals, orbitals, orbitals} ||
      occupied == 0 || occupied >= orbitals)
    throw std::invalid_argument("inconsistent CCSD dimensions");
  const auto virtuals = orbitals - occupied;
  if (options.projection_mask)
    validate_projection_mask(*options.projection_mask, occupied, virtuals);
  Matrix d1({occupied, virtuals});
  Tensor4 d2({occupied, occupied, virtuals, virtuals});
  if (options.initial_t1.has_value() != options.initial_t2.has_value())
    throw std::invalid_argument(
        "CCSD restart requires both T1 and T2 amplitudes");
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t a = 0; a < virtuals; ++a) {
      d1(i, a) = fock(i, i) - fock(occupied + a, occupied + a);
      if (std::abs(d1(i, a)) < options.denominator_floor)
        throw std::runtime_error("small CCSD preconditioner");
    }
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t j = 0; j < occupied; ++j)
      for (std::size_t a = 0; a < virtuals; ++a)
        for (std::size_t b = 0; b < virtuals; ++b) {
          d2(i, j, a, b) = d1(i, a) + d1(j, b);
          if (std::abs(d2(i, j, a, b)) < options.denominator_floor)
            throw std::runtime_error("small CCSD preconditioner");
        }
  const auto t2_elements = checked_memory_product(
      checked_memory_product(occupied, occupied),
      checked_memory_product(virtuals, virtuals));
  const auto t2_bytes = checked_memory_product(t2_elements, sizeof(double));
  const auto ga_active = global_arrays_runtime_active();
  bool use_distributed_amplitudes = false;
  switch (options.amplitude_backend) {
  case CcsdAmplitudeBackend::automatic:
    use_distributed_amplitudes =
        ga_active && options.use_global_arrays_diis &&
        t2_bytes > options.replicated_t2_max_bytes;
    break;
  case CcsdAmplitudeBackend::replicated:
    use_distributed_amplitudes = false;
    break;
  case CcsdAmplitudeBackend::global_arrays:
    if (!ga_active || !options.use_global_arrays_diis)
      throw std::runtime_error(
          "explicit Global Arrays CCSD amplitudes require an active GA "
          "runtime and distributed DIIS");
    use_distributed_amplitudes = true;
    break;
  }
  if (use_distributed_amplitudes)
    if (options.projection_mask)
      throw std::runtime_error(
          "the selected CCSD projection oracle currently requires the "
          "replicated backend; the block-sparse Global Arrays executor has "
          "not passed equivalence validation");
  if (use_distributed_amplitudes)
    if (options.wabef_backend == CcsdWabefBackend::cuda)
      throw std::runtime_error(
          "CUDA CCSD W_abef contraction currently requires the exact "
          "replicated small-system amplitude backend");
  if (use_distributed_amplitudes)
    if (options.wabef_backend == CcsdWabefBackend::tiled_cpu)
      throw std::runtime_error(
          "tiled CPU CCSD W_abef currently requires the exact replicated "
          "amplitude backend; distributed tensor tiles have not passed "
          "equivalence validation");
  if (use_distributed_amplitudes &&
      options.wabef_packing_backend ==
          CcsdWabefPackingBackend::cached_invariants)
    throw std::runtime_error(
        "cached CCSD W_abef invariant packing currently requires the exact "
        "replicated amplitude backend");
  if (use_distributed_amplitudes && options.use_tiled_doubles)
    throw std::runtime_error(
        "tiled CPU CCSD doubles currently require the exact replicated "
        "amplitude backend; distributed tensor tiles have not passed "
        "equivalence validation");
  if (use_distributed_amplitudes && options.use_tiled_intermediates)
    throw std::runtime_error(
        "tiled CPU CCSD intermediates currently require the exact "
        "replicated amplitude backend; distributed tensor tiles have not "
        "passed equivalence validation");
  if (use_distributed_amplitudes)
    return solve_ccsd_distributed(fock, eri, occupied, d1, d2, options);

  Matrix t1({occupied, virtuals});
  Tensor4 t2({occupied, occupied, virtuals, virtuals});
  if (!options.initial_t2)
    for (std::size_t i = 0; i < occupied; ++i)
      for (std::size_t j = 0; j < occupied; ++j)
        for (std::size_t a = 0; a < virtuals; ++a)
          for (std::size_t b = 0; b < virtuals; ++b)
            t2(i, j, a, b) =
                eri(i, j, occupied + a, occupied + b) / d2(i, j, a, b);
  if (!options.initial_t2 && options.projection_mask)
    apply_projection_mask(*options.projection_mask, t1, t2);
  if (options.initial_t1) {
    if (options.initial_t1->dimensions() !=
            std::array<std::size_t, 2>{occupied, virtuals} ||
        options.initial_t2->dimensions() !=
            std::array<std::size_t, 4>{occupied, occupied, virtuals,
                                       virtuals})
      throw std::invalid_argument("CCSD restart amplitude dimensions mismatch");
    t1 = *options.initial_t1;
    t2 = *options.initial_t2;
    for (const auto value : t1.data())
      if (!std::isfinite(value))
        throw std::invalid_argument("non-finite restarted T1 amplitude");
    for (const auto value : t2.data())
      if (!std::isfinite(value))
        throw std::invalid_argument("non-finite restarted T2 amplitude");
    if (t2_antisymmetry_error(t2) > 1.0e-10)
      throw std::invalid_argument("restarted T2 amplitudes are not antisymmetric");
    if (options.projection_mask &&
        projection_excluded_maximum(*options.projection_mask, t1, t2) >
            1.0e-14)
      throw std::invalid_argument(
          "restarted CCSD amplitudes violate the selected projection mask");
  } else {
    t2 = antisym(t2);
    if (options.projection_mask)
      apply_projection_mask(*options.projection_mask, t1, t2);
  }

  const auto* contraction_mask =
      options.projection_mask &&
              options.projection_mask->selected_doubles != t2.size()
          ? options.projection_mask.get()
          : nullptr;
  if (options.wabef_packing_backend ==
          CcsdWabefPackingBackend::cached_invariants &&
      options.wabef_backend != CcsdWabefBackend::tiled_cpu)
    throw std::invalid_argument(
        "cached CCSD W_abef invariant packing requires tiled-cpu W_abef");
  std::optional<TiledWabefInvariantCache> wabef_invariant_cache;
  double wabef_invariant_cache_build_seconds = 0.0;
  if (options.wabef_packing_backend ==
      CcsdWabefPackingBackend::cached_invariants) {
    const auto cache_begin = SteadyClock::now();
    wabef_invariant_cache.emplace(build_tiled_wabef_invariant_cache(
        eri, occupied, virtuals, contraction_mask,
        options.wabef_invariant_cache_max_bytes,
        options.use_wabef_pair_symmetry));
    wabef_invariant_cache_build_seconds =
        elapsed_seconds(cache_begin, SteadyClock::now());
  }
  std::optional<SelectedIntegralPanelCache> selected_eri_panel_cache;
  double selected_eri_panel_cache_build_seconds = 0.0;
  if constexpr (std::is_same_v<std::remove_cvref_t<Eri>,
                               SelectedIntegralStore>) {
    if (options.use_tiled_intermediates || options.use_tiled_doubles) {
      const auto cache_begin = SteadyClock::now();
      selected_eri_panel_cache.emplace(
          eri, options.selected_eri_panel_cache_max_bytes);
      selected_eri_panel_cache_build_seconds =
          elapsed_seconds(cache_begin, SteadyClock::now());
    }
  }

  Diis diis(options.diis_size,
            options.use_incremental_diis_gram
                ? DiisGramBackend::incremental
                : DiisGramBackend::recomputed);
  CcsdResult result{};
  result.amplitude_backend = CcsdAmplitudeBackend::replicated;
  result.wabef_backend = options.wabef_backend;
  result.wabef_packing_backend = options.wabef_packing_backend;
  result.wabef_invariant_cache_bytes = wabef_invariant_cache
      ? wabef_invariant_cache->allocated_bytes
      : 0;
  result.wabef_invariant_cache_build_seconds =
      wabef_invariant_cache_build_seconds;
  result.selected_eri_panel_cache_enabled =
      selected_eri_panel_cache && selected_eri_panel_cache->enabled();
  result.selected_eri_panel_cache_required_bytes = selected_eri_panel_cache
      ? selected_eri_panel_cache->required_bytes()
      : 0;
  result.selected_eri_panel_cache_bytes = selected_eri_panel_cache
      ? selected_eri_panel_cache->allocated_bytes()
      : 0;
  result.selected_eri_panel_cache_populated_elements =
      selected_eri_panel_cache
          ? selected_eri_panel_cache->populated_elements()
          : 0;
  result.selected_eri_panel_cache_build_seconds =
      selected_eri_panel_cache_build_seconds;
  result.history = options.initial_history;
  if (!result.history.empty() &&
      result.history.back().iteration != options.completed_iterations)
    throw std::invalid_argument("CCSD restart history/iteration mismatch");
  const auto evaluate_energy = [&](const Matrix& amplitudes_t1,
                                   const Tensor4& amplitudes_t2) {
    return options.use_distributed_replicated_energy
        ? distributed_openmp_ccsd_energy_impl(
              fock, eri, amplitudes_t1, amplitudes_t2, occupied)
        : ccsd_energy_impl(fock, eri, amplitudes_t1, amplitudes_t2,
                           occupied);
  };
  double old_energy = evaluate_energy(t1, t2);
  for (std::size_t iteration = options.completed_iterations + 1;
       iteration <= options.max_iterations;
       ++iteration) {
    CcsdIterationTiming iteration_timing;
    const auto iteration_begin = SteadyClock::now();
    // Ordinary references avoid an AppleClang OpenMP frontend crash when
    // structured-binding names are captured by the residual/damping loops.
    auto next_amplitudes =
        fixed_point(fock, eri, t1, t2, occupied, d1, d2,
                    &iteration_timing, options.wabef_backend,
                    options.use_tiled_intermediates,
                    options.use_tiled_doubles,
                    options.wabef_tile_max_bytes, contraction_mask,
                    wabef_invariant_cache ? &*wabef_invariant_cache : nullptr,
                    selected_eri_panel_cache &&
                            selected_eri_panel_cache->enabled()
                        ? &*selected_eri_panel_cache
                        : nullptr);
    auto& next_t1 = next_amplitudes.first;
    auto& next_t2 = next_amplitudes.second;
    if (options.projection_mask)
      apply_projection_mask(*options.projection_mask, next_t1, next_t2);
    const auto residual_begin = SteadyClock::now();
    Matrix r1({occupied, virtuals});
    Tensor4 r2({occupied, occupied, virtuals, virtuals});
    if (r2.size() >
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
      throw std::overflow_error(
          "replicated CCSD residual extent exceeds ptrdiff_t");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(r1.size() >= 16384)
#endif
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(r1.size()); ++index) {
      const auto element = static_cast<std::size_t>(index);
      r1.data()[element] =
          d1.data()[element] *
          (next_t1.data()[element] - t1.data()[element]);
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(r2.size() >= 16384)
#endif
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(r2.size()); ++index) {
      const auto element = static_cast<std::size_t>(index);
      r2.data()[element] =
          d2.data()[element] *
          (next_t2.data()[element] - t2.data()[element]);
    }
    r2 = antisym(r2);
    result.r1_norm = norm(r1.data());
    result.r2_norm = norm(r2.data());
    const auto r1_convergence_measure = ccsd_residual_measure(
        result.r1_norm, r1.size(), options.residual_metric);
    const auto r2_convergence_measure = ccsd_residual_measure(
        result.r2_norm, r2.size(), options.residual_metric);
    const auto residual_end = SteadyClock::now();
    iteration_timing.residual_seconds +=
        elapsed_seconds(residual_begin, residual_end);
    if (!result.history.empty() &&
        std::abs(result.history.back().delta_energy) <
            options.energy_tolerance &&
        std::max(r1_convergence_measure, r2_convergence_measure) <
            options.residual_tolerance) {
      result.converged = true;
      result.history.back().r1_norm = result.r1_norm;
      result.history.back().r2_norm = result.r2_norm;
      iteration_timing.iteration_seconds =
          elapsed_seconds(iteration_begin, SteadyClock::now());
      add_timing(result.cumulative_timing, iteration_timing);
      if (options.iteration_observer)
        options.iteration_observer(CcsdIterationReport{
            result.history.back().iteration, old_energy,
            result.history.back().delta_energy,
            result.r1_norm, result.r2_norm, r1_convergence_measure,
            r2_convergence_measure, true,
            CcsdAmplitudeBackend::replicated, options.wabef_backend,
            iteration_timing});
      if (options.observer)
        options.observer(CcsdProgress{
            result.history.back().iteration, old_energy, result.r1_norm,
            result.r2_norm, true, t1, t2, result.history,
            CcsdAmplitudeBackend::replicated, options.wabef_backend,
            iteration_timing});
      break;
    }
    if (t2.size() >
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
      throw std::overflow_error(
          "replicated CCSD damping extent exceeds ptrdiff_t");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(t1.size() >= 16384)
#endif
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(t1.size()); ++index) {
      const auto element = static_cast<std::size_t>(index);
      next_t1.data()[element] =
          t1.data()[element] +
          options.damping *
              (next_t1.data()[element] - t1.data()[element]);
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(t2.size() >= 16384)
#endif
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(t2.size()); ++index) {
      const auto element = static_cast<std::size_t>(index);
      next_t2.data()[element] =
          t2.data()[element] +
          options.damping *
              (next_t2.data()[element] - t2.data()[element]);
    }
    const auto diis_begin = SteadyClock::now();
    if (iteration >= options.diis_start) {
      const auto communicator = parallel::world();
      const auto vector_size = checked_memory_sum(t1.size(), t2.size());
      if (vector_size >
          static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        throw std::overflow_error(
            "replicated CCSD DIIS extent exceeds ptrdiff_t");
      std::vector<double> vector(vector_size);
      if (!options.use_root_replicated_diis || communicator.rank == 0) {
        std::vector<double> error(vector_size);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(vector_size >= 16384)
#endif
        for (std::ptrdiff_t index = 0;
             index < static_cast<std::ptrdiff_t>(vector_size); ++index) {
          const auto element = static_cast<std::size_t>(index);
          if (element < t1.size()) {
            vector[element] = next_t1.data()[element];
            error[element] = r1.data()[element];
          } else {
            const auto t2_element = element - t1.size();
            vector[element] = next_t2.data()[t2_element];
            error[element] = r2.data()[t2_element];
          }
        }
        vector = diis.extrapolate(vector, error);
        const auto diis_work = diis.last_work();
        iteration_timing.diis_error_dot_products +=
            diis_work.error_dot_products;
        iteration_timing.diis_history_vector_accumulations +=
            diis_work.history_vector_accumulations;
      }
      if (options.use_root_replicated_diis)
        parallel::broadcast_in_place(vector);
      if (vector.size() != vector_size)
        throw std::runtime_error(
            "replicated CCSD DIIS output dimensions are inconsistent");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(vector_size >= 16384)
#endif
      for (std::ptrdiff_t index = 0;
           index < static_cast<std::ptrdiff_t>(vector_size); ++index) {
        const auto element = static_cast<std::size_t>(index);
        if (element < t1.size())
          next_t1.data()[element] = vector[element];
        else
          next_t2.data()[element - t1.size()] = vector[element];
      }
      next_t2 = antisym(next_t2);
      if (options.projection_mask)
        apply_projection_mask(*options.projection_mask, next_t1, next_t2);
    }
    const auto diis_end = SteadyClock::now();
    iteration_timing.diis_seconds += elapsed_seconds(diis_begin, diis_end);
    const auto energy_begin = SteadyClock::now();
    const auto new_energy = evaluate_energy(next_t1, next_t2);
    const auto energy_end = SteadyClock::now();
    iteration_timing.energy_seconds +=
        elapsed_seconds(energy_begin, energy_end);
    result.history.push_back({iteration, new_energy, new_energy - old_energy,
                              result.r1_norm, result.r2_norm});
    t1 = std::move(next_t1);
    t2 = std::move(next_t2);
    old_energy = new_energy;
    iteration_timing.iteration_seconds =
        elapsed_seconds(iteration_begin, SteadyClock::now());
    add_timing(result.cumulative_timing, iteration_timing);
    if (options.iteration_observer)
      options.iteration_observer(CcsdIterationReport{
          iteration, old_energy, result.history.back().delta_energy,
          result.r1_norm, result.r2_norm, r1_convergence_measure,
          r2_convergence_measure, result.converged,
          CcsdAmplitudeBackend::replicated, options.wabef_backend,
          iteration_timing});
    if (options.observer &&
        (result.converged || options.checkpoint_interval == 0 ||
         iteration % options.checkpoint_interval == 0))
      options.observer(CcsdProgress{iteration, old_energy, result.r1_norm,
                                    result.r2_norm, result.converged, t1, t2,
                                    result.history,
                                    CcsdAmplitudeBackend::replicated,
                                    options.wabef_backend, iteration_timing});
    if (result.converged)
      break;
  }
  result.correlation_energy = old_energy;
  result.t1 = std::move(t1);
  result.t2 = std::move(t2);
  return result;
}

double ccsd_energy(const Matrix& fock, const Tensor4& eri, const Matrix& t1,
                   const Tensor4& t2, const std::size_t occupied) {
  return ccsd_energy_impl(fock, eri, t1, t2, occupied);
}

CcsdResidualEvaluation evaluate_ccsd_residual(
    const Matrix& fock, const Tensor4& eri, const Matrix& t1,
    const Tensor4& t2, const std::size_t occupied,
    const CcsdOptions& options) {
  validate_wabef_pair_symmetry_options(options);
  if (options.pm_symmetry)
    throw std::invalid_argument("PM symmetry residual admission requires the spatial Hamiltonian interface");
  const auto orbitals = fock.dimensions()[0];
  if (fock.dimensions()[1] != orbitals || occupied == 0 ||
      occupied >= orbitals)
    throw std::invalid_argument(
        "inconsistent CCSD residual-oracle dimensions");
  const auto virtuals = orbitals - occupied;
  if (eri.dimensions() !=
          std::array<std::size_t, 4>{orbitals, orbitals, orbitals,
                                     orbitals} ||
      t1.dimensions() !=
          std::array<std::size_t, 2>{occupied, virtuals} ||
      t2.dimensions() !=
          std::array<std::size_t, 4>{occupied, occupied, virtuals,
                                     virtuals})
    throw std::invalid_argument(
        "inconsistent CCSD residual-oracle amplitude dimensions");
  for (const double value : t1.data())
    if (!std::isfinite(value))
      throw std::invalid_argument(
          "non-finite T1 amplitude in CCSD residual oracle");
  for (const double value : t2.data())
    if (!std::isfinite(value))
      throw std::invalid_argument(
          "non-finite T2 amplitude in CCSD residual oracle");
  if (t2_antisymmetry_error(t2) > 1.0e-12)
    throw std::invalid_argument(
        "CCSD residual-oracle T2 amplitudes are not antisymmetric");
  if (options.projection_mask)
    validate_projection_mask(*options.projection_mask, occupied, virtuals);

  Matrix d1({occupied, virtuals});
  Tensor4 d2({occupied, occupied, virtuals, virtuals});
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t a = 0; a < virtuals; ++a) {
      d1(i, a) = fock(i, i) - fock(occupied + a, occupied + a);
      if (std::abs(d1(i, a)) < options.denominator_floor)
        throw std::runtime_error("small CCSD residual-oracle preconditioner");
    }
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t j = 0; j < occupied; ++j)
      for (std::size_t a = 0; a < virtuals; ++a)
        for (std::size_t b = 0; b < virtuals; ++b) {
          d2(i, j, a, b) = d1(i, a) + d1(j, b);
          if (std::abs(d2(i, j, a, b)) < options.denominator_floor)
            throw std::runtime_error(
                "small CCSD residual-oracle preconditioner");
        }

  CcsdResidualEvaluation evaluation;
  const auto* contraction_mask =
      options.projection_mask &&
              options.projection_mask->selected_doubles != t2.size()
          ? options.projection_mask.get()
          : nullptr;
  if (options.wabef_packing_backend ==
          CcsdWabefPackingBackend::cached_invariants &&
      options.wabef_backend != CcsdWabefBackend::tiled_cpu)
    throw std::invalid_argument(
        "cached CCSD W_abef invariant packing requires tiled-cpu W_abef");
  std::optional<TiledWabefInvariantCache> wabef_invariant_cache;
  if (options.wabef_packing_backend ==
      CcsdWabefPackingBackend::cached_invariants) {
    const auto cache_begin = SteadyClock::now();
    wabef_invariant_cache.emplace(build_tiled_wabef_invariant_cache(
        eri, occupied, virtuals, contraction_mask,
        options.wabef_invariant_cache_max_bytes,
        options.use_wabef_pair_symmetry));
    const auto cache_seconds = elapsed_seconds(cache_begin, SteadyClock::now());
    evaluation.timing.wabef_build_seconds += cache_seconds;
    evaluation.timing.wabef_seconds += cache_seconds;
  }
  auto [next_t1, next_t2] = fixed_point(
      fock, eri, t1, t2, occupied, d1, d2, &evaluation.timing,
      options.wabef_backend, options.use_tiled_intermediates,
      options.use_tiled_doubles,
      options.wabef_tile_max_bytes, contraction_mask,
      wabef_invariant_cache ? &*wabef_invariant_cache : nullptr);
  if (options.projection_mask)
    apply_projection_mask(*options.projection_mask, next_t1, next_t2);

  evaluation.r1 = Matrix({occupied, virtuals});
  evaluation.r2 = Tensor4({occupied, occupied, virtuals, virtuals});
  const auto residual_begin = SteadyClock::now();
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t a = 0; a < virtuals; ++a)
      evaluation.r1(i, a) =
          d1(i, a) * (next_t1(i, a) - t1(i, a));
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t j = 0; j < occupied; ++j)
      for (std::size_t a = 0; a < virtuals; ++a)
        for (std::size_t b = 0; b < virtuals; ++b)
          evaluation.r2(i, j, a, b) =
              d2(i, j, a, b) *
              (next_t2(i, j, a, b) - t2(i, j, a, b));
  evaluation.r2 = antisym(evaluation.r2);
  evaluation.r1_norm = norm(evaluation.r1.data());
  evaluation.r2_norm = norm(evaluation.r2.data());
  evaluation.timing.residual_seconds +=
      elapsed_seconds(residual_begin, SteadyClock::now());
  return evaluation;
}

namespace {
const PmSymmetry* residual_pm_group(const CcsdOptions& options,
                                   const SpatialRccsdAmplitudes& amplitudes) {
  if (!options.use_pm_residual_orbits) return nullptr;
  if (!options.pm_symmetry || !options.use_tiled_spatial_pairs)
    throw std::invalid_argument("PM residual orbits require verified symmetry and native spatial pairs");
  validate_pm_spatial_amplitudes(*options.pm_symmetry,amplitudes.t1,amplitudes.t2);
  return options.pm_symmetry.get();
}
}

SpatialRccsdSinglesEvaluation evaluate_spatial_rccsd_singles_direct(
    const SpatialRhfHamiltonian& hamiltonian,
    const SpatialRccsdAmplitudes& amplitudes,
    const CcsdOptions& options) {
  const auto occupied = hamiltonian.nocc;
  const auto virtuals = hamiltonian.nvirt;
  const auto orbitals = checked_memory_sum(occupied, virtuals);
  if (occupied == 0 || virtuals == 0 ||
      hamiltonian.fock.dimensions() !=
          std::array<std::size_t, 2>{orbitals, orbitals} ||
      hamiltonian.eri_chemist.dimensions() !=
          std::array<std::size_t, 4>{orbitals, orbitals, orbitals, orbitals} ||
      amplitudes.t1.dimensions() !=
          std::array<std::size_t, 2>{occupied, virtuals} ||
      amplitudes.t2.dimensions() !=
          std::array<std::size_t, 4>{occupied, occupied, virtuals, virtuals})
    throw std::invalid_argument(
        "inconsistent direct spatial RCCSD singles dimensions");
  if (spatial_rccsd_pair_symmetry_error(amplitudes.t2) > 1.0e-12)
    throw std::invalid_argument(
        "direct spatial RCCSD singles T2 violates pair symmetry");
  for (const auto value : hamiltonian.fock.data())
    if (!std::isfinite(value))
      throw std::invalid_argument(
          "direct spatial RCCSD singles received a non-finite Fock matrix");
  for (const auto value : amplitudes.t1.data())
    if (!std::isfinite(value))
      throw std::invalid_argument(
          "direct spatial RCCSD singles received non-finite T1 amplitudes");
  for (const auto value : amplitudes.t2.data())
    if (!std::isfinite(value))
      throw std::invalid_argument(
          "direct spatial RCCSD singles received non-finite T2 amplitudes");
  if (options.projection_mask)
    throw std::invalid_argument(
        "direct spatial RCCSD singles does not accept a spin projection mask");
  if (occupied > std::numeric_limits<std::size_t>::max() / 2 ||
      virtuals > std::numeric_limits<std::size_t>::max() / 2)
    throw std::overflow_error(
        "direct spatial RCCSD singles spin-index extent overflows");

  const auto spin_occupied = 2 * occupied;
  const auto spin_virtuals = 2 * virtuals;
  const auto spin_virtual_offset = spin_occupied;
  const SpinOrbitalEriView eri(hamiltonian);
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t a = 0; a < virtuals; ++a)
      if (std::abs(hamiltonian.fock(i, i) -
                   hamiltonian.fock(occupied + a, occupied + a)) <
          options.denominator_floor)
        throw std::runtime_error(
            "small direct spatial RCCSD singles preconditioner");

  const auto spin_t1 = [&](const std::size_t i, const std::size_t a) {
    return i % 2 == a % 2 ? amplitudes.t1(i / 2, a / 2) : 0.0;
  };
  const auto spin_t2 = [&](const std::size_t i, const std::size_t j,
                           const std::size_t a, const std::size_t b) {
    double value = 0.0;
    if (i % 2 == a % 2 && j % 2 == b % 2)
      value += amplitudes.t2(i / 2, j / 2, a / 2, b / 2);
    if (i % 2 == b % 2 && j % 2 == a % 2)
      value -= amplitudes.t2(i / 2, j / 2, b / 2, a / 2);
    return value;
  };
  const auto spin_fock = [&](const std::size_t p,
                             const std::size_t q) {
    return p % 2 == q % 2 ? hamiltonian.fock(p / 2, q / 2) : 0.0;
  };
  const auto tilde_tau = [&](const std::size_t i, const std::size_t j,
                             const std::size_t a, const std::size_t b) {
    const auto cross =
        spin_t1(i, a) * spin_t1(j, b) -
        spin_t1(i, b) * spin_t1(j, a);
    return spin_t2(i, j, a, b) + 0.5 * cross;
  };

  Matrix fae({spin_virtuals, spin_virtuals});
  Matrix fmi({spin_occupied, spin_occupied});
  Matrix fme({spin_occupied, spin_virtuals});
  SpatialRccsdSinglesEvaluation evaluation;
  const auto communicator = parallel::world();
  const auto local_count = [&](const std::size_t count) {
    return communicator.rank < count
        ? 1 + (count - 1 - communicator.rank) / communicator.size
        : 0;
  };
  const auto fae_tasks = checked_memory_product(spin_virtuals, spin_virtuals);
  const auto fmi_tasks = checked_memory_product(spin_occupied, spin_occupied);
  const auto fme_tasks = checked_memory_product(spin_occupied, spin_virtuals);
  evaluation.global_intermediate_tasks = checked_memory_sum(
      checked_memory_sum(fae_tasks, fmi_tasks), fme_tasks);
  evaluation.local_intermediate_tasks = checked_memory_sum(
      checked_memory_sum(local_count(fae_tasks), local_count(fmi_tasks)),
      local_count(fme_tasks));
  evaluation.replicated_intermediate_bytes_per_rank = checked_memory_product(
      evaluation.global_intermediate_tasks, sizeof(double));
  const auto expanded_t1_elements =
      checked_memory_product(spin_occupied, spin_virtuals);
  const auto expanded_t2_elements = checked_memory_product(
      expanded_t1_elements, expanded_t1_elements);
  evaluation.avoided_spin_amplitude_bytes_per_rank = checked_memory_product(
      checked_memory_sum(expanded_t1_elements, expanded_t2_elements),
      sizeof(double));

  const auto intermediates_begin = SteadyClock::now();
  const auto* residual_group=residual_pm_group(options,amplitudes);
  const detail::PmOrbitPlan<2> ae_plan(residual_group,{spin_virtuals,spin_virtuals},{occupied,occupied},true);
  const detail::PmOrbitPlan<2> mi_plan(residual_group,{spin_occupied,spin_occupied},{0,0},true);
  const detail::PmOrbitPlan<2> me_plan(residual_group,{spin_occupied,spin_virtuals},{0,occupied},true);
  evaluation.global_intermediate_tasks=ae_plan.tasks()+mi_plan.tasks()+me_plan.tasks();
  evaluation.local_intermediate_tasks=local_count(ae_plan.tasks())+local_count(mi_plan.tasks())+local_count(me_plan.tasks());
  distributed_for(ae_plan.tasks(), [&](const std::size_t task) {
    const auto flat=ae_plan.representative(task);
    const auto a = flat / spin_virtuals;
    const auto e = flat % spin_virtuals;
    double value = a == e
        ? 0.0
        : spin_fock(spin_virtual_offset + a, spin_virtual_offset + e);
    for (std::size_t m = 0; m < spin_occupied; ++m)
      value -= 0.5 *
               spin_fock(m, spin_virtual_offset + e) * spin_t1(m, a);
    for (std::size_t m = 0; m < spin_occupied; ++m)
      for (std::size_t q = 0; q < spin_virtuals; ++q)
        value += spin_t1(m, q) *
                 eri(m, spin_virtual_offset + a, spin_virtual_offset + q,
                     spin_virtual_offset + e);
    for (std::size_t m = 0; m < spin_occupied; ++m)
      for (std::size_t n = 0; n < spin_occupied; ++n)
        for (std::size_t q = 0; q < spin_virtuals; ++q)
          value -= 0.5 * tilde_tau(m, n, a, q) *
                   eri(m, n, spin_virtual_offset + e,
                       spin_virtual_offset + q);
    ae_plan.scatter(flat,value,fae.data());
  });
  parallel::sum_in_place(fae.data());

  distributed_for(mi_plan.tasks(), [&](const std::size_t task) {
    const auto flat=mi_plan.representative(task);
    const auto m = flat / spin_occupied;
    const auto i = flat % spin_occupied;
    double value = m == i ? 0.0 : spin_fock(m, i);
    for (std::size_t e = 0; e < spin_virtuals; ++e)
      value += 0.5 * spin_t1(i, e) *
               spin_fock(m, spin_virtual_offset + e);
    for (std::size_t n = 0; n < spin_occupied; ++n)
      for (std::size_t e = 0; e < spin_virtuals; ++e)
        value += spin_t1(n, e) * eri(m, n, i, spin_virtual_offset + e);
    for (std::size_t n = 0; n < spin_occupied; ++n)
      for (std::size_t e = 0; e < spin_virtuals; ++e)
        for (std::size_t q = 0; q < spin_virtuals; ++q)
          value += 0.5 * tilde_tau(i, n, e, q) *
                   eri(m, n, spin_virtual_offset + e,
                       spin_virtual_offset + q);
    mi_plan.scatter(flat,value,fmi.data());
  });
  parallel::sum_in_place(fmi.data());

  distributed_for(me_plan.tasks(), [&](const std::size_t task) {
    const auto flat=me_plan.representative(task);
    const auto m = flat / spin_virtuals;
    const auto e = flat % spin_virtuals;
    double value = spin_fock(m, spin_virtual_offset + e);
    for (std::size_t n = 0; n < spin_occupied; ++n)
      for (std::size_t q = 0; q < spin_virtuals; ++q)
        value += spin_t1(n, q) *
                 eri(m, n, spin_virtual_offset + e,
                     spin_virtual_offset + q);
    me_plan.scatter(flat,value,fme.data());
  });
  parallel::sum_in_place(fme.data());
  evaluation.timing.intermediates_seconds +=
      elapsed_seconds(intermediates_begin, SteadyClock::now());

  evaluation.r1 = Matrix({occupied, virtuals});
  evaluation.global_output_tasks =
      checked_memory_product(occupied, virtuals);
  evaluation.local_output_tasks = local_count(evaluation.global_output_tasks);
  const auto singles_begin = SteadyClock::now();
  const detail::PmOrbitPlan<2> ia_plan(residual_group,{occupied,virtuals},{0,occupied});
  evaluation.global_output_tasks=ia_plan.tasks();
  evaluation.local_output_tasks=local_count(ia_plan.tasks());
  distributed_for(ia_plan.tasks(), [&](const std::size_t task) {
    const auto flat=ia_plan.representative(task);
    const auto spatial_i = flat / virtuals;
    const auto spatial_a = flat % virtuals;
    const auto i = 2 * spatial_i;
    const auto a = 2 * spatial_a;
    double value = spin_fock(i, spin_virtual_offset + a);
    for (std::size_t e = 0; e < spin_virtuals; ++e)
      value += spin_t1(i, e) * fae(a, e);
    for (std::size_t m = 0; m < spin_occupied; ++m)
      value -= spin_t1(m, a) * fmi(m, i);
    for (std::size_t m = 0; m < spin_occupied; ++m)
      for (std::size_t e = 0; e < spin_virtuals; ++e)
        value += spin_t2(i, m, a, e) * fme(m, e) -
                 spin_t1(m, e) *
                     eri(m, spin_virtual_offset + a, i,
                         spin_virtual_offset + e);
    for (std::size_t m = 0; m < spin_occupied; ++m)
      for (std::size_t e = 0; e < spin_virtuals; ++e)
        for (std::size_t q = 0; q < spin_virtuals; ++q)
          value -= 0.5 * spin_t2(i, m, e, q) *
                   eri(m, spin_virtual_offset + a,
                       spin_virtual_offset + e,
                       spin_virtual_offset + q);
    for (std::size_t m = 0; m < spin_occupied; ++m)
      for (std::size_t n = 0; n < spin_occupied; ++n)
        for (std::size_t e = 0; e < spin_virtuals; ++e)
          value -= 0.5 * spin_t2(m, n, a, e) *
                   eri(n, m, spin_virtual_offset + e, i);
    const auto denominator =
        hamiltonian.fock(spatial_i, spatial_i) -
        hamiltonian.fock(occupied + spatial_a,
                         occupied + spatial_a);
    ia_plan.scatter(flat,value-denominator*amplitudes.t1(spatial_i,spatial_a),evaluation.r1.data());
  });
  parallel::sum_in_place(evaluation.r1.data());
  const auto norm_begin=SteadyClock::now();
  evaluation.r1_norm = norm(evaluation.r1.data());
  evaluation.timing.singles_seconds +=
      elapsed_seconds(singles_begin, SteadyClock::now());
  evaluation.timing.residual_seconds =
      options.use_tiled_spatial_pairs ? elapsed_seconds(norm_begin,SteadyClock::now()) :
      evaluation.timing.intermediates_seconds + evaluation.timing.singles_seconds;
  return evaluation;
}

SpatialRccsdDoublesEvaluation evaluate_spatial_rccsd_doubles_direct(
    const SpatialRhfHamiltonian& hamiltonian,
    const SpatialRccsdAmplitudes& amplitudes,
    const CcsdOptions& options) {
  if (options.use_compact_spatial_doubles &&
      (!options.use_tiled_spatial_pairs || options.use_pm_residual_orbits))
    throw std::invalid_argument("compact spatial doubles requires native pairs without PM residual orbits");
  const auto occupied = hamiltonian.nocc;
  const auto virtuals = hamiltonian.nvirt;
  const auto orbitals = checked_memory_sum(occupied, virtuals);
  if (occupied == 0 || virtuals == 0 ||
      hamiltonian.fock.dimensions() !=
          std::array<std::size_t, 2>{orbitals, orbitals} ||
      hamiltonian.eri_chemist.dimensions() !=
          std::array<std::size_t, 4>{orbitals, orbitals, orbitals, orbitals} ||
      amplitudes.t1.dimensions() !=
          std::array<std::size_t, 2>{occupied, virtuals} ||
      amplitudes.t2.dimensions() !=
          std::array<std::size_t, 4>{occupied, occupied, virtuals, virtuals})
    throw std::invalid_argument(
        "inconsistent direct spatial RCCSD doubles dimensions");
  if (spatial_rccsd_pair_symmetry_error(amplitudes.t2) > 1.0e-12)
    throw std::invalid_argument(
        "direct spatial RCCSD doubles T2 violates pair symmetry");
  for (const auto value : hamiltonian.fock.data())
    if (!std::isfinite(value))
      throw std::invalid_argument(
          "direct spatial RCCSD doubles received a non-finite Fock matrix");
  for (const auto value : amplitudes.t1.data())
    if (!std::isfinite(value))
      throw std::invalid_argument(
          "direct spatial RCCSD doubles received non-finite T1 amplitudes");
  for (const auto value : amplitudes.t2.data())
    if (!std::isfinite(value))
      throw std::invalid_argument(
          "direct spatial RCCSD doubles received non-finite T2 amplitudes");
  if (options.projection_mask)
    throw std::invalid_argument(
        "direct spatial RCCSD doubles does not accept a spin projection mask");
  if (occupied > std::numeric_limits<std::size_t>::max() / 2 ||
      virtuals > std::numeric_limits<std::size_t>::max() / 2)
    throw std::overflow_error(
        "direct spatial RCCSD doubles spin-index extent overflows");

  const auto spin_occupied = 2 * occupied;
  const auto spin_virtuals = 2 * virtuals;
  const auto spin_virtual_offset = spin_occupied;
  const SpinOrbitalEriView eri(hamiltonian);
  const auto spin_t1 = [&](const std::size_t i, const std::size_t a) {
    return i % 2 == a % 2 ? amplitudes.t1(i / 2, a / 2) : 0.0;
  };
  const auto spin_t2 = [&](const std::size_t i, const std::size_t j,
                           const std::size_t a, const std::size_t b) {
    double value = 0.0;
    if (i % 2 == a % 2 && j % 2 == b % 2)
      value += amplitudes.t2(i / 2, j / 2, a / 2, b / 2);
    if (i % 2 == b % 2 && j % 2 == a % 2)
      value -= amplitudes.t2(i / 2, j / 2, b / 2, a / 2);
    return value;
  };
  const auto spin_fock = [&](const std::size_t p,
                             const std::size_t q) {
    return p % 2 == q % 2 ? hamiltonian.fock(p / 2, q / 2) : 0.0;
  };
  const auto tau = [&](const std::size_t i, const std::size_t j,
                       const std::size_t a, const std::size_t b) {
    return spin_t2(i, j, a, b) +
           spin_t1(i, a) * spin_t1(j, b) -
           spin_t1(i, b) * spin_t1(j, a);
  };
  const auto tilde_tau = [&](const std::size_t i, const std::size_t j,
                             const std::size_t a, const std::size_t b) {
    return spin_t2(i, j, a, b) +
           0.5 * (spin_t1(i, a) * spin_t1(j, b) -
                  spin_t1(i, b) * spin_t1(j, a));
  };
  const auto d1 = [&](const std::size_t i, const std::size_t a) {
    return spin_fock(i, i) -
           spin_fock(spin_virtual_offset + a, spin_virtual_offset + a);
  };
  const auto d2 = [&](const std::size_t i, const std::size_t j,
                      const std::size_t a, const std::size_t b) {
    return d1(i, a) + d1(j, b);
  };
  for (std::size_t i = 0; i < spin_occupied; ++i)
    for (std::size_t j = 0; j < spin_occupied; ++j)
      for (std::size_t a = 0; a < spin_virtuals; ++a)
        for (std::size_t b = 0; b < spin_virtuals; ++b)
          if (std::abs(d2(i, j, a, b)) < options.denominator_floor)
            throw std::runtime_error(
                "small direct spatial RCCSD doubles preconditioner");

  SpatialRccsdDoublesEvaluation evaluation;
  Matrix fae({spin_virtuals, spin_virtuals});
  Matrix fmi({spin_occupied, spin_occupied});
  Matrix fme({spin_occupied, spin_virtuals});
  const auto communicator = parallel::world();
  const auto local_count = [&](const std::size_t count) {
    return communicator.rank < count
        ? 1 + (count - 1 - communicator.rank) / communicator.size
        : 0;
  };
  const auto fae_tasks = checked_memory_product(spin_virtuals, spin_virtuals);
  const auto fmi_tasks = checked_memory_product(spin_occupied, spin_occupied);
  const auto fme_tasks = checked_memory_product(spin_occupied, spin_virtuals);
  const auto small_elements = checked_memory_sum(
      checked_memory_sum(fae_tasks, fmi_tasks), fme_tasks);
  evaluation.replicated_small_intermediate_bytes_per_rank =
      checked_memory_product(small_elements, sizeof(double));
  const auto spin_t1_elements = fme_tasks;
  const auto spin_t2_elements = checked_memory_product(
      spin_t1_elements, spin_t1_elements);
  evaluation.avoided_spin_amplitude_bytes_per_rank = checked_memory_product(
      checked_memory_sum(spin_t1_elements, spin_t2_elements),
      sizeof(double));
  const auto spin_occupied_fourth = checked_memory_product(
      checked_memory_product(spin_occupied, spin_occupied),
      checked_memory_product(spin_occupied, spin_occupied));
  const auto spin_four_index_work = checked_memory_sum(
      checked_memory_product(4, spin_t2_elements), spin_occupied_fourth);
  evaluation.avoided_spin_four_index_intermediate_bytes_per_rank =
      checked_memory_product(spin_four_index_work, sizeof(double));

  const auto intermediates_begin = SteadyClock::now();
  const auto* residual_group=residual_pm_group(options,amplitudes);
  const detail::PmOrbitPlan<2> ae_plan(residual_group,{spin_virtuals,spin_virtuals},{occupied,occupied},true);
  const detail::PmOrbitPlan<2> mi_plan(residual_group,{spin_occupied,spin_occupied},{0,0},true);
  const detail::PmOrbitPlan<2> me_plan(residual_group,{spin_occupied,spin_virtuals},{0,occupied},true);
  distributed_for(ae_plan.tasks(), [&](const std::size_t task) {
    const auto flat=ae_plan.representative(task);
    const auto a = flat / spin_virtuals;
    const auto e = flat % spin_virtuals;
    double value = a == e
        ? 0.0
        : spin_fock(spin_virtual_offset + a, spin_virtual_offset + e);
    for (std::size_t m = 0; m < spin_occupied; ++m)
      value -= 0.5 *
               spin_fock(m, spin_virtual_offset + e) * spin_t1(m, a);
    for (std::size_t m = 0; m < spin_occupied; ++m)
      for (std::size_t q = 0; q < spin_virtuals; ++q)
        value += spin_t1(m, q) *
                 eri(m, spin_virtual_offset + a, spin_virtual_offset + q,
                     spin_virtual_offset + e);
    for (std::size_t m = 0; m < spin_occupied; ++m)
      for (std::size_t n = 0; n < spin_occupied; ++n)
        for (std::size_t q = 0; q < spin_virtuals; ++q)
          value -= 0.5 * tilde_tau(m, n, a, q) *
                   eri(m, n, spin_virtual_offset + e,
                       spin_virtual_offset + q);
    ae_plan.scatter(flat,value,fae.data());
  });
  parallel::sum_in_place(fae.data());
  distributed_for(mi_plan.tasks(), [&](const std::size_t task) {
    const auto flat=mi_plan.representative(task);
    const auto m = flat / spin_occupied;
    const auto i = flat % spin_occupied;
    double value = m == i ? 0.0 : spin_fock(m, i);
    for (std::size_t e = 0; e < spin_virtuals; ++e)
      value += 0.5 * spin_t1(i, e) *
               spin_fock(m, spin_virtual_offset + e);
    for (std::size_t n = 0; n < spin_occupied; ++n)
      for (std::size_t e = 0; e < spin_virtuals; ++e)
        value += spin_t1(n, e) * eri(m, n, i, spin_virtual_offset + e);
    for (std::size_t n = 0; n < spin_occupied; ++n)
      for (std::size_t e = 0; e < spin_virtuals; ++e)
        for (std::size_t q = 0; q < spin_virtuals; ++q)
          value += 0.5 * tilde_tau(i, n, e, q) *
                   eri(m, n, spin_virtual_offset + e,
                       spin_virtual_offset + q);
    mi_plan.scatter(flat,value,fmi.data());
  });
  parallel::sum_in_place(fmi.data());
  distributed_for(me_plan.tasks(), [&](const std::size_t task) {
    const auto flat=me_plan.representative(task);
    const auto m = flat / spin_virtuals;
    const auto e = flat % spin_virtuals;
    double value = spin_fock(m, spin_virtual_offset + e);
    for (std::size_t n = 0; n < spin_occupied; ++n)
      for (std::size_t q = 0; q < spin_virtuals; ++q)
        value += spin_t1(n, q) *
                 eri(m, n, spin_virtual_offset + e,
                     spin_virtual_offset + q);
    me_plan.scatter(flat,value,fme.data());
  });
  parallel::sum_in_place(fme.data());
  evaluation.timing.intermediates_seconds +=
      elapsed_seconds(intermediates_begin, SteadyClock::now());

  const auto wmnij = [&](const std::size_t m, const std::size_t n,
                         const std::size_t i, const std::size_t j) {
    double value = eri(m, n, i, j);
    for (std::size_t e = 0; e < spin_virtuals; ++e)
      value += spin_t1(j, e) * eri(m, n, i, spin_virtual_offset + e) -
               spin_t1(i, e) * eri(m, n, j, spin_virtual_offset + e);
    for (std::size_t e = 0; e < spin_virtuals; ++e)
      for (std::size_t q = 0; q < spin_virtuals; ++q)
        value += 0.25 * tau(i, j, e, q) *
                 eri(m, n, spin_virtual_offset + e,
                     spin_virtual_offset + q);
    return value;
  };
  const auto wmbej = [&](const std::size_t m, const std::size_t b,
                         const std::size_t e, const std::size_t j) {
    double value = eri(m, spin_virtual_offset + b,
                       spin_virtual_offset + e, j);
    for (std::size_t q = 0; q < spin_virtuals; ++q)
      value += spin_t1(j, q) *
               eri(m, spin_virtual_offset + b, spin_virtual_offset + e,
                   spin_virtual_offset + q);
    for (std::size_t n = 0; n < spin_occupied; ++n)
      value -= spin_t1(n, b) * eri(m, n, spin_virtual_offset + e, j);
    for (std::size_t n = 0; n < spin_occupied; ++n)
      for (std::size_t q = 0; q < spin_virtuals; ++q)
        value -= (0.5 * spin_t2(j, n, q, b) +
                  spin_t1(j, q) * spin_t1(n, b)) *
                 eri(m, n, spin_virtual_offset + e,
                     spin_virtual_offset + q);
    return value;
  };
  const auto wabef = [&](const std::size_t a, const std::size_t b,
                         const std::size_t e, const std::size_t q) {
    double value = eri(spin_virtual_offset + a, spin_virtual_offset + b,
                       spin_virtual_offset + e, spin_virtual_offset + q);
    for (std::size_t m = 0; m < spin_occupied; ++m)
      value -= spin_t1(m, b) *
                   eri(spin_virtual_offset + a, m,
                       spin_virtual_offset + e,
                       spin_virtual_offset + q) -
               spin_t1(m, a) *
                   eri(spin_virtual_offset + b, m,
                       spin_virtual_offset + e,
                       spin_virtual_offset + q);
    for (std::size_t m = 0; m < spin_occupied; ++m)
      for (std::size_t n = 0; n < spin_occupied; ++n)
        value += 0.25 * tau(m, n, a, b) *
                 eri(m, n, spin_virtual_offset + e,
                     spin_virtual_offset + q);
    return value;
  };
  const auto fae_term = [&](const std::size_t i, const std::size_t j,
                            const std::size_t a, const std::size_t b) {
    double value = 0.0;
    for (std::size_t e = 0; e < spin_virtuals; ++e) {
      double intermediate = fae(b, e);
      for (std::size_t m = 0; m < spin_occupied; ++m)
        intermediate -= 0.5 * spin_t1(m, b) * fme(m, e);
      value += spin_t2(i, j, a, e) * intermediate;
    }
    return value;
  };
  const auto fmi_term = [&](const std::size_t i, const std::size_t j,
                            const std::size_t a, const std::size_t b) {
    double value = 0.0;
    for (std::size_t m = 0; m < spin_occupied; ++m) {
      double intermediate = fmi(m, j);
      for (std::size_t e = 0; e < spin_virtuals; ++e)
        intermediate += 0.5 * spin_t1(j, e) * fme(m, e);
      value += spin_t2(i, m, a, b) * intermediate;
    }
    return value;
  };
  const auto wm_term = [&](const std::size_t i, const std::size_t j,
                           const std::size_t a, const std::size_t b) {
    double value = 0.0;
    for (std::size_t m = 0; m < spin_occupied; ++m)
      for (std::size_t e = 0; e < spin_virtuals; ++e)
        value += spin_t2(i, m, a, e) * wmbej(m, b, e, j) -
                 spin_t1(i, e) * spin_t1(m, a) *
                     eri(m, spin_virtual_offset + b,
                         spin_virtual_offset + e, j);
    return value;
  };
  const auto abej = [&](const std::size_t i, const std::size_t j,
                        const std::size_t a, const std::size_t b) {
    double value = 0.0;
    for (std::size_t e = 0; e < spin_virtuals; ++e)
      value += spin_t1(i, e) *
               eri(spin_virtual_offset + a, spin_virtual_offset + b,
                   spin_virtual_offset + e, j);
    return value;
  };
  const auto mbij = [&](const std::size_t i, const std::size_t j,
                        const std::size_t a, const std::size_t b) {
    double value = 0.0;
    for (std::size_t m = 0; m < spin_occupied; ++m)
      value += spin_t1(m, a) *
               eri(m, spin_virtual_offset + b, i, j);
    return value;
  };
  const auto raw_fixed_point = [&](const std::size_t i,
                                   const std::size_t j,
                                   const std::size_t a,
                                   const std::size_t b) {
    double value = eri(i, j, spin_virtual_offset + a,
                       spin_virtual_offset + b) +
                   fae_term(i, j, a, b) - fae_term(i, j, b, a) -
                   fmi_term(i, j, a, b) + fmi_term(j, i, a, b);
    if (!options.use_tiled_spatial_pairs) {
      for (std::size_t m = 0; m < spin_occupied; ++m)
        for (std::size_t n = 0; n < spin_occupied; ++n)
          value += 0.5 * tau(m, n, a, b) * wmnij(m, n, i, j);
      value += wm_term(i, j, a, b) - wm_term(j, i, a, b) -
               wm_term(i, j, b, a) + wm_term(j, i, b, a);
    }
    value += abej(i, j, a, b) - abej(j, i, a, b) -
             mbij(i, j, a, b) + mbij(i, j, b, a);
    if (!options.use_tiled_spatial_pairs) {
      for (std::size_t e = 0; e < spin_virtuals; ++e)
        for (std::size_t q = 0; q < spin_virtuals; ++q)
          value += 0.5 * tau(i, j, e, q) * wabef(a, b, e, q);
    }
    return value / d2(i, j, a, b);
  };

  evaluation.r2 = Tensor4({occupied, occupied, virtuals, virtuals});
  evaluation.global_output_tasks = evaluation.r2.size();
  evaluation.local_output_tasks = local_count(evaluation.global_output_tasks);
  const auto doubles_begin = SteadyClock::now();
  const detail::PmOrbitPlan<4> ijab_plan(residual_group,
      {occupied,occupied,virtuals,virtuals},{0,0,occupied,occupied},false,
      options.wabef_invariant_cache_max_bytes);
  evaluation.global_output_tasks=ijab_plan.tasks();
  evaluation.local_output_tasks=local_count(ijab_plan.tasks());
  // These are the same-spin blocks used by fae_term/fmi_term below. Hoist
  // their T1.Fme corrections out of every ijab/e/m contraction. Off-diagonal
  // localized Fock elements remain present; no diagonal approximation.
  Matrix compact_fae, compact_fmi;
  if (options.use_compact_spatial_doubles) {
    compact_fae=Matrix({virtuals,virtuals});
    compact_fmi=Matrix({occupied,occupied});
    distributed_for(checked_memory_product(virtuals,virtuals),[&](std::size_t k) {
      const auto b=k/virtuals,e=k%virtuals;
      double x=fae(2*b,2*e);
      for(std::size_t m=0;m<occupied;++m)
        x-=0.5*amplitudes.t1(m,b)*fme(2*m,2*e);
      compact_fae(b,e)=x;
    });
    distributed_for(checked_memory_product(occupied,occupied),[&](std::size_t k) {
      const auto m=k/occupied,j=k%occupied;
      double x=fmi(2*m,2*j);
      for(std::size_t e=0;e<virtuals;++e)
        x+=0.5*amplitudes.t1(j,e)*fme(2*m,2*e);
      compact_fmi(m,j)=x;
    });
    parallel::sum_in_place(compact_fae.data());
    parallel::sum_in_place(compact_fmi.data());
  }
  distributed_for(ijab_plan.tasks(), [&](const std::size_t task) {
    const auto flat=ijab_plan.representative(task);
    auto index = flat;
    const auto spatial_b = index % virtuals;
    index /= virtuals;
    const auto spatial_a = index % virtuals;
    index /= virtuals;
    const auto spatial_j = index % occupied;
    const auto spatial_i = index / occupied;
    if (options.use_compact_spatial_doubles) {
      const auto i=spatial_i,j=spatial_j,a=spatial_a,b=spatial_b;
      const auto& t1=amplitudes.t1;
      const auto& t2=amplitudes.t2;
      const auto& g=hamiltonian.eri_chemist;
      // Opposite-spin ijab component after exact spin summation. Its four
      // antisymmetrized spin permutations give this same spatial expression.
      // W_mnij, W_mbej and W_abef are added by spatial_tiled_pair_terms.
      double x=g(i,occupied+a,j,occupied+b);
      for(std::size_t e=0;e<virtuals;++e) {
        x+=t2(i,j,a,e)*compact_fae(b,e)+t2(i,j,e,b)*compact_fae(a,e);
        x+=t1(i,e)*g(occupied+a,occupied+e,occupied+b,j)
          // P(ij) t_i^e <ab||ej>: the exchanged singles amplitude
          // carries j, but the remaining occupied ERI index is i.
          +t1(j,e)*g(occupied+a,i,occupied+b,occupied+e);
      }
      for(std::size_t m=0;m<occupied;++m) {
        x-=t2(i,m,a,b)*compact_fmi(m,j)+t2(m,j,a,b)*compact_fmi(m,i);
        x-=t1(m,a)*g(m,i,occupied+b,j)+t1(m,b)*g(m,j,occupied+a,i);
      }
      evaluation.r2(i,j,a,b)=x-d2(2*i,2*j+1,2*a,2*b+1)*t2(i,j,a,b);
      return;
    }
    const auto i = 2 * spatial_i;
    const auto j = 2 * spatial_j + 1;
    const auto a = 2 * spatial_a;
    const auto b = 2 * spatial_b + 1;
    const auto next = 0.25 *
        (raw_fixed_point(i, j, a, b) - raw_fixed_point(j, i, a, b) -
         raw_fixed_point(i, j, b, a) + raw_fixed_point(j, i, b, a));
    ijab_plan.scatter(flat,d2(i,j,a,b)*(next-spin_t2(i,j,a,b)),evaluation.r2.data());
  });
  parallel::sum_in_place(evaluation.r2.data());
  evaluation.timing.doubles_seconds +=
      elapsed_seconds(doubles_begin, SteadyClock::now());
  if (options.use_tiled_spatial_pairs) {
    const auto pairs=spatial_tiled_pair_terms(
        hamiltonian, amplitudes, options, evaluation.timing);
    for (std::size_t k=0;k<evaluation.r2.size();++k)
      evaluation.r2.data()[k]+=pairs.data()[k];
  }
  const auto norm_begin=SteadyClock::now();
  evaluation.r2_norm = norm(evaluation.r2.data());
  evaluation.timing.residual_seconds = options.use_tiled_spatial_pairs ?
      elapsed_seconds(norm_begin,SteadyClock::now()) :
      evaluation.timing.intermediates_seconds + evaluation.timing.doubles_seconds;
  return evaluation;
}

namespace {

struct SpatialRccsdResidualWorkspace {
  Matrix spin_fock;
  Matrix d1;
  Tensor4 d2;
  Matrix spin_t1;
  Tensor4 spin_t2;
  std::optional<TiledWabefInvariantCache> wabef_invariant_cache;
  std::size_t allocated_bytes{};
  double build_seconds{};
  double wabef_invariant_cache_build_seconds{};
};

void validate_spatial_rccsd_residual_inputs(
    const SpatialRhfHamiltonian& hamiltonian,
    const SpatialRccsdAmplitudes& amplitudes,
    const CcsdOptions& options) {
  validate_wabef_pair_symmetry_options(options);
  if (options.audit_compact_spatial_doubles && !options.use_compact_spatial_doubles)
    throw std::invalid_argument("compact residual audit requires compact spatial doubles");
  if (options.use_compact_spatial_doubles &&
      (!options.use_tiled_spatial_pairs || options.use_pm_residual_orbits))
    throw std::invalid_argument("compact spatial doubles requires native pairs without PM residual orbits");
  if (options.use_pm_residual_orbits && (!options.pm_symmetry ||
      !options.use_tiled_spatial_pairs || !options.use_direct_spatial_singles ||
      !options.use_direct_spatial_doubles))
    throw std::invalid_argument("PM residual orbits require the complete native spatial executor and verified PM group");
  const auto occupied = hamiltonian.nocc;
  const auto virtuals = hamiltonian.nvirt;
  const auto orbitals = occupied + virtuals;
  if (occupied == 0 || virtuals == 0 ||
      hamiltonian.fock.dimensions() !=
          std::array<std::size_t, 2>{orbitals, orbitals} ||
      hamiltonian.eri_chemist.dimensions() !=
          std::array<std::size_t, 4>{orbitals, orbitals, orbitals, orbitals} ||
      amplitudes.t1.dimensions() !=
          std::array<std::size_t, 2>{occupied, virtuals} ||
      amplitudes.t2.dimensions() !=
          std::array<std::size_t, 4>{occupied, occupied, virtuals, virtuals})
    throw std::invalid_argument(
        "inconsistent spatial RCCSD residual-oracle dimensions");
  if (spatial_rccsd_pair_symmetry_error(amplitudes.t2) > 1.0e-12)
    throw std::invalid_argument(
        "spatial RCCSD residual-oracle T2 violates pair symmetry");
  if (options.projection_mask)
    throw std::invalid_argument(
        "spatial RCCSD residual oracle does not accept a spin projection mask");
  if (options.wabef_packing_backend ==
          CcsdWabefPackingBackend::cached_invariants &&
      options.wabef_backend != CcsdWabefBackend::tiled_cpu)
    throw std::invalid_argument(
        "cached spatial RCCSD W_abef invariant packing requires tiled-cpu "
        "W_abef");
  if (options.use_direct_spatial_doubles &&
      !options.use_direct_spatial_singles)
    throw std::invalid_argument(
        "direct spatial RCCSD doubles requires direct spatial singles");
  if (options.use_tiled_spatial_pairs &&
      (!options.use_direct_spatial_singles || !options.use_direct_spatial_doubles ||
       options.wabef_backend != CcsdWabefBackend::tiled_cpu))
    throw std::invalid_argument("native spatial pairs requires direct spatial R1/R2 and tiled-cpu BLAS");
}

SpatialRccsdResidualWorkspace build_spatial_rccsd_residual_workspace(
    const SpatialRhfHamiltonian& hamiltonian,
    const CcsdOptions& options) {
  const auto build_begin = SteadyClock::now();
  if (options.pm_symmetry) validate_pm_symmetry(hamiltonian, *options.pm_symmetry);
  const auto spin_occupied = 2 * hamiltonian.nocc;
  const auto spin_virtuals = 2 * hamiltonian.nvirt;
  SpatialRccsdResidualWorkspace workspace{
      expand_spatial_one_body_to_spin_orbital(hamiltonian.fock),
      Matrix({spin_occupied, spin_virtuals}),
      Tensor4({spin_occupied, spin_occupied, spin_virtuals, spin_virtuals}),
      Matrix({spin_occupied, spin_virtuals}),
      Tensor4({spin_occupied, spin_occupied, spin_virtuals, spin_virtuals}),
      std::nullopt, 0, 0.0, 0.0};
  for (std::size_t i = 0; i < spin_occupied; ++i)
    for (std::size_t a = 0; a < spin_virtuals; ++a) {
      workspace.d1(i, a) = workspace.spin_fock(i, i) -
          workspace.spin_fock(spin_occupied + a, spin_occupied + a);
      if (std::abs(workspace.d1(i, a)) < options.denominator_floor)
        throw std::runtime_error(
            "small spatial RCCSD singles preconditioner");
    }
  for (std::size_t i = 0; i < spin_occupied; ++i)
    for (std::size_t j = 0; j < spin_occupied; ++j)
      for (std::size_t a = 0; a < spin_virtuals; ++a)
        for (std::size_t b = 0; b < spin_virtuals; ++b) {
          workspace.d2(i, j, a, b) =
              workspace.d1(i, a) + workspace.d1(j, b);
          if (std::abs(workspace.d2(i, j, a, b)) <
              options.denominator_floor)
            throw std::runtime_error(
                "small spatial RCCSD doubles preconditioner");
        }

  if (options.wabef_packing_backend ==
      CcsdWabefPackingBackend::cached_invariants) {
    const auto cache_begin = SteadyClock::now();
    const SpinOrbitalEriView spin_eri(hamiltonian);
    workspace.wabef_invariant_cache.emplace(
        build_tiled_wabef_invariant_cache(
            spin_eri, spin_occupied, spin_virtuals, nullptr,
            options.wabef_invariant_cache_max_bytes,
            options.use_wabef_pair_symmetry, options.pm_symmetry));
    workspace.wabef_invariant_cache_build_seconds =
        elapsed_seconds(cache_begin, SteadyClock::now());
  }
  workspace.allocated_bytes = checked_memory_sum(
      checked_memory_sum(
          checked_memory_product(workspace.spin_fock.size(), sizeof(double)),
          checked_memory_product(workspace.d1.size(), sizeof(double))),
      checked_memory_product(workspace.d2.size(), sizeof(double)));
  workspace.allocated_bytes = checked_memory_sum(
      workspace.allocated_bytes,
      checked_memory_sum(
          checked_memory_product(workspace.spin_t1.size(), sizeof(double)),
          checked_memory_product(workspace.spin_t2.size(), sizeof(double))));
  if (workspace.wabef_invariant_cache)
    workspace.allocated_bytes = checked_memory_sum(
        workspace.allocated_bytes,
        workspace.wabef_invariant_cache->allocated_bytes);
  workspace.build_seconds =
      elapsed_seconds(build_begin, SteadyClock::now());
  return workspace;
}

void refresh_spatial_rccsd_spin_amplitudes(
    const SpatialRccsdAmplitudes& amplitudes,
    SpatialRccsdResidualWorkspace& workspace) {
  const auto occupied = amplitudes.t1.dimensions()[0];
  const auto virtuals = amplitudes.t1.dimensions()[1];
  const auto spin_occupied = 2 * occupied;
  const auto spin_virtuals = 2 * virtuals;
  if (workspace.spin_t1.size() >
          static_cast<std::size_t>(
              std::numeric_limits<std::ptrdiff_t>::max()) ||
      workspace.spin_t2.size() >
          static_cast<std::size_t>(
              std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::overflow_error(
        "spatial RCCSD spin-amplitude workspace exceeds ptrdiff_t");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(workspace.spin_t1.size() >= 16384)
#endif
  for (std::ptrdiff_t flat = 0;
       flat < static_cast<std::ptrdiff_t>(workspace.spin_t1.size()); ++flat) {
    const auto index = static_cast<std::size_t>(flat);
    const auto i = index / spin_virtuals;
    const auto a = index % spin_virtuals;
    workspace.spin_t1(i, a) = i % 2 == a % 2
        ? amplitudes.t1(i / 2, a / 2)
        : 0.0;
  }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(workspace.spin_t2.size() >= 16384)
#endif
  for (std::ptrdiff_t flat = 0;
       flat < static_cast<std::ptrdiff_t>(workspace.spin_t2.size()); ++flat) {
    auto index = static_cast<std::size_t>(flat);
    const auto b = index % spin_virtuals;
    index /= spin_virtuals;
    const auto a = index % spin_virtuals;
    index /= spin_virtuals;
    const auto j = index % spin_occupied;
    const auto i = index / spin_occupied;
    double value = 0.0;
    if (i % 2 == a % 2 && j % 2 == b % 2)
      value += amplitudes.t2(i / 2, j / 2, a / 2, b / 2);
    if (i % 2 == b % 2 && j % 2 == a % 2)
      value -= amplitudes.t2(i / 2, j / 2, b / 2, a / 2);
    workspace.spin_t2(i, j, a, b) = value;
  }
}

SpatialRccsdResidualEvaluation evaluate_spatial_rccsd_residual_with_workspace(
    const SpatialRhfHamiltonian& hamiltonian,
    const SpatialRccsdAmplitudes& amplitudes,
    const CcsdOptions& options,
    SpatialRccsdResidualWorkspace& workspace) {
  validate_spatial_rccsd_residual_inputs(hamiltonian, amplitudes, options);
  if (options.use_direct_spatial_doubles) {
    auto singles = evaluate_spatial_rccsd_singles_direct(
        hamiltonian, amplitudes, options);
    auto doubles = evaluate_spatial_rccsd_doubles_direct(
        hamiltonian, amplitudes, options);
    SpatialRccsdResidualEvaluation direct;
    direct.r1 = std::move(singles.r1);
    direct.r2 = std::move(doubles.r2);
    direct.r1_norm = singles.r1_norm;
    direct.r2_norm = doubles.r2_norm;
    add_timing(direct.timing, singles.timing);
    add_timing(direct.timing, doubles.timing);
    return direct;
  }

  const auto occupied = hamiltonian.nocc;
  const auto virtuals = hamiltonian.nvirt;
  const auto spin_occupied = 2 * occupied;
  const SpinOrbitalEriView spin_eri(hamiltonian);
  refresh_spatial_rccsd_spin_amplitudes(amplitudes, workspace);
  const auto& spin_t1 = workspace.spin_t1;
  const auto& spin_t2 = workspace.spin_t2;

  SpatialRccsdResidualEvaluation evaluation;
  auto [next_t1, next_t2] = fixed_point(
      workspace.spin_fock, spin_eri, spin_t1, spin_t2, spin_occupied,
      workspace.d1, workspace.d2,
      &evaluation.timing, options.wabef_backend,
      options.use_tiled_intermediates, options.use_tiled_doubles,
      options.wabef_tile_max_bytes, nullptr,
      workspace.wabef_invariant_cache
          ? &*workspace.wabef_invariant_cache
          : nullptr,
      nullptr,
      options.use_direct_spatial_singles);

  evaluation.r1 = Matrix({occupied, virtuals});
  evaluation.r2 = Tensor4({occupied, occupied, virtuals, virtuals});
  const auto residual_begin = SteadyClock::now();
  if (options.use_direct_spatial_singles) {
    auto direct = evaluate_spatial_rccsd_singles_direct(
        hamiltonian, amplitudes, options);
    evaluation.r1 = std::move(direct.r1);
    evaluation.r1_norm = direct.r1_norm;
    add_timing(evaluation.timing, direct.timing);
  } else {
    for (std::size_t i = 0; i < occupied; ++i)
      for (std::size_t a = 0; a < virtuals; ++a) {
        const auto spin_i = 2 * i;
        const auto spin_a = 2 * a;
        evaluation.r1(i, a) =
            workspace.d1(spin_i, spin_a) *
            (next_t1(spin_i, spin_a) - spin_t1(spin_i, spin_a));
      }
    evaluation.r1_norm = norm(evaluation.r1.data());
  }
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t j = 0; j < occupied; ++j)
      for (std::size_t a = 0; a < virtuals; ++a)
        for (std::size_t b = 0; b < virtuals; ++b) {
          const auto spin_i = 2 * i;
          const auto spin_j = 2 * j + 1;
          const auto spin_a = 2 * a;
          const auto spin_b = 2 * b + 1;
          evaluation.r2(i, j, a, b) =
              workspace.d2(spin_i, spin_j, spin_a, spin_b) *
              (next_t2(spin_i, spin_j, spin_a, spin_b) -
               spin_t2(spin_i, spin_j, spin_a, spin_b));
        }
  evaluation.r2_norm = norm(evaluation.r2.data());
  evaluation.timing.residual_seconds +=
      elapsed_seconds(residual_begin, SteadyClock::now());
  return evaluation;
}

} // namespace

SpatialRccsdResidualEvaluation evaluate_spatial_rccsd_residual(
    const SpatialRhfHamiltonian& hamiltonian,
    const SpatialRccsdAmplitudes& amplitudes,
    const CcsdOptions& options) {
  validate_spatial_rccsd_residual_inputs(hamiltonian, amplitudes, options);
  if (options.pm_symmetry && options.use_tiled_spatial_pairs)
    validate_pm_symmetry(hamiltonian, *options.pm_symmetry);
  if (options.use_direct_spatial_doubles) {
    SpatialRccsdResidualWorkspace unused_workspace;
    return evaluate_spatial_rccsd_residual_with_workspace(
        hamiltonian, amplitudes, options, unused_workspace);
  }
  auto workspace =
      build_spatial_rccsd_residual_workspace(hamiltonian, options);
  return evaluate_spatial_rccsd_residual_with_workspace(
      hamiltonian, amplitudes, options, workspace);
}

SpatialRccsdResult solve_spatial_rccsd_reference(
    const SpatialRhfHamiltonian& hamiltonian,
    const CcsdOptions& options) {
  const auto occupied = hamiltonian.nocc;
  const auto virtuals = hamiltonian.nvirt;
  const auto orbitals = occupied + virtuals;
  if (occupied == 0 || virtuals == 0 ||
      hamiltonian.fock.dimensions() !=
          std::array<std::size_t, 2>{orbitals, orbitals} ||
      hamiltonian.eri_chemist.dimensions() !=
          std::array<std::size_t, 4>{orbitals, orbitals, orbitals, orbitals})
    throw std::invalid_argument(
        "inconsistent spatial RCCSD reference-solver dimensions");
  if (options.amplitude_backend == CcsdAmplitudeBackend::global_arrays)
    throw std::invalid_argument(
        "spatial RCCSD reference solver does not claim distributed storage");
  if (options.projection_mask)
    throw std::invalid_argument(
        "spatial RCCSD reference solver does not accept a spin projection mask");
  if (options.observer)
    throw std::invalid_argument(
        "spatial RCCSD reference solver has no spin-amplitude checkpoint observer");
  if (options.initial_t1.has_value() != options.initial_t2.has_value())
    throw std::invalid_argument(
        "spatial RCCSD restart requires both spin-orbital T1 and T2");
  if (options.pm_symmetry && options.use_tiled_spatial_pairs)
    validate_pm_symmetry(hamiltonian, *options.pm_symmetry);

  Matrix d1({occupied, virtuals});
  Tensor4 d2({occupied, occupied, virtuals, virtuals});
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t a = 0; a < virtuals; ++a) {
      d1(i, a) = hamiltonian.fock(i, i) -
                 hamiltonian.fock(occupied + a, occupied + a);
      if (std::abs(d1(i, a)) < options.denominator_floor)
        throw std::runtime_error(
            "small spatial RCCSD singles preconditioner");
    }
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t j = 0; j < occupied; ++j)
      for (std::size_t a = 0; a < virtuals; ++a)
        for (std::size_t b = 0; b < virtuals; ++b) {
          d2(i, j, a, b) = d1(i, a) + d1(j, b);
          if (std::abs(d2(i, j, a, b)) < options.denominator_floor)
            throw std::runtime_error(
                "small spatial RCCSD doubles preconditioner");
        }

  SpatialRccsdAmplitudes amplitudes{
      Matrix({occupied, virtuals}),
      Tensor4({occupied, occupied, virtuals, virtuals})};
  if (options.initial_t1) {
    const auto spin_occupied = 2 * occupied;
    const auto spin_virtuals = 2 * virtuals;
    if (options.initial_t1->dimensions() !=
            std::array<std::size_t, 2>{spin_occupied, spin_virtuals} ||
        options.initial_t2->dimensions() !=
            std::array<std::size_t, 4>{spin_occupied, spin_occupied,
                                       spin_virtuals, spin_virtuals})
      throw std::invalid_argument(
          "spatial RCCSD restart spin-amplitude dimensions mismatch");
    for (std::size_t i = 0; i < occupied; ++i)
      for (std::size_t a = 0; a < virtuals; ++a)
        amplitudes.t1(i, a) = (*options.initial_t1)(2 * i, 2 * a);
    for (std::size_t i = 0; i < occupied; ++i)
      for (std::size_t j = 0; j < occupied; ++j)
        for (std::size_t a = 0; a < virtuals; ++a)
          for (std::size_t b = 0; b < virtuals; ++b)
            amplitudes.t2(i, j, a, b) =
                (*options.initial_t2)(2 * i, 2 * j + 1, 2 * a, 2 * b + 1);
    const auto reconstructed_t1 =
        expand_spatial_rccsd_t1_to_spin_orbital(amplitudes.t1);
    const auto reconstructed_t2 =
        expand_spatial_rccsd_t2_to_spin_orbital(amplitudes.t2);
    double restart_error = 0.0;
    for (std::size_t index = 0; index < reconstructed_t1.size(); ++index)
      restart_error = std::max(
          restart_error,
          std::abs(reconstructed_t1.data()[index] -
                   options.initial_t1->data()[index]));
    for (std::size_t index = 0; index < reconstructed_t2.size(); ++index)
      restart_error = std::max(
          restart_error,
          std::abs(reconstructed_t2.data()[index] -
                   options.initial_t2->data()[index]));
    if (restart_error > 1.0e-10)
      throw std::invalid_argument(
          "spatial RCCSD restart amplitudes leave the closed-shell manifold");
  } else {
    for (std::size_t i = 0; i < occupied; ++i)
      for (std::size_t j = 0; j < occupied; ++j)
        for (std::size_t a = 0; a < virtuals; ++a)
          for (std::size_t b = 0; b < virtuals; ++b)
            amplitudes.t2(i, j, a, b) =
                hamiltonian.eri_chemist(i, occupied + a, j, occupied + b) /
                d2(i, j, a, b);
  }
  if (spatial_rccsd_pair_symmetry_error(amplitudes.t2) > 1.0e-12)
    throw std::runtime_error(
        "spatial RCCSD initial amplitudes violate pair symmetry");

  Diis diis(options.diis_size,
            options.use_incremental_diis_gram
                ? DiisGramBackend::incremental
                : DiisGramBackend::recomputed);
  SpatialRccsdResult result;
  std::optional<SpatialRccsdResidualWorkspace> residual_workspace;
  if (!options.use_direct_spatial_doubles) {
    residual_workspace.emplace(
        build_spatial_rccsd_residual_workspace(hamiltonian, options));
    result.residual_workspace_bytes = residual_workspace->allocated_bytes;
    result.residual_workspace_build_seconds =
        residual_workspace->build_seconds;
    result.wabef_invariant_cache_bytes =
        residual_workspace->wabef_invariant_cache
            ? residual_workspace->wabef_invariant_cache->allocated_bytes
            : 0;
    result.wabef_invariant_cache_build_seconds =
        residual_workspace->wabef_invariant_cache_build_seconds;
  } else if (options.use_tiled_spatial_pairs) {
    // Empty dispatch object only: no spin storage. Hamiltonian covariance
    // was checked once above, not redundantly in every residual evaluation.
    residual_workspace.emplace();
  }
  result.history = options.initial_history;
  if (!result.history.empty() &&
      result.history.back().iteration != options.completed_iterations)
    throw std::invalid_argument(
        "spatial RCCSD restart history/iteration mismatch");
  double old_energy = spatial_rccsd_energy(hamiltonian, amplitudes);

  for (std::size_t iteration = options.completed_iterations + 1;
       iteration <= options.max_iterations; ++iteration) {
    const auto iteration_begin = SteadyClock::now();
    auto residual = residual_workspace
        ? evaluate_spatial_rccsd_residual_with_workspace(
              hamiltonian, amplitudes, options, *residual_workspace)
        : evaluate_spatial_rccsd_residual(
              hamiltonian, amplitudes, options);
    auto iteration_timing = residual.timing;
    result.r1_norm = residual.r1_norm;
    result.r2_norm = residual.r2_norm;
    const auto r1_measure = ccsd_residual_measure(
        result.r1_norm, residual.r1.size(), options.residual_metric);
    const auto r2_measure = ccsd_residual_measure(
        result.r2_norm, residual.r2.size(), options.residual_metric);
    const bool converged = !result.history.empty() &&
        std::abs(result.history.back().delta_energy) <
            options.energy_tolerance &&
        std::max(r1_measure, r2_measure) < options.residual_tolerance;
    if (options.audit_compact_spatial_doubles &&
        (iteration == options.completed_iterations + 1 || converged)) {
      if (!options.use_compact_spatial_doubles)
        throw std::invalid_argument("compact residual audit requires compact spatial doubles");
      if (parallel::world().rank==0)
        std::cerr << "progress                   compact spatial molecular residual audit started; iteration=" << iteration << '\n';
      const auto audit_begin=SteadyClock::now();
      auto old_options=options;
      old_options.use_compact_spatial_doubles=false;
      old_options.audit_compact_spatial_doubles=false;
      const auto old=evaluate_spatial_rccsd_residual(hamiltonian,amplitudes,old_options);
      std::ostringstream audit;
      audit.setf(std::ios::scientific); audit.precision(12);
      audit << "progress                   compact spatial molecular residual audit; iteration=" << iteration;
      bool passed=true;
      for(const auto tensors : {std::pair{&old.r1.data(),&residual.r1.data()},
                                std::pair{&old.r2.data(),&residual.r2.data()}}) {
        double inf=0,l2=0,scale=0;
        for(std::size_t k=0;k<tensors.first->size();++k) {
          const double x=(*tensors.first)[k],y=(*tensors.second)[k];
          passed=passed && std::isfinite(x) && std::isfinite(y);
          inf=std::max(inf,std::abs(x-y)); l2+=(x-y)*(x-y);
          scale=std::max({scale,std::abs(x),std::abs(y)});
        }
        passed=passed && inf<=1e-12+1e-12*scale;
        audit << "; infinity=" << inf << "; two=" << std::sqrt(l2);
      }
      audit << "; old_executor_seconds=" << elapsed_seconds(audit_begin,SteadyClock::now())
            << "; passed=" << (passed?1:0);
      if(parallel::world().rank==0) std::cerr << audit.str() << '\n';
      if(!passed) throw std::runtime_error("compact spatial molecular residual audit failed; no tolerance relaxation");
    }
    if (converged) {
      result.converged = true;
      result.history.back().r1_norm = result.r1_norm;
      result.history.back().r2_norm = result.r2_norm;
      iteration_timing.iteration_seconds =
          elapsed_seconds(iteration_begin, SteadyClock::now());
      add_timing(result.cumulative_timing, iteration_timing);
      if (options.iteration_observer)
        options.iteration_observer(CcsdIterationReport{
            result.history.back().iteration, old_energy,
            result.history.back().delta_energy, result.r1_norm,
            result.r2_norm, r1_measure, r2_measure, true,
            CcsdAmplitudeBackend::replicated, options.wabef_backend,
            iteration_timing});
      if (options.spatial_observer)
        options.spatial_observer(SpatialRccsdProgress{
            result.history.back().iteration, old_energy, result.r1_norm,
            result.r2_norm, true, amplitudes, result.history});
      break;
    }

    SpatialRccsdAmplitudes next = amplitudes;
    for (std::size_t i = 0; i < occupied; ++i)
      for (std::size_t a = 0; a < virtuals; ++a)
        next.t1(i, a) +=
            options.damping * residual.r1(i, a) / d1(i, a);
    for (std::size_t i = 0; i < occupied; ++i)
      for (std::size_t j = 0; j < occupied; ++j)
        for (std::size_t a = 0; a < virtuals; ++a)
          for (std::size_t b = 0; b < virtuals; ++b)
            next.t2(i, j, a, b) +=
                options.damping * residual.r2(i, j, a, b) /
                d2(i, j, a, b);

    const auto diis_begin = SteadyClock::now();
    if (iteration >= options.diis_start) {
      std::vector<double> vector = next.t1.data();
      vector.insert(vector.end(), next.t2.data().begin(), next.t2.data().end());
      std::vector<double> error = residual.r1.data();
      error.insert(error.end(), residual.r2.data().begin(),
                   residual.r2.data().end());
      std::vector<int> diis_failed(1,0);
      if (!options.use_root_replicated_diis || parallel::world().rank==0) {
        try {
          if (options.spatial_diis_metric != SpatialDiisMetric::raw)
            weight_spatial_diis_error(error, d1.data(), d2.data(),
                                     options.spatial_diis_metric);
          vector = diis.extrapolate(vector, error);
          const auto diis_work = diis.last_work();
          iteration_timing.diis_error_dot_products += diis_work.error_dot_products;
          iteration_timing.diis_history_vector_accumulations +=
              diis_work.history_vector_accumulations;
        } catch (...) {
          if (!options.use_root_replicated_diis) throw;
          diis_failed[0]=1;
        }
      }
      if (options.use_root_replicated_diis) {
        parallel::broadcast_in_place(diis_failed);
        if(diis_failed[0]) throw std::runtime_error("rank-zero spatial DIIS failed");
        parallel::broadcast_in_place(vector);
      }
      if (vector.size() != next.t1.size() + next.t2.size())
        throw std::runtime_error(
            "spatial RCCSD DIIS output dimensions are inconsistent");
      std::copy_n(vector.begin(), next.t1.size(), next.t1.data().begin());
      std::copy(vector.begin() + static_cast<std::ptrdiff_t>(next.t1.size()),
                vector.end(), next.t2.data().begin());
      for (std::size_t i = 0; i < occupied; ++i)
        for (std::size_t j = 0; j < occupied; ++j)
          for (std::size_t a = 0; a < virtuals; ++a)
            for (std::size_t b = 0; b < virtuals; ++b)
              next.t2(i, j, a, b) =
                  0.5 * (next.t2(i, j, a, b) +
                         next.t2(j, i, b, a));
    }
    iteration_timing.diis_seconds +=
        elapsed_seconds(diis_begin, SteadyClock::now());
    if (spatial_rccsd_pair_symmetry_error(next.t2) > 1.0e-11)
      throw std::runtime_error(
          "spatial RCCSD update violates pair symmetry");
    const auto energy_begin = SteadyClock::now();
    const auto new_energy = spatial_rccsd_energy(hamiltonian, next);
    iteration_timing.energy_seconds +=
        elapsed_seconds(energy_begin, SteadyClock::now());
    result.history.push_back({iteration, new_energy, new_energy - old_energy,
                              result.r1_norm, result.r2_norm});
    amplitudes = std::move(next);
    old_energy = new_energy;
    iteration_timing.iteration_seconds =
        elapsed_seconds(iteration_begin, SteadyClock::now());
    add_timing(result.cumulative_timing, iteration_timing);
    if (options.iteration_observer)
      options.iteration_observer(CcsdIterationReport{
          iteration, old_energy, result.history.back().delta_energy,
          result.r1_norm, result.r2_norm, r1_measure, r2_measure, false,
          CcsdAmplitudeBackend::replicated, options.wabef_backend,
          iteration_timing});
    if (options.spatial_observer &&
        iteration % options.checkpoint_interval == 0)
      options.spatial_observer(SpatialRccsdProgress{
          iteration, old_energy, result.r1_norm, result.r2_norm, false,
          amplitudes, result.history});
  }
  result.correlation_energy = old_energy;
  result.amplitudes = std::move(amplitudes);
  return result;
}

CcsdResult solve_ccsd(const Matrix& fock, const Tensor4& eri,
                      const std::size_t occupied,
                      const CcsdOptions& options) {
  return solve_ccsd_impl(fock, eri, occupied, options);
}

CcsdResult solve_ccsd(const SpatialRhfHamiltonian& hamiltonian,
                      const CcsdOptions& options) {
  if (hamiltonian.nocc > std::numeric_limits<std::size_t>::max() / 2)
    throw std::overflow_error("spatial occupied count cannot be spin expanded");
  const auto spin_fock =
      expand_spatial_one_body_to_spin_orbital(hamiltonian.fock);
  const SpinOrbitalEriView spin_eri(hamiltonian);
  return solve_ccsd_impl(spin_fock, spin_eri, 2 * hamiltonian.nocc, options);
}

CcsdResult solve_ccsd(const Matrix& spatial_fock,
                      const std::size_t spatial_occupied,
                      const std::size_t spatial_virtuals,
                      const SelectedIntegralStore& integrals,
                      const CcsdOptions& options) {
  if (spatial_occupied == 0 || spatial_virtuals == 0 ||
      spatial_occupied > std::numeric_limits<std::size_t>::max() / 2 ||
      spatial_virtuals > std::numeric_limits<std::size_t>::max() / 2)
    throw std::invalid_argument(
        "selected-integral CCSD spatial dimensions are invalid");
  const auto orbitals = spatial_occupied + spatial_virtuals;
  if (spatial_fock.dimensions() !=
          std::array<std::size_t, 2>{orbitals, orbitals} ||
      integrals.dimensions() !=
          std::array<std::size_t, 4>{2 * orbitals, 2 * orbitals,
                                     2 * orbitals, 2 * orbitals})
    throw std::invalid_argument(
        "selected-integral CCSD Hamiltonian dimensions mismatch");
  const auto spin_fock =
      expand_spatial_one_body_to_spin_orbital(spatial_fock);
  return solve_ccsd_impl(spin_fock, integrals, 2 * spatial_occupied,
                         options);
}

} // namespace localized_cc
