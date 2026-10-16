#pragma once

#include "modernqc/basis/basis_set.hpp"
#include "modernqc/linalg/matrix.hpp"
#include "modernqc/localization/orbital_localization.hpp"
#include "modernqc/molecule/molecule.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

namespace modernqc::integrals {

struct OneElectronIntegrals {
  linalg::Matrix overlap;
  linalg::Matrix kinetic;
  linalg::Matrix nuclear_attraction;
  linalg::Matrix core_hamiltonian;
};

// Backend-neutral data needed to construct an AO orthogonalization from an
// atom-centered reference basis. target_source_overlap is
// <target AO|source AO>; source_basis owns only copied metadata.
struct ReferenceBasisOverlap {
  basis::BasisSet source_basis;
  linalg::Matrix target_source_overlap;
};

struct DirectFockResult {
  linalg::Matrix two_electron;
  std::uint64_t unique_shell_quartets;
  std::uint64_t owned_shell_quartets;
  std::uint64_t evaluated_shell_quartets;
  std::uint64_t screened_shell_quartets;
  std::uint64_t schwarz_screened_shell_quartets;
  std::uint64_t density_screened_shell_quartets;
  // Conservative sum of per-quartet bounds. This upper-bounds the maximum
  // elementwise Fock error for this density, although it can be very loose.
  double screened_fock_maximum_error_bound;
};

struct ScreeningOptions {
  // Shell quartets with Q_ab Q_cd below this ERI upper bound are omitted.
  double schwarz_threshold{0.0};
  // If enabled, omit a quartet when a conservative bound on every one of its
  // individual Fock-matrix element contributions is below this threshold.
  bool density_aware{false};
  double fock_contribution_threshold{0.0};
};

struct ParallelBuildOptions {
  std::size_t rank{0};
  std::size_t ranks{1};
  std::size_t threads{1};
};

struct ValidationAoEriTensor {
  std::size_t ao_count;
  // Chemists' notation (mu nu | lambda sigma), sigma fastest.
  std::vector<double> values;

  [[nodiscard]] double operator()(std::size_t mu, std::size_t nu,
                                  std::size_t lambda,
                                  std::size_t sigma) const {
    return values.at(
        (((mu * ao_count) + nu) * ao_count + lambda) * ao_count + sigma);
  }
};

// Exact first-index AO-to-MO transformation for a caller-owned subset of
// spatial orbitals,
//
//   T[p,nu,lambda,sigma] = sum_mu C[mu,p] (mu nu|lambda sigma).
//
// This is the semidirect production boundary used by CCSD(T): AO shell
// quartets are evaluated directly and no AO four-index tensor is formed.
// Different MPI ranks may request disjoint orbital_indices and assemble the
// remaining three transformations collectively in their orchestration layer.
struct FirstIndexTransformedEriResult {
  std::size_t ao_count;
  std::vector<std::size_t> orbital_indices;
  // Row-major [owned orbital, nu, lambda, sigma], sigma fastest.
  std::vector<double> values;
  std::size_t estimated_peak_additional_bytes;
  // Global number of unique shell quartets and the number evaluated by this
  // caller's shell_quartet_parallel rank, respectively.
  std::uint64_t unique_shell_quartets;
  std::uint64_t evaluated_shell_quartets;

  [[nodiscard]] double operator()(std::size_t owned_orbital,
                                  std::size_t nu,
                                  std::size_t lambda,
                                  std::size_t sigma) const {
    return values.at(
        (((owned_orbital * ao_count) + nu) * ao_count + lambda) * ao_count +
        sigma);
  }
};

struct OvovTileOptions {
  std::size_t left_occupied_first;
  std::size_t left_occupied_count;
  std::size_t left_virtual_first;
  std::size_t left_virtual_count;
  std::size_t right_occupied_first;
  std::size_t right_occupied_count;
  std::size_t right_virtual_first;
  std::size_t right_virtual_count;
  std::size_t threads{1};
  // AO shell quartets with Q_mn Q_ls below this ERI bound are omitted. Zero
  // is the exact no-screen reference.
  double schwarz_threshold{0.0};
  // Fail rather than entering the higher-scaling pair-matrix fallback when
  // the separable four-pass intermediates exceed the memory limit.
  bool require_separable_four_pass{false};
  // Emit coarse, rank-zero progress for long staged transformations.
  bool report_progress{false};
  // Retain the exact two-index-transformed intermediate
  //
  //   T[(a,i),lambda,sigma]
  //     = sum_{mu,nu} C_occ[mu,i] C_vir[nu,a]
  //       (mu nu|lambda sigma)
  //
  // from the separable four-pass route. This is an experimental streaming
  // boundary used to form multiple selected localized OVOV tiles without
  // revisiting AO shell quartets. It is not a density-fitted or three-index
  // ERI representation. The option requires the separable route.
  bool retain_left_pair_ao_intermediate{false};
  std::size_t maximum_additional_memory_bytes{
      512U * 1024U * 1024U};
};

struct OvovTileResult {
  std::size_t left_occupied_count;
  std::size_t left_virtual_count;
  std::size_t right_occupied_count;
  std::size_t right_virtual_count;
  // Column-major combined-pair matrix: row (i,a), column (j,b).
  std::vector<double> values;
  // Optional column-major [(a,i),(lambda,sigma)] intermediate. The combined
  // AO pair is fastest inside each (a,i) column. Empty unless explicitly
  // requested by retain_left_pair_ao_intermediate.
  std::vector<double> left_pair_ao_intermediate;
  std::size_t ao_count;
  std::size_t estimated_peak_additional_bytes;
  // Number of shell-quartet transform tasks in the active kernel. The
  // separable four-pass route uses ordered quartets; the memory fallback uses
  // symmetry-compressed shell-pair quartets.
  std::uint64_t ordered_shell_quartets;
  std::uint64_t evaluated_shell_quartets;
  std::uint64_t schwarz_screened_shell_quartets;

  [[nodiscard]] double operator()(std::size_t i, std::size_t a,
                                  std::size_t j,
                                  std::size_t b) const {
    const std::size_t left_pair = i * left_virtual_count + a;
    const std::size_t right_pair = j * right_virtual_count + b;
    const std::size_t left_pair_count =
        left_occupied_count * left_virtual_count;
    return values.at(right_pair * left_pair_count + left_pair);
  }
};

// One right-AO block of exact first-index-transformed four-center ERIs for a
// batch of coefficient matrices. Every ordered AO shell quartet belongs to
// exactly one block over a complete traversal. Values use the layout
// [point, lambda_local, sigma, occupied_local, nu], with nu fastest.
struct BatchedFirstIndexAoBlock {
  std::size_t ao_count;
  std::size_t lambda_ao_first;
  std::size_t lambda_ao_count;
  std::size_t occupied_count;
  std::vector<std::vector<double>> point_values;
  std::size_t estimated_peak_additional_bytes;
  std::uint64_t ordered_shell_quartets;
  std::uint64_t evaluated_shell_quartets;
  std::uint64_t schwarz_screened_shell_quartets;
};

struct BatchedFirstIndexAoOptions {
  std::size_t occupied_first;
  std::size_t occupied_count;
  // Consecutive lambda shells are grouped without splitting a shell. The
  // actual AO count can exceed this target only for one larger shell.
  std::size_t target_lambda_ao_block_size{16};
  std::size_t threads{1};
  double schwarz_threshold{0.0};
  bool report_progress{false};
  std::size_t maximum_additional_memory_bytes{
      512U * 1024U * 1024U};
};

using BatchedFirstIndexAoBlockConsumer =
    std::function<void(const BatchedFirstIndexAoBlock&)>;

class Libint2IntegralProvider {
 public:
  Libint2IntegralProvider(const molecule::Molecule& molecule,
                          std::string_view basis_name, bool spherical);
  ~Libint2IntegralProvider();

  Libint2IntegralProvider(const Libint2IntegralProvider&) = delete;
  Libint2IntegralProvider& operator=(const Libint2IntegralProvider&) = delete;
  Libint2IntegralProvider(Libint2IntegralProvider&&) noexcept;
  Libint2IntegralProvider& operator=(Libint2IntegralProvider&&) noexcept;

  [[nodiscard]] const basis::BasisSet& basis_metadata() const noexcept;
  [[nodiscard]] OneElectronIntegrals compute_one_electron() const;
  // Rectangular cross overlap <target_mu|source_nu>. Both providers must use
  // the same atoms, coordinates, charge/multiplicity convention, and
  // spherical/cartesian convention. This is the metric bridge used for
  // auditable basis-progression restarts; it does not form AO ERIs.
  [[nodiscard]] linalg::Matrix compute_cross_overlap(
      const Libint2IntegralProvider& source) const;
  // Builds an internal atom-centered reference basis without exposing
  // Libint2 types. The initial compatibility path intentionally accepts only
  // ANO-RCC for H/O meta-Lowdin population analysis.
  [[nodiscard]] ReferenceBasisOverlap compute_ano_reference_overlap() const;
  // Analytic Cartesian position moments <mu|x^a y^b z^c|nu> for total
  // degree one through four, transformed to the provider's spherical AO
  // ordering. Used by FM1--FM4 localization.
  [[nodiscard]] localization::CartesianMomentIntegrals
  compute_cartesian_moments_through_fourth() const;
  // Returns AO values phi_mu(r_p) as a [point, AO] matrix. Coordinates are
  // in bohr and AO ordering matches basis_metadata().
  [[nodiscard]] linalg::Matrix evaluate_basis_values(
      const std::vector<std::array<double, 3>>& points_bohr) const;
  // Validation-only exact AO ERI tensor. Production RHF/MP2 paths must not
  // use this interface. The byte limit is checked before allocation or ERIs.
  [[nodiscard]] ValidationAoEriTensor compute_validation_ao_eris(
      std::size_t maximum_bytes) const;
  // Exact, bounded semidirect first-index transformation. orbital_indices
  // must be strictly increasing and refer to columns of coefficients. The
  // explicit byte limit covers the returned slice and is checked before
  // allocation or shell-quartet evaluation. shell_quartet_parallel may split
  // the unique shell quartets across cooperating callers; in that case each
  // returned values array is a partial sum and the caller must sum all ranks
  // in that topology before using it.
  [[nodiscard]] FirstIndexTransformedEriResult
  compute_first_index_transformed_eris(
      const linalg::Matrix& coefficients,
      const std::vector<std::size_t>& orbital_indices,
      std::size_t maximum_additional_memory_bytes,
      const ParallelBuildOptions& shell_quartet_parallel = {}) const;
  // Direct, no-screen AO-to-MO transformation of one bounded OVOV tile.
  // No AO four-index tensor is formed or retained.
  [[nodiscard]] OvovTileResult compute_ovov_tile(
      const linalg::Matrix& occupied_coefficients,
      const linalg::Matrix& virtual_coefficients,
      const OvovTileOptions& options) const;
  // Streams exact, lambda-blocked first-index transformations for several
  // Laplace-scaled occupied coefficient matrices. Schwarz bounds and each
  // surviving four-center AO shell quartet are evaluated once, then reused
  // across every matrix in the batch. No AO ERI tensor or factorized
  // three-index representation is formed.
  void for_each_batched_first_index_ao_block(
      const std::vector<linalg::Matrix>& occupied_coefficients_by_point,
      const BatchedFirstIndexAoOptions& options,
      const BatchedFirstIndexAoBlockConsumer& consume) const;
  // The density is spin-summed. The returned matrix is J[P] - 0.5 K[P].
  [[nodiscard]] DirectFockResult build_rhf_two_electron(
      const linalg::Matrix& spin_summed_density,
      ScreeningOptions screening = {},
      ParallelBuildOptions parallel = {}) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> implementation_;
};

}  // namespace modernqc::integrals
