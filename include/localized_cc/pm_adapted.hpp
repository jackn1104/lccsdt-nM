#pragma once
#include "localized_cc/pm_symmetry.hpp"
#include "modernqc/integrals/integral_provider.hpp"
#include <functional>
#include <string>

namespace localized_cc {
// Audit existing coefficients only: no localization, projection, or snapping.
modernqc::linalg::Matrix inversion_ao_action(
    const modernqc::integrals::Libint2IntegralProvider& provider,
    const modernqc::molecule::Molecule& molecule,
    const modernqc::linalg::Matrix& overlap);
PmSymmetry match_pm_inversion(
    const modernqc::linalg::Matrix& occupied,
    const modernqc::linalg::Matrix& virtuals,
    const modernqc::linalg::Matrix& overlap,
    const modernqc::linalg::Matrix& ao_action,
    const std::function<void(const std::string&)>& progress);
struct AdaptedPmResult {
  modernqc::linalg::Matrix coefficients;
  modernqc::linalg::Matrix rotation;
  double objective{};
  double gradient_norm{};
  int iterations{};
};
// Cyclic order 3/4 implementation. W4 wrappers below preserve existing callers.
modernqc::linalg::Matrix cyclic_ao_symmetry_action(
    const modernqc::integrals::Libint2IntegralProvider& provider,
    const modernqc::molecule::Molecule& molecule,
    const modernqc::linalg::Matrix& overlap, const std::vector<int>& cycle);
AdaptedPmResult adapt_cyclic_pm(
    const modernqc::linalg::Matrix& canonical,
    const modernqc::linalg::Matrix& seed,
    const std::vector<int>& seed_labels,
    const modernqc::linalg::Matrix& overlap,
    const modernqc::linalg::Matrix& ao_action,
    const modernqc::linalg::Matrix& meta_lowdin,
    const std::vector<std::size_t>& ao_atoms, std::size_t order,
    const std::function<void(const std::string&)>& progress);
PmSymmetry cyclic_adapted_permutation(std::size_t occupied,
                                     std::size_t virtuals, std::size_t order);
// Construct the AO action using shell-local collocation, independently checked
// on a second point set and in the AO overlap metric. No integral screening.
modernqc::linalg::Matrix w4_ao_symmetry_action(
    const modernqc::integrals::Libint2IntegralProvider& provider,
    const modernqc::molecule::Molecule& molecule,
    const modernqc::linalg::Matrix& overlap, const std::array<int,4>& cycle);
AdaptedPmResult adapt_w4_pm(
    const modernqc::linalg::Matrix& canonical,
    const modernqc::linalg::Matrix& seed,
    const std::vector<int>& seed_labels,
    const modernqc::linalg::Matrix& overlap,
    const modernqc::linalg::Matrix& ao_action,
    const modernqc::linalg::Matrix& meta_lowdin,
    const std::vector<std::size_t>& ao_atoms,
    const std::function<void(const std::string&)>& progress);
PmSymmetry w4_adapted_permutation(std::size_t occupied, std::size_t virtuals);
} // namespace localized_cc
