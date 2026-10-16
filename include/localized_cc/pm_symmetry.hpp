#pragma once

#include "localized_cc/frozen_core.hpp"
#include <array>
#include <vector>
#include <utility>

namespace localized_cc {

// Real, spatial active orbitals: occupied followed by virtual. No rotations
// between orbitals are approximated by permutations. This path supports
// cyclic order-two, order-three and order-four groups only.
struct PmSymmetryOperation {
  std::vector<std::size_t> image;
  std::vector<int> sign;
};
struct PmSymmetry {
  std::size_t occupied{};
  std::size_t virtuals{};
  std::vector<PmSymmetryOperation> operations;
  double maximum_hamiltonian_error{};
};

// Complete O,H,H water cycles of order 3 or 4, including all hydrogen
// distances and atom-map closure. Coordinates in bohr; no symmetrization.
std::vector<int> find_cyclic_water_cycle(
    const std::vector<std::array<double, 3>>& positions,
    const std::vector<int>& atomic_numbers);

// O,H,H ordering; coordinates in bohr. Finds a genuine four-water cycle
// preserving ALL interatomic distances, including hydrogens. No snapping.
std::array<int, 4> find_w4_water_cycle(
    const std::vector<std::array<double, 3>>& positions,
    const std::vector<int>& atomic_numbers);

// Conservative candidate matching by local orbital invariants, followed by
// exhaustive signed Fock/hcore/ERI covariance validation. Ambiguous or
// symmetry-broken PM orbitals fail closed; no coefficients are changed.
PmSymmetry find_w4_pm_symmetry(
    const SpatialRhfHamiltonian& h,
    const std::vector<int>& occupied_monomers,
    const std::vector<int>& virtual_monomers,
    const std::array<int, 4>& water_cycle);
void validate_pm_symmetry(const SpatialRhfHamiltonian& h,
                          const PmSymmetry& symmetry);
void validate_pm_symmetry_group(const PmSymmetry& symmetry);
void validate_pm_symmetry_amplitudes(const PmSymmetry& symmetry,
                                     const Matrix& spin_t1,
                                     const Tensor4& spin_t2);
void validate_pm_spatial_amplitudes(const PmSymmetry& symmetry,
                                    const Matrix& t1, const Tensor4& t2);
void validate_pm_symmetry_labels(const PmSymmetry& symmetry,
                                 const std::vector<int>& occupied_labels,
                                 const std::vector<int>& virtual_labels);
std::size_t pm_spin_image(const PmSymmetryOperation& operation,
                          std::size_t spin_index, std::size_t spatial_offset);
int pm_spin_sign(const PmSymmetryOperation& operation,
                 std::size_t spin_index, std::size_t spatial_offset);
// Map a sorted unique spin triple; the returned sign includes sorting parity.
std::pair<std::array<std::size_t, 3>, int> pm_map_triple(
    const PmSymmetryOperation& operation,
    std::array<std::size_t, 3> triple, std::size_t spatial_offset);
} // namespace localized_cc
