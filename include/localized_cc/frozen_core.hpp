#pragma once

#include "localized_cc/tensor.hpp"
#include <array>
#include <cstddef>
#include <vector>

namespace localized_cc {

struct FrozenCoreInput {
  Matrix C_core;
  Matrix C_occ_localized;
  Matrix C_virtual_localized;
  Matrix overlap_ao;
  Matrix hcore_ao;
  Tensor4 eri_ao_chemist;
  double nuclear_energy{};
  double rhf_energy{};
  bool validate_rhf_energy{true};
};

// Full-MO input used by bounded AO-to-MO backends.  The coefficient order is
// frozen core, active occupied, then virtual.  eri_mo_chemist stores exact
// (pq|rs) integrals in that order; no AO four-index tensor is required.
struct FrozenCoreMoInput {
  Matrix hcore_mo;
  SymmetricEri4 eri_mo_chemist;
  std::size_t frozen_core{};
  std::size_t active_occupied{};
  std::size_t virtuals{};
  double nuclear_energy{};
  double rhf_energy{};
  double orthonormality_error{};
  bool validate_rhf_energy{true};
};

struct FrozenCoreHamiltonian {
  Matrix fock;
  Matrix hcore;
  Tensor4 eri;
  std::size_t nocc{};
  std::size_t nvirt{};
  double reference_energy{};
  double nuclear_energy{};
  double orthonormality_error{};
  std::vector<std::size_t> core_indices;
  std::vector<std::size_t> active_occ_indices;
  std::vector<std::size_t> virtual_indices;
};

// Closed-shell spatial-orbital Hamiltonian. eri_chemist(p,q,r,s) stores
// (pq|rs); fock includes Coulomb/exchange from frozen and active doubly
// occupied spatial orbitals. Working indices contain active occupied first,
// followed by virtual orbitals.
struct SpatialRhfHamiltonian {
  Matrix fock;
  Matrix hcore;
  SymmetricEri4 eri_chemist;
  std::size_t nocc{};
  std::size_t nvirt{};
  std::size_t frozen_core{};
  double reference_energy{};
  double nuclear_energy{};
  double orthonormality_error{};
  std::vector<std::size_t> core_indices;
  std::vector<std::size_t> active_occ_indices;
  std::vector<std::size_t> virtual_indices;
};

// Read-only antisymmetrized spin-orbital ERIs synthesized from a closed-shell
// spatial tensor. Spin orbitals are interleaved alpha/beta. This view retains
// only the n^4 spatial chemists' integrals instead of materializing the 16 n^4
// spin-orbital reference tensor.
class SpinOrbitalEriView {
public:
  explicit SpinOrbitalEriView(const SpatialRhfHamiltonian& spatial);

  [[nodiscard]] std::array<std::size_t, 4> dimensions() const noexcept;
  [[nodiscard]] double operator()(std::size_t p, std::size_t q,
                                  std::size_t r,
                                  std::size_t s) const noexcept;
  [[nodiscard]] std::size_t spatial_elements() const;
  [[nodiscard]] std::size_t spatial_stored_elements() const noexcept;
  [[nodiscard]] std::size_t spin_reference_elements() const;
  [[nodiscard]] bool all_finite() const noexcept;

private:
  const SymmetricEri4* eri_chemist_{};
  std::size_t spatial_orbitals_{};
};

SpatialRhfHamiltonian build_frozen_core_spatial_hamiltonian(
    const FrozenCoreInput& input,
    double orthonormality_tolerance = 1.0e-10,
    double reference_tolerance = 1.0e-8);

SpatialRhfHamiltonian assemble_frozen_core_spatial_hamiltonian(
    FrozenCoreMoInput input, double reference_tolerance = 1.0e-8);

// Expand only a spatial one-body matrix. Unlike the reference Hamiltonian
// adapter below, this does not allocate a spin-orbital four-index tensor.
Matrix expand_spatial_one_body_to_spin_orbital(const Matrix& spatial);

// Reference adapter for the transitional spin-orbital CCSD(T) equations.
// Spin orbitals are interleaved alpha/beta for every spatial orbital.
FrozenCoreHamiltonian expand_spatial_hamiltonian_to_spin_orbital(
    const SpatialRhfHamiltonian& spatial);

FrozenCoreHamiltonian build_frozen_core_hamiltonian(
    const FrozenCoreInput& input,
    double orthonormality_tolerance = 1.0e-10,
    double reference_tolerance = 1.0e-8);

} // namespace localized_cc
