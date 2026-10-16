#pragma once

#include "localized_cc/frozen_core.hpp"
#include "localized_cc/tensor.hpp"

namespace localized_cc {

// Restricted closed-shell spatial amplitudes. T2 uses the spin-free
// convention T(i,j,a,b) = T(j,i,b,a); it is not antisymmetric in either pair.
struct SpatialRccsdAmplitudes {
  Matrix t1;
  Tensor4 t2;
};

[[nodiscard]] double spatial_rccsd_pair_symmetry_error(const Tensor4& t2);

// Exact reference adapters used to compare the spatial convention with the
// retained interleaved spin-orbital equations.
[[nodiscard]] Matrix expand_spatial_rccsd_t1_to_spin_orbital(
    const Matrix& spatial_t1);
[[nodiscard]] Tensor4 expand_spatial_rccsd_t2_to_spin_orbital(
    const Tensor4& spatial_t2);

// E_corr = 2 f_ia t_i^a + [2(ia|jb)-(ib|ja)]
//          [T_ij^ab + t_i^a t_j^b].
[[nodiscard]] double spatial_rccsd_energy(
    const SpatialRhfHamiltonian& hamiltonian,
    const SpatialRccsdAmplitudes& amplitudes);

} // namespace localized_cc
