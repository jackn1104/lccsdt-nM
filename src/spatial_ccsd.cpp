#include "localized_cc/spatial_ccsd.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace localized_cc {
namespace {

void validate_spatial_amplitudes(const Matrix& t1, const Tensor4& t2) {
  const auto occupied = t1.dimensions()[0];
  const auto virtuals = t1.dimensions()[1];
  if (occupied == 0 || virtuals == 0 ||
      t2.dimensions() !=
          std::array<std::size_t, 4>{occupied, occupied, virtuals, virtuals})
    throw std::invalid_argument("inconsistent spatial RCCSD amplitudes");
  for (const auto value : t1.data())
    if (!std::isfinite(value))
      throw std::invalid_argument("non-finite spatial RCCSD T1 amplitude");
  for (const auto value : t2.data())
    if (!std::isfinite(value))
      throw std::invalid_argument("non-finite spatial RCCSD T2 amplitude");
}

} // namespace

double spatial_rccsd_pair_symmetry_error(const Tensor4& t2) {
  const auto dimensions = t2.dimensions();
  if (dimensions[0] != dimensions[1] || dimensions[2] != dimensions[3])
    throw std::invalid_argument("spatial RCCSD T2 dimensions mismatch");
  double error = 0.0;
  for (std::size_t i = 0; i < dimensions[0]; ++i)
    for (std::size_t j = 0; j < dimensions[1]; ++j)
      for (std::size_t a = 0; a < dimensions[2]; ++a)
        for (std::size_t b = 0; b < dimensions[3]; ++b)
          error = std::max(
              error, std::abs(t2(i, j, a, b) - t2(j, i, b, a)));
  return error;
}

Matrix expand_spatial_rccsd_t1_to_spin_orbital(const Matrix& spatial_t1) {
  const auto occupied = spatial_t1.dimensions()[0];
  const auto virtuals = spatial_t1.dimensions()[1];
  if (occupied == 0 || virtuals == 0 ||
      occupied > std::numeric_limits<std::size_t>::max() / 2 ||
      virtuals > std::numeric_limits<std::size_t>::max() / 2)
    throw std::invalid_argument("invalid spatial RCCSD T1 dimensions");
  for (const auto value : spatial_t1.data())
    if (!std::isfinite(value))
      throw std::invalid_argument("non-finite spatial RCCSD T1 amplitude");
  Matrix spin_t1({2 * occupied, 2 * virtuals});
  for (std::size_t i = 0; i < 2 * occupied; ++i)
    for (std::size_t a = 0; a < 2 * virtuals; ++a)
      if (i % 2 == a % 2)
        spin_t1(i, a) = spatial_t1(i / 2, a / 2);
  return spin_t1;
}

Tensor4 expand_spatial_rccsd_t2_to_spin_orbital(
    const Tensor4& spatial_t2) {
  const auto dimensions = spatial_t2.dimensions();
  if (dimensions[0] == 0 || dimensions[2] == 0 ||
      dimensions[0] != dimensions[1] || dimensions[2] != dimensions[3] ||
      dimensions[0] > std::numeric_limits<std::size_t>::max() / 2 ||
      dimensions[2] > std::numeric_limits<std::size_t>::max() / 2)
    throw std::invalid_argument("invalid spatial RCCSD T2 dimensions");
  for (const auto value : spatial_t2.data())
    if (!std::isfinite(value))
      throw std::invalid_argument("non-finite spatial RCCSD T2 amplitude");
  if (spatial_rccsd_pair_symmetry_error(spatial_t2) > 1.0e-12)
    throw std::invalid_argument(
        "spatial RCCSD T2 violates T(i,j,a,b)=T(j,i,b,a)");
  const auto occupied = dimensions[0];
  const auto virtuals = dimensions[2];
  Tensor4 spin_t2(
      {2 * occupied, 2 * occupied, 2 * virtuals, 2 * virtuals});
  for (std::size_t i = 0; i < 2 * occupied; ++i)
    for (std::size_t j = 0; j < 2 * occupied; ++j)
      for (std::size_t a = 0; a < 2 * virtuals; ++a)
        for (std::size_t b = 0; b < 2 * virtuals; ++b) {
          double value = 0.0;
          if (i % 2 == a % 2 && j % 2 == b % 2)
            value += spatial_t2(i / 2, j / 2, a / 2, b / 2);
          if (i % 2 == b % 2 && j % 2 == a % 2)
            value -= spatial_t2(i / 2, j / 2, b / 2, a / 2);
          spin_t2(i, j, a, b) = value;
        }
  return spin_t2;
}

double spatial_rccsd_energy(const SpatialRhfHamiltonian& hamiltonian,
                            const SpatialRccsdAmplitudes& amplitudes) {
  validate_spatial_amplitudes(amplitudes.t1, amplitudes.t2);
  const auto occupied = amplitudes.t1.dimensions()[0];
  const auto virtuals = amplitudes.t1.dimensions()[1];
  const auto orbitals = occupied + virtuals;
  if (hamiltonian.nocc != occupied || hamiltonian.nvirt != virtuals ||
      hamiltonian.fock.dimensions() !=
          std::array<std::size_t, 2>{orbitals, orbitals} ||
      hamiltonian.eri_chemist.dimensions() !=
          std::array<std::size_t, 4>{orbitals, orbitals, orbitals, orbitals})
    throw std::invalid_argument("spatial RCCSD Hamiltonian dimensions mismatch");
  if (spatial_rccsd_pair_symmetry_error(amplitudes.t2) > 1.0e-12)
    throw std::invalid_argument(
        "spatial RCCSD T2 violates T(i,j,a,b)=T(j,i,b,a)");
  double energy = 0.0;
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t a = 0; a < virtuals; ++a)
      energy += 2.0 * hamiltonian.fock(i, occupied + a) *
                amplitudes.t1(i, a);
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t j = 0; j < occupied; ++j)
      for (std::size_t a = 0; a < virtuals; ++a)
        for (std::size_t b = 0; b < virtuals; ++b) {
          const auto direct = hamiltonian.eri_chemist(
              i, occupied + a, j, occupied + b);
          const auto exchange = hamiltonian.eri_chemist(
              i, occupied + b, j, occupied + a);
          energy += (2.0 * direct - exchange) *
                    (amplitudes.t2(i, j, a, b) +
                     amplitudes.t1(i, a) * amplitudes.t1(j, b));
        }
  return energy;
}

} // namespace localized_cc
