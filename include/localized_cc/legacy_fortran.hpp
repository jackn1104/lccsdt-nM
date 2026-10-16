#pragma once

#include "localized_cc/tensor.hpp"

namespace localized_cc {

// The stable C ABI isolates C++ ownership from the Fortran numerical kernels.
// This first boundary covers energy contractions; the distributed CCSD
// iteration and streamed triples driver will be added behind the same module.
[[nodiscard]] bool legacy_fortran_backend_available() noexcept;

// distributed=true is collective over parallel::world(); every rank must call
// it. Local solver and rank-subset call sites must pass false.
double legacy_fortran_ccsd_energy(const Matrix& fock, const Tensor4& eri,
                                  const Matrix& t1, const Tensor4& t2,
                                  std::size_t occupied, bool distributed);

double legacy_fortran_triples_energy(const Tensor6& left_moment,
                                     const Tensor6& response,
                                     bool distributed);

Tensor4 legacy_fortran_wabef(const Tensor4& eri, const Matrix& t1,
                             const Tensor4& tau, std::size_t occupied,
                             bool distributed);

void legacy_fortran_wabef_block(const Tensor4& eri, const Matrix& t1,
                                const Tensor4& tau, std::size_t occupied,
                                std::size_t a, std::size_t b, Matrix& output);

// Distributed tau supplies only tau(m,n,a,b) for one external (a,b) task.
// tau_ab is row-major (m,n), and the output is row-major (e,q).
void legacy_fortran_wabef_block_from_pair_scalars(
    const Tensor4& eri, const Matrix& t1, const Matrix& tau_ab,
    std::size_t occupied, std::size_t a, std::size_t b, Matrix& output);

} // namespace localized_cc
