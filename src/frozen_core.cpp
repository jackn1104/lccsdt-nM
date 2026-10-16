#include "localized_cc/frozen_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace localized_cc {
namespace {

std::size_t rows(const Matrix& x) { return x.dimensions()[0]; }
std::size_t cols(const Matrix& x) { return x.dimensions()[1]; }

Matrix concatenate(const Matrix& a, const Matrix& b, const Matrix& c) {
  const auto coefficient_rows = rows(b);
  const bool empty_core = rows(a) == 0 && cols(a) == 0;
  if ((!empty_core && rows(a) != coefficient_rows) ||
      rows(c) != coefficient_rows)
    throw std::invalid_argument("MO coefficient row mismatch");
  Matrix result({coefficient_rows, cols(a) + cols(b) + cols(c)});
  for (std::size_t u = 0; u < coefficient_rows; ++u) {
    for (std::size_t p = 0; p < cols(a); ++p) result(u, p) = a(u, p);
    for (std::size_t p = 0; p < cols(b); ++p) result(u, cols(a) + p) = b(u, p);
    for (std::size_t p = 0; p < cols(c); ++p) result(u, cols(a) + cols(b) + p) = c(u, p);
  }
  return result;
}

Matrix transform_one_body(const Matrix& ao, const Matrix& c) {
  const auto nao = rows(c), nmo = cols(c);
  Matrix result({nmo, nmo});
  for (std::size_t p = 0; p < nmo; ++p)
    for (std::size_t q = 0; q < nmo; ++q)
      for (std::size_t u = 0; u < nao; ++u)
        for (std::size_t v = 0; v < nao; ++v) result(p, q) += c(u, p) * ao(u, v) * c(v, q);
  return result;
}

SymmetricEri4 transform_eri(const Tensor4& ao, const Matrix& c) {
  const auto nao = rows(c), nmo = cols(c);
  if (ao.dimensions() != std::array<std::size_t, 4>{nao, nao, nao, nao})
    throw std::invalid_argument("AO ERI dimensions do not match coefficients");
  Tensor4 t1({nmo, nao, nao, nao});
  for (std::size_t p=0;p<nmo;++p) for(std::size_t v=0;v<nao;++v)
    for(std::size_t w=0;w<nao;++w) for(std::size_t x=0;x<nao;++x)
      for(std::size_t u=0;u<nao;++u) t1(p,v,w,x)+=c(u,p)*ao(u,v,w,x);
  Tensor4 t2({nmo,nmo,nao,nao});
  for(std::size_t p=0;p<nmo;++p) for(std::size_t q=0;q<nmo;++q)
    for(std::size_t w=0;w<nao;++w) for(std::size_t x=0;x<nao;++x)
      for(std::size_t v=0;v<nao;++v) t2(p,q,w,x)+=c(v,q)*t1(p,v,w,x);
  Tensor4 t3({nmo,nmo,nmo,nao});
  for(std::size_t p=0;p<nmo;++p) for(std::size_t q=0;q<nmo;++q)
    for(std::size_t r=0;r<nmo;++r) for(std::size_t x=0;x<nao;++x)
      for(std::size_t w=0;w<nao;++w) t3(p,q,r,x)+=c(w,r)*t2(p,q,w,x);
  Tensor4 dense_result({nmo,nmo,nmo,nmo});
  for(std::size_t p=0;p<nmo;++p) for(std::size_t q=0;q<nmo;++q)
    for(std::size_t r=0;r<nmo;++r) for(std::size_t s=0;s<nmo;++s)
      for(std::size_t x=0;x<nao;++x)
        dense_result(p,q,r,s)+=c(x,s)*t3(p,q,r,x);
  SymmetricEri4 result(nmo);
  std::size_t first_pair = 0;
  for (std::size_t p = 0; p < nmo; ++p)
    for (std::size_t q = 0; q <= p; ++q, ++first_pair) {
      std::size_t second_pair = 0;
      for (std::size_t r = 0; r < nmo; ++r)
        for (std::size_t s = 0; s <= r; ++s, ++second_pair)
          if (second_pair <= first_pair)
            result(p, q, r, s) = dense_result(p, q, r, s);
    }
  return result;
}

} // namespace

SpinOrbitalEriView::SpinOrbitalEriView(
    const SpatialRhfHamiltonian& spatial)
    : eri_chemist_(&spatial.eri_chemist) {
  if (spatial.nvirt >
      std::numeric_limits<std::size_t>::max() - spatial.nocc)
    throw std::overflow_error("spatial RHF orbital count overflows");
  spatial_orbitals_ = spatial.nocc + spatial.nvirt;
  if (spatial_orbitals_ == 0 ||
      spatial_orbitals_ > std::numeric_limits<std::size_t>::max() / 2)
    throw std::overflow_error(
        "spatial RHF orbital count cannot be viewed as spin orbitals");
  if (spatial.eri_chemist.dimensions() !=
      std::array<std::size_t, 4>{spatial_orbitals_, spatial_orbitals_,
                                 spatial_orbitals_, spatial_orbitals_})
    throw std::invalid_argument("spatial RHF ERI dimensions mismatch");
  if (!all_finite())
    throw std::invalid_argument("non-finite spatial RHF ERI tensor");
}

std::array<std::size_t, 4> SpinOrbitalEriView::dimensions() const noexcept {
  const auto spin_orbitals = 2 * spatial_orbitals_;
  return {spin_orbitals, spin_orbitals, spin_orbitals, spin_orbitals};
}

double SpinOrbitalEriView::operator()(const std::size_t p,
                                     const std::size_t q,
                                     const std::size_t r,
                                     const std::size_t s) const noexcept {
  double value = 0.0;
  if (p % 2 == r % 2 && q % 2 == s % 2)
    value += (*eri_chemist_)(p / 2, r / 2, q / 2, s / 2);
  if (p % 2 == s % 2 && q % 2 == r % 2)
    value -= (*eri_chemist_)(p / 2, s / 2, q / 2, r / 2);
  return value;
}

std::size_t SpinOrbitalEriView::spatial_elements() const {
  return eri_chemist_->logical_size();
}

std::size_t SpinOrbitalEriView::spatial_stored_elements() const noexcept {
  return eri_chemist_->size();
}

std::size_t SpinOrbitalEriView::spin_reference_elements() const {
  if (spatial_elements() > std::numeric_limits<std::size_t>::max() / 16)
    throw std::overflow_error("spin-reference ERI element count overflows");
  return 16 * spatial_elements();
}

bool SpinOrbitalEriView::all_finite() const noexcept {
  for (const auto value : eri_chemist_->data())
    if (!std::isfinite(value))
      return false;
  return true;
}

Matrix expand_spatial_one_body_to_spin_orbital(const Matrix& spatial) {
  const auto dimensions = spatial.dimensions();
  if (dimensions[0] == 0 || dimensions[1] != dimensions[0] ||
      dimensions[0] > std::numeric_limits<std::size_t>::max() / 2)
    throw std::invalid_argument(
        "spatial one-body matrix must be nonempty and square");
  for (const auto value : spatial.data())
    if (!std::isfinite(value))
      throw std::invalid_argument("non-finite spatial one-body matrix");
  const auto spin_orbitals = 2 * dimensions[0];
  Matrix result({spin_orbitals, spin_orbitals});
  for (std::size_t p = 0; p < spin_orbitals; ++p)
    for (std::size_t q = 0; q < spin_orbitals; ++q)
      if (p % 2 == q % 2)
        result(p, q) = spatial(p / 2, q / 2);
  return result;
}

SpatialRhfHamiltonian assemble_frozen_core_spatial_hamiltonian(
    FrozenCoreMoInput in, const double ref_tol) {
  if (in.frozen_core >
          std::numeric_limits<std::size_t>::max() - in.active_occupied ||
      in.frozen_core + in.active_occupied >
          std::numeric_limits<std::size_t>::max() - in.virtuals)
    throw std::overflow_error("full-MO orbital count overflows");
  const auto nc = in.frozen_core;
  const auto no = in.active_occupied;
  const auto nv = in.virtuals;
  const auto nmo = nc + no + nv;
  if (nmo == 0 ||
      in.hcore_mo.dimensions() != std::array<std::size_t, 2>{nmo, nmo} ||
      in.eri_mo_chemist.dimensions() !=
          std::array<std::size_t, 4>{nmo, nmo, nmo, nmo})
    throw std::invalid_argument("full-MO Hamiltonian dimensions mismatch");
  if (!std::isfinite(in.nuclear_energy) || !std::isfinite(in.rhf_energy) ||
      !std::isfinite(in.orthonormality_error) ||
      in.orthonormality_error < 0.0 || ref_tol < 0.0 ||
      !std::isfinite(ref_tol))
    throw std::invalid_argument("invalid full-MO Hamiltonian metadata");
  for (const auto value : in.hcore_mo.data())
    if (!std::isfinite(value))
      throw std::invalid_argument("non-finite full-MO core Hamiltonian");
  for (const auto value : in.eri_mo_chemist.data())
    if (!std::isfinite(value))
      throw std::invalid_argument("non-finite full-MO ERI tensor");

  Matrix f_spatial({nmo, nmo});
  for (std::size_t p = 0; p < nmo; ++p)
    for (std::size_t q = 0; q < nmo; ++q) {
      f_spatial(p, q) = in.hcore_mo(p, q);
      for (std::size_t i = 0; i < nc + no; ++i)
        f_spatial(p, q) +=
            2.0 * in.eri_mo_chemist(p, q, i, i) -
            in.eri_mo_chemist(p, i, i, q);
    }
  double eref = in.nuclear_energy;
  for (std::size_t i = 0; i < nc + no; ++i)
    eref += 2.0 * in.hcore_mo(i, i);
  for (std::size_t i = 0; i < nc + no; ++i)
    for (std::size_t j = 0; j < nc + no; ++j)
      eref += 2.0 * in.eri_mo_chemist(i, i, j, j) -
              in.eri_mo_chemist(i, j, j, i);
  if (in.validate_rhf_energy && std::abs(eref - in.rhf_energy) > ref_tol)
    throw std::runtime_error(
        "reconstructed frozen-core reference energy disagrees with supplied "
        "RHF energy");

  Matrix fw({no + nv, no + nv}), hw({no + nv, no + nv});
  SymmetricEri4 ew(no + nv);
  for (std::size_t p = 0; p < no + nv; ++p)
    for (std::size_t q = 0; q < no + nv; ++q) {
      fw(p, q) = f_spatial(nc + p, nc + q);
      hw(p, q) = in.hcore_mo(nc + p, nc + q);
    }
  std::size_t first_pair = 0;
  for (std::size_t p = 0; p < no + nv; ++p)
    for (std::size_t q = 0; q <= p; ++q, ++first_pair) {
      std::size_t second_pair = 0;
      for (std::size_t r = 0; r < no + nv; ++r)
        for (std::size_t s = 0; s <= r; ++s, ++second_pair)
          if (second_pair <= first_pair)
            ew(p, q, r, s) = in.eri_mo_chemist(
                nc + p, nc + q, nc + r, nc + s);
    }
  SpatialRhfHamiltonian out{
      std::move(fw), std::move(hw), std::move(ew), no, nv, nc, eref,
      in.nuclear_energy, in.orthonormality_error, {}, {}, {}};
  for (std::size_t x = 0; x < nc; ++x) out.core_indices.push_back(x);
  for (std::size_t x = nc; x < nc + no; ++x)
    out.active_occ_indices.push_back(x);
  for (std::size_t x = nc + no; x < nc + no + nv; ++x)
    out.virtual_indices.push_back(x);
  return out;
}

SpatialRhfHamiltonian build_frozen_core_spatial_hamiltonian(
    const FrozenCoreInput& in, const double ortho_tol,
    const double ref_tol) {
  const auto nao = rows(in.overlap_ao);
  if (cols(in.overlap_ao) != nao || in.hcore_ao.dimensions() != in.overlap_ao.dimensions())
    throw std::invalid_argument("AO overlap and core Hamiltonian must be equally sized square matrices");
  const auto c = concatenate(in.C_core, in.C_occ_localized, in.C_virtual_localized);
  const auto nmo = cols(c);
  double ortho_error = 0.0;
  for(std::size_t p=0;p<nmo;++p) for(std::size_t q=0;q<nmo;++q) {
    double value=0.0;
    for(std::size_t u=0;u<nao;++u) for(std::size_t v=0;v<nao;++v)
      value += c(u,p)*in.overlap_ao(u,v)*c(v,q);
    ortho_error=std::max(ortho_error,std::abs(value-(p==q?1.0:0.0)));
  }
  if (ortho_error > ortho_tol) throw std::invalid_argument("localized orbitals are not S-orthonormal");
  auto h_spatial=transform_one_body(in.hcore_ao,c);
  auto e_spatial=transform_eri(in.eri_ao_chemist,c);
  const auto nc=cols(in.C_core), no=cols(in.C_occ_localized);
  const auto nv=cols(in.C_virtual_localized);
  return assemble_frozen_core_spatial_hamiltonian(
      FrozenCoreMoInput{std::move(h_spatial), std::move(e_spatial), nc, no,
                        nv, in.nuclear_energy, in.rhf_energy, ortho_error,
                        in.validate_rhf_energy},
      ref_tol);
}

FrozenCoreHamiltonian expand_spatial_hamiltonian_to_spin_orbital(
    const SpatialRhfHamiltonian& spatial) {
  if(spatial.nvirt>std::numeric_limits<std::size_t>::max()-spatial.nocc)
    throw std::overflow_error("spatial RHF orbital count overflows");
  const auto nmo=spatial.nocc+spatial.nvirt;
  if(nmo==0 || nmo>std::numeric_limits<std::size_t>::max()/2)
    throw std::overflow_error("spatial RHF orbital count cannot be spin expanded");
  if(spatial.fock.dimensions()!=std::array<std::size_t,2>{nmo,nmo} ||
     spatial.hcore.dimensions()!=std::array<std::size_t,2>{nmo,nmo} ||
     spatial.eri_chemist.dimensions()!=
         std::array<std::size_t,4>{nmo,nmo,nmo,nmo})
    throw std::invalid_argument("spatial RHF Hamiltonian dimensions mismatch");
  if(spatial.core_indices.size()!=spatial.frozen_core ||
     spatial.active_occ_indices.size()!=spatial.nocc ||
     spatial.virtual_indices.size()!=spatial.nvirt)
    throw std::invalid_argument("spatial RHF orbital-index metadata mismatch");
  if(!std::isfinite(spatial.reference_energy) ||
     !std::isfinite(spatial.nuclear_energy) ||
     !std::isfinite(spatial.orthonormality_error))
    throw std::invalid_argument("non-finite spatial RHF metadata");
  for(const auto value:spatial.fock.data())
    if(!std::isfinite(value))
      throw std::invalid_argument("non-finite spatial RHF Fock matrix");
  for(const auto value:spatial.hcore.data())
    if(!std::isfinite(value))
      throw std::invalid_argument("non-finite spatial RHF core Hamiltonian");
  for(const auto value:spatial.eri_chemist.data())
    if(!std::isfinite(value))
      throw std::invalid_argument("non-finite spatial RHF ERI tensor");
  const auto nso=2*nmo;
  Matrix f=expand_spatial_one_body_to_spin_orbital(spatial.fock);
  Matrix h=expand_spatial_one_body_to_spin_orbital(spatial.hcore);
  Tensor4 g({nso,nso,nso,nso});
  const SpinOrbitalEriView eri_view(spatial);
  for(std::size_t p=0;p<nso;++p) for(std::size_t q=0;q<nso;++q)
    for(std::size_t r=0;r<nso;++r) for(std::size_t s=0;s<nso;++s) {
      g(p,q,r,s)=eri_view(p,q,r,s);
    }
  const auto nc=2*spatial.frozen_core;
  const auto no=2*spatial.nocc;
  const auto nv=2*spatial.nvirt;
  FrozenCoreHamiltonian out{std::move(f),std::move(h),std::move(g),no,nv,
                            spatial.reference_energy,spatial.nuclear_energy,
                            spatial.orthonormality_error,{},{},{}};
  for(std::size_t x=0;x<nc;++x) out.core_indices.push_back(x);
  for(std::size_t x=nc;x<nc+no;++x) out.active_occ_indices.push_back(x);
  for(std::size_t x=nc+no;x<nc+no+nv;++x) out.virtual_indices.push_back(x);
  return out;
}

FrozenCoreHamiltonian build_frozen_core_hamiltonian(
    const FrozenCoreInput& in, const double ortho_tol,
    const double ref_tol) {
  return expand_spatial_hamiltonian_to_spin_orbital(
      build_frozen_core_spatial_hamiltonian(in,ortho_tol,ref_tol));
}

} // namespace localized_cc
