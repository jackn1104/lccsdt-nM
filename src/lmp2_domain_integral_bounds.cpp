#include "localized_cc/lmp2_preintegral.hpp"
#include <libint2/basis.h>
#include <libint2/engine.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace localized_cc {
Matrix lmp2_ao_schwarz_pair_bounds(
    const modernqc::molecule::Molecule& molecule,
    const modernqc::integrals::Libint2IntegralProvider& provider,
    const modernqc::linalg::Matrix& co,const modernqc::linalg::Matrix& cv) {
  const auto& meta=provider.basis_metadata();
  if(co.rows()!=meta.number_of_aos||cv.rows()!=co.rows()||!co.columns()||!cv.columns())
    throw std::invalid_argument("Schwarz/PM coefficient dimension mismatch");
  modernqc::linalg::require_finite(co,"PM occupied coefficients");
  modernqc::linalg::require_finite(cv,"PM virtual coefficients");
  std::vector<libint2::Atom> atoms;
  for(const auto& a:molecule.atoms())
    atoms.push_back({a.atomic_number,a.position_bohr[0],a.position_bohr[1],a.position_bohr[2]});
  // Recreate only the provider's basis, not an AO ERI tensor. Its runtime
  // remains alive throughout this call. Validate ordering and coefficients.
  libint2::BasisSet basis{meta.canonical_name,atoms,true};basis.set_pure(meta.spherical);
  if(basis.size()!=meta.shells.size()||static_cast<std::size_t>(basis.nbf())!=co.rows())
    throw std::runtime_error("Schwarz basis disagrees with integral provider");
  for(std::size_t s=0;s<basis.size();++s) {
    const auto& x=basis[s];const auto& y=meta.shells[s];
    if(x.contr.size()!=1||basis.shell2bf()[s]!=y.first_ao||x.size()!=y.function_count||
       x.contr[0].l!=y.angular_momentum||x.contr[0].pure!=y.spherical||
       std::vector<double>(x.alpha.begin(),x.alpha.end())!=y.exponents||
       std::vector<double>(x.contr[0].coeff.begin(),x.contr[0].coeff.end())!=y.contraction_coefficients)
      throw std::runtime_error("Schwarz shell ordering/normalization mismatch");
  }
  using Dense=modernqc::linalg::Matrix;
  Dense q(basis.size(),basis.size()),so(basis.size(),co.columns()),sv(basis.size(),cv.columns());
  libint2::Engine engine{libint2::Operator::coulomb,basis.max_nprim(),static_cast<int>(basis.max_l()),0,0.0};
  for(std::size_t s=0;s<basis.size();++s) {
    const auto& shell=meta.shells[s];
    for(std::size_t mu=shell.first_ao;mu<shell.first_ao+shell.function_count;++mu) {
      for(std::size_t i=0;i<co.columns();++i) so(s,i)+=std::abs(co(mu,i));
      for(std::size_t a=0;a<cv.columns();++a) sv(s,a)+=std::abs(cv(mu,a));
    }
    for(std::size_t t=0;t<=s;++t) {
      engine.compute(basis[s],basis[t],basis[s],basis[t]);
      const auto* buffer=engine.results()[0];
      if(!buffer) throw std::runtime_error("no-screen Schwarz self quartet is null");
      double diagonal=0;
      const auto ns=basis[s].size(),nt=basis[t].size();
      for(std::size_t mu=0;mu<ns;++mu) for(std::size_t nu=0;nu<nt;++nu) {
        const auto value=buffer[((mu*nt+nu)*ns+mu)*nt+nu];
        if(!std::isfinite(value)) throw std::runtime_error("nonfinite Schwarz self integral");
        diagonal=std::max(diagonal,std::abs(value));
      }
      q(s,t)=q(t,s)=std::sqrt(diagonal)*(1+1e-12);
    }
  }
  const auto bound=modernqc::linalg::multiply(modernqc::linalg::transpose(so),modernqc::linalg::multiply(q,sv));
  Matrix result({co.columns(),cv.columns()});
  for(std::size_t i=0;i<co.columns();++i) for(std::size_t a=0;a<cv.columns();++a)
    result(i,a)=bound(i,a)*(1+1e-12);
  return result;
}
} // namespace localized_cc
