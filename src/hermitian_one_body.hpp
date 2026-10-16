#pragma once

#include "localized_cc/frozen_core.hpp"
#include "modernqc/linalg/matrix.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace localized_cc::detail {

// This is an arithmetic error bound, not a configurable symmetry tolerance.
// Keep the AO-input and downstream eigensolver checks at 1e-12 Hartree.
struct HermitianProjection {
  Matrix value;
  modernqc::linalg::Matrix roundoff_bound;
};

inline HermitianProjection project_hermitian_one_body(
    const modernqc::linalg::Matrix& ao,
    const modernqc::linalg::Matrix& c) {
  const auto n = c.rows(), m = c.columns();
  if (!n || !m || ao.rows() != n || ao.columns() != n)
    throw std::invalid_argument("Hermitian one-body projection dimensions mismatch");
  modernqc::linalg::require_finite(ao, "AO one-body matrix");
  modernqc::linalg::require_finite(c, "one-body coefficients");
  if (modernqc::linalg::maximum_asymmetry(ao) > 1e-12)
    throw std::runtime_error("AO one-body input exceeds unchanged 1e-12 symmetry tolerance");
  auto symmetric = ao;
  modernqc::linalg::symmetrize_in_place(symmetric);
  auto absolute = ao;
  auto abs_c = c;
  modernqc::linalg::Matrix input_correction(n, n);
  for (std::size_t p=0;p<n;++p) {
    for (std::size_t q=0;q<n;++q) {
      absolute(p,q)=std::abs(ao(p,q));
      input_correction(p,q)=std::abs(symmetric(p,q)-ao(p,q));
    }
    for (std::size_t q=0;q<m;++q) abs_c(p,q)=std::abs(c(p,q));
  }
  const auto ct=modernqc::linalg::transpose(c);
  const auto abs_ct=modernqc::linalg::transpose(abs_c);
  const auto raw=modernqc::linalg::multiply(ct,
      modernqc::linalg::multiply(symmetric,c));
  auto bound=modernqc::linalg::multiply(abs_ct,
      modernqc::linalg::multiply(absolute,abs_c));
  const auto input_bound=modernqc::linalg::multiply(abs_ct,
      modernqc::linalg::multiply(input_correction,abs_c));
  // Covers two independent two-DGEMM projections, error in the absolute
  // majorant, and averaging. eps (rather than eps/2) is conservative.
  const double neps=(8.0*static_cast<double>(n)+16.0)*
      std::numeric_limits<double>::epsilon();
  if (!(neps < 0.01)) throw std::overflow_error("one-body rounding bound dimension");
  const double gamma=neps/(1.0-neps);
  HermitianProjection out{Matrix({m,m}), std::move(bound)};
  for (std::size_t p=0;p<m;++p) for (std::size_t q=0;q<m;++q) {
    out.roundoff_bound(p,q)=gamma*out.roundoff_bound(p,q)+input_bound(p,q);
    if (!std::isfinite(raw(p,q)) || !std::isfinite(out.roundoff_bound(p,q)))
      throw std::runtime_error("non-finite Hermitian projection or error bound");
  }
  for (std::size_t p=0;p<m;++p) for (std::size_t q=0;q<=p;++q) {
    if (std::abs(raw(p,q)-raw(q,p)) >
        out.roundoff_bound(p,q)+out.roundoff_bound(q,p))
      throw std::runtime_error("one-body projection asymmetry exceeds arithmetic error bound");
    const double value=std::midpoint(raw(p,q),raw(q,p));
    out.value(p,q)=out.value(q,p)=value;
  }
  return out;
}

struct FockRebuildDiagnostics {
  double original_asymmetry{};
  double hcore_maximum_change{};
  double fock_maximum_change{};
  double maximum_arithmetic_bound{};
};

// Legacy semidirect checkpoints hold bare active hcore plus a Fock matrix
// with frozen+active Coulomb/exchange. Reuse those two-electron contributions;
// do not re-evaluate AO ERIs, change T1/T2, or average arbitrary corrupt Fock.
inline FockRebuildDiagnostics rebuild_legacy_one_body(
    SpatialRhfHamiltonian& h, const modernqc::linalg::Matrix& ao_hcore,
    const modernqc::linalg::Matrix& coefficients) {
  const auto n=coefficients.columns();
  if (h.fock.dimensions()!=std::array<std::size_t,2>{n,n} ||
      h.hcore.dimensions()!=h.fock.dimensions())
    throw std::invalid_argument("legacy one-body rebuild dimensions mismatch");
  const auto projected=project_hermitian_one_body(ao_hcore,coefficients);
  FockRebuildDiagnostics d;
  Matrix new_fock({n,n});
  for (std::size_t p=0;p<n;++p) for (std::size_t q=0;q<n;++q) {
    if (!std::isfinite(h.fock(p,q)) || !std::isfinite(h.hcore(p,q)))
      throw std::runtime_error("non-finite legacy Hamiltonian matrix");
    const double error=std::abs(projected.value(p,q)-h.hcore(p,q));
    d.maximum_arithmetic_bound=std::max(d.maximum_arithmetic_bound,projected.roundoff_bound(p,q));
    d.hcore_maximum_change=std::max(d.hcore_maximum_change,error);
    if (error > projected.roundoff_bound(p,q)) {
      std::ostringstream message;
      message<<std::scientific<<std::setprecision(12)
          <<"legacy hcore disagrees with saved AO/PM projection at ("<<p<<","<<q
          <<"); error_Eh="<<error<<"; arithmetic_bound_Eh="<<projected.roundoff_bound(p,q)
          <<"; no repair applied";
      throw std::runtime_error(message.str());
    }
    d.original_asymmetry=std::max(d.original_asymmetry,std::abs(h.fock(p,q)-h.fock(q,p)));
  }
  for (std::size_t p=0;p<n;++p) for (std::size_t q=0;q<=p;++q) {
    const double left=h.fock(p,q)-h.hcore(p,q);
    const double right=h.fock(q,p)-h.hcore(q,p);
    if (!std::isfinite(left) || !std::isfinite(right) || std::abs(left-right)>1e-12) {
      std::ostringstream message;
      message<<std::scientific<<std::setprecision(12)
          <<"legacy Fock-minus-hcore at ("<<p<<","<<q<<"); asymmetry_Eh="<<std::abs(left-right)
          <<" exceeds unchanged 1e-12 symmetry tolerance; no repair applied";
      throw std::runtime_error(message.str());
    }
    const double value=projected.value(p,q)+std::midpoint(left,right);
    if (!std::isfinite(value)) throw std::runtime_error("non-finite rebuilt Fock");
    d.fock_maximum_change=std::max({d.fock_maximum_change,
        std::abs(value-h.fock(p,q)),std::abs(value-h.fock(q,p))});
    new_fock(p,q)=new_fock(q,p)=value;
  }
  // Commit only after every independent check passes; never change disk data.
  h.hcore=projected.value;
  h.fock=std::move(new_fock);
  return d;
}
} // namespace localized_cc::detail
