#pragma once

#include "localized_cc/frozen_core.hpp"
#include "localized_cc/parallel.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace localized_cc::detail {

// All norms refer to the existing UNIQUE occupied/virtual spin-triple
// representation: E=-L^T K M, with no extra 1/36. Bounds include the full
// 36-term antisymmetrizer and its 1/4 prefactor before restriction to a<b<c.
struct TriplesBlockScreenPlan {
  std::size_t tiles{}, candidates{}, skipped{}, workspace_bytes{};
  double energy_bound{};
  std::vector<double> keep; // doubles for the existing MPI broadcast interface
};

inline std::size_t screen_product(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::overflow_error("triples screen size overflow");
  return a*b;
}

// A sorted tile can have no spin-allowed a<b<c entries. Detect this without
// enumerating virtual triples. Greedy placement is exact for each spin string.
inline bool screen_tile_has_entries(const std::array<std::size_t,3>& abc,
                                    std::size_t width, std::size_t v,
                                    std::size_t alpha_count) {
  for (std::size_t bits=0; bits<8; ++bits) {
    std::size_t alphas=0, previous=0;
    bool valid=true;
    for (std::size_t axis=0; axis<3; ++axis) {
      const auto spin=(bits>>axis)&1U;
      alphas += spin==0;
      auto first=abc[axis]*width;
      if (axis) first=std::max(first,previous+1);
      if (first%2!=spin) ++first;
      if (first>=std::min(v,(abc[axis]+1)*width)) valid=false;
      previous=first;
    }
    if (valid && alphas==alpha_count) return true;
  }
  return false;
}

template<class T2>
TriplesBlockScreenPlan plan_triples_block_screen(
    const SpatialRhfHamiltonian& h, const Matrix& t1, const T2& t2,
    std::size_t width, double propagator_norm, double budget,
    std::size_t memory_limit,
    const std::function<void(const std::array<std::size_t,3>&,
        const std::array<std::size_t,3>&,double,double)>& bound_observer = {}) {
  const auto o=t1.dimensions()[0], v=t1.dimensions()[1];
  if (o<3 || v<3 || !width || !(budget>0) || !std::isfinite(budget) ||
      !(propagator_norm>0) || !std::isfinite(propagator_norm))
    throw std::invalid_argument("invalid triples block screening controls");
  width=std::min(width,v);
  const auto nt=v/width+(v%width!=0);
  const auto nt2=screen_product(nt,nt), nt3=screen_product(nt2,nt);
  const auto rows=screen_product(screen_product(o,o-1),o-2)/6;
  const auto slots=screen_product(rows,nt3);
  const auto pair=screen_product(screen_product(o,o),nt2);
  const auto one=screen_product(o,nt2);
  const auto three=screen_product(screen_product(o,o),nt);
  const auto single=screen_product(o,nt);
  // Upper bound for norm tables, two bounds/slot, decision mask, sort indices.
  const auto norm_slots=screen_product(8,std::max({pair,one,three,single}));
  const auto plan_slots=screen_product(4,slots);
  if (norm_slots>std::numeric_limits<std::size_t>::max()-plan_slots)
    throw std::overflow_error("triples screening memory overflow");
  const auto bytes=screen_product(norm_slots+plan_slots,
                                  std::max(sizeof(double),sizeof(std::size_t)));
  if (bytes>memory_limit)
    throw std::runtime_error("triples screening norm/decision tables exceed memory guard");
  std::vector<double> tp(pair), gp(pair), gi(one), gj(three), ss(single), ff(single);
  const SpinOrbitalEriView g(h);
  const auto f=expand_spatial_one_body_to_spin_orbital(h.fock);
  const auto world=parallel::world();
  const auto pair_index=[&](std::size_t i,std::size_t j,std::size_t a,std::size_t b) {
    return ((i*o+j)*nt+a)*nt+b;
  };
  // Disjoint i slices; no MPI calls inside contractions or OpenMP regions.
  // These O(o v^3 + o^2 v^2 + o^3 v) norm tables are built once, not per
  // triples tile or quadrature point. No triples moment is constructed here.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t ii=static_cast<std::ptrdiff_t>(world.rank);
       ii<static_cast<std::ptrdiff_t>(o); ii+=static_cast<std::ptrdiff_t>(world.size)) {
    const auto i=static_cast<std::size_t>(ii);
    for (std::size_t a=0; a<v; ++a) {
      const auto A=a/width;
      ss[i*nt+A]+=t1(i,a)*t1(i,a);
      ff[i*nt+A]+=f(o+a,i)*f(o+a,i);
    }
    for (std::size_t j=0; j<o; ++j) {
      for (std::size_t a=0; a<v; ++a) {
        const auto A=a/width;
        for (std::size_t b=0; b<v; ++b) {
          const auto p=pair_index(i,j,A,b/width);
          const auto x=t2(i,j,a,b), y=g(o+a,o+b,i,j);
          tp[p]+=x*x; gp[p]+=y*y;
        }
        long double sum=0;
        for (std::size_t m=0; m<o; ++m) {
          const long double x=g(m,o+a,i,j); sum+=x*x;
        }
        gj[(i*o+j)*nt+A]+=static_cast<double>(sum);
      }
    }
    for (std::size_t b=0; b<v; ++b)
      for (std::size_t c=0; c<v; ++c) {
        long double sum=0;
        for (std::size_t e=0; e<v; ++e) {
          const long double x=g(o+b,o+c,o+e,i); sum+=x*x;
        }
        gi[(i*nt+b/width)*nt+c/width]+=static_cast<double>(sum);
      }
  }
  for (auto* values:{&tp,&gp,&gi,&gj,&ss,&ff}) parallel::sum_in_place(*values);
  std::vector<double> ta(three), tm(one);
  for (std::size_t i=0;i<o;++i)
    for (std::size_t j=0;j<o;++j)
      for (std::size_t A=0;A<nt;++A)
        for (std::size_t B=0;B<nt;++B) {
          const auto x=tp[pair_index(i,j,A,B)];
          ta[(i*o+j)*nt+A]+=x;
          tm[(i*nt+A)*nt+B]+=x;
        }
  // Dimension-aware gamma guard for nonnegative norm accumulation. This is
  // an analytic norm bound with a numerical guard, not interval arithmetic.
  const long double operations=std::max(
      static_cast<long double>(width)*width*std::max(o,v),
      static_cast<long double>(slots));
  const long double gamma=64*std::numeric_limits<double>::epsilon()*operations;
  if (!(gamma<0.01L))
    throw std::runtime_error("triples screen roundoff guard is too large");
  const double guard=static_cast<double>(1.0L/(1.0L-gamma))*(1.0+1.e-10);
  for (auto* values:{&tp,&gp,&gi,&gj,&ss,&ff,&ta,&tm})
    for (auto& x:*values) {
      if (!(x>=0) || !std::isfinite(x))
        throw std::runtime_error("nonfinite triples screening norm");
      x=std::sqrt(x)*guard;
    }
  constexpr std::array<std::array<std::size_t,3>,6> perm{{
      {{0,1,2}},{{0,2,1}},{{1,0,2}},{{1,2,0}},{{2,0,1}},{{2,1,0}}}};
  TriplesBlockScreenPlan out;
  out.tiles=nt; out.workspace_bytes=bytes; out.keep.assign(slots,1.0);
  std::vector<double> right(slots), left(slots);
  std::vector<std::size_t> candidates;
  candidates.reserve(slots);
  std::size_t row=0;
  for (std::size_t i=0;i<o;++i)
    for (std::size_t j=i+1;j<o;++j)
      for (std::size_t k=j+1;k<o;++k,++row) {
        const std::array<std::size_t,3> occ{i,j,k};
        const auto alpha=(i%2==0)+(j%2==0)+(k%2==0);
        for (std::size_t A=0;A<nt;++A)
          for (std::size_t B=A;B<nt;++B)
            for (std::size_t C=B;C<nt;++C) {
              const std::array<std::size_t,3> abc{A,B,C};
              const auto slot=row*nt3+(A*nt+B)*nt+C;
              if (!screen_tile_has_entries(abc,width,v,alpha)) continue;
              long double r=0,s=0;
              for (const auto& po:perm) for (const auto& pv:perm) {
                const auto I=occ[po[0]],J=occ[po[1]],K=occ[po[2]];
                const auto a=abc[pv[0]],b=abc[pv[1]],c=abc[pv[2]];
                r+=static_cast<long double>(ta[(J*o+K)*nt+a])*gi[(I*nt+b)*nt+c]
                  +static_cast<long double>(tm[(I*nt+b)*nt+c])*gj[(J*o+K)*nt+a];
                s+=static_cast<long double>(ss[I*nt+a])*gp[pair_index(J,K,b,c)]
                  +static_cast<long double>(ff[I*nt+a])*tp[pair_index(J,K,b,c)];
              }
              right[slot]=static_cast<double>(0.25L*r)*(1.0+1.e-10);
              left[slot]=static_cast<double>(0.25L*(r+s))*(1.0+1.e-10);
              if (!std::isfinite(right[slot]) || !std::isfinite(left[slot]))
                throw std::runtime_error("nonfinite triples block bound");
              if (bound_observer)
                bound_observer(occ,{A*width,B*width,C*width},right[slot],left[slot]);
              candidates.push_back(slot);
            }
      }
  out.candidates=candidates.size();
  // Each tile contains disjoint unique external indices. Frobenius bounds
  // therefore combine in quadrature, not by an unsafe per-term threshold.
  long double r2=0,l2=0;
  for (const auto p:candidates) {
    r2+=static_cast<long double>(right[p])*right[p];
    l2+=static_cast<long double>(left[p])*left[p];
  }
  const long double rn=std::sqrt(r2),ln=std::sqrt(l2);
  if (!std::isfinite(rn) || !std::isfinite(ln))
    throw std::runtime_error("nonfinite total triples screening norm");
  std::sort(candidates.begin(),candidates.end(),[&](auto a,auto b) {
    const auto ca=rn*left[a]+ln*right[a], cb=rn*left[b]+ln*right[b];
    return ca==cb ? a<b : ca<cb;
  });
  long double dr2=0,dl2=0;
  for (const auto p:candidates) {
    const auto nr=dr2+static_cast<long double>(right[p])*right[p];
    const auto nl=dl2+static_cast<long double>(left[p])*left[p];
    const auto bound=static_cast<long double>(propagator_norm)*
        (std::sqrt(nl)*rn+ln*std::sqrt(nr))*(1.0L+1.e-10L);
    const auto rounded_bound=std::nextafter(static_cast<double>(bound),
                                           std::numeric_limits<double>::infinity());
    if (!std::isfinite(rounded_bound) || rounded_bound>budget) continue;
    dr2=nr; dl2=nl; out.keep[p]=0; ++out.skipped;
    out.energy_bound=rounded_bound;
  }
  // Every rank has the same norm tables; make rank zero authoritative for
  // decisions and diagnostics to avoid any platform-specific ordering drift.
  parallel::broadcast_in_place(out.keep);
  std::vector<double> diagnostics{out.energy_bound,static_cast<double>(out.skipped)};
  parallel::broadcast_in_place(diagnostics);
  out.energy_bound=diagnostics[0];
  out.skipped=static_cast<std::size_t>(diagnostics[1]);
  if (out.energy_bound>budget)
    throw std::runtime_error("rounded triples screening bound exceeds budget");
  return out;
}
} // namespace localized_cc::detail
