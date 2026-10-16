#include "localized_cc/pm_symmetry.hpp"
#include "localized_cc/parallel.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>

namespace localized_cc {
namespace {
constexpr double covariance_tolerance = 1.e-10;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check_group(const PmSymmetry& s) {
  require(s.occupied <= std::numeric_limits<std::size_t>::max()/2 &&
              s.virtuals <= std::numeric_limits<std::size_t>::max()/2,
          "PM symmetry spin dimensions overflow");
  const auto n = s.occupied + s.virtuals;
  const auto order = s.operations.size();
  require(s.occupied > 0 && s.virtuals > 0 && (order == 2 || order == 3 || order == 4),
          "PM symmetry requires a cyclic order-two, three or four active orbital group");
  for (std::size_t g = 0; g < order; ++g) {
    const auto& op = s.operations[g];
    require(op.image.size() == n && op.sign.size() == n,
            "PM symmetry orbital dimensions mismatch");
    std::vector<bool> seen(n, false);
    for (std::size_t p = 0; p < n; ++p) {
      require(op.image[p] < n && !seen[op.image[p]] &&
                  (op.sign[p] == 1 || op.sign[p] == -1) &&
                  ((p < s.occupied) == (op.image[p] < s.occupied)),
              "invalid PM signed occupied/virtual permutation");
      seen[op.image[p]] = true;
      if (g == 0)
        require(op.image[p] == p && op.sign[p] == 1,
                "PM symmetry operation zero must be identity");
    }
  }
  for (std::size_t g = 0; g < order; ++g)
    for (std::size_t h = 0; h < order; ++h)
      for (std::size_t p = 0; p < n; ++p) {
        const auto q = s.operations[h].image[p];
        const auto& product = s.operations[(g+h)%order];
        require(product.image[p] == s.operations[g].image[q] &&
                    product.sign[p] == s.operations[g].sign[q] *
                                           s.operations[h].sign[p],
                "PM symmetry signed maps are not a closed cyclic group");
      }
}
double hamiltonian_error(const SpatialRhfHamiltonian& h, const PmSymmetry& s) {
  check_group(s);
  const auto n = h.nocc + h.nvirt;
  require(h.nocc == s.occupied && h.nvirt == s.virtuals &&
              h.fock.dimensions() == std::array<std::size_t,2>{n,n} &&
              h.hcore.dimensions() == std::array<std::size_t,2>{n,n} &&
              h.eri_chemist.dimensions() == std::array<std::size_t,4>{n,n,n,n},
          "PM symmetry Hamiltonian dimensions mismatch");
  const auto world = parallel::world();
  double error = 0.;
  bool finite = true;
  auto compare = [&](double a, double b) {
    finite = finite && std::isfinite(a) && std::isfinite(b);
    error = std::max(error, std::abs(a-b));
  };
  // Generator covariance suffices for its powers. Split unique ERI quartets
  // over ranks; no MPI inside the integral loop, no sampling of the gate.
  const auto& g = s.operations[1];
  for (std::size_t p = world.rank; p < n; p += world.size)
    for (std::size_t q = 0; q <= p; ++q) {
      const double sign = g.sign[p]*g.sign[q];
      compare(h.fock(p,q), sign*h.fock(g.image[p],g.image[q]));
      compare(h.hcore(p,q), sign*h.hcore(g.image[p],g.image[q]));
      for (std::size_t r = 0; r <= p; ++r)
        for (std::size_t t = 0; t <= r; ++t) {
          if (r == p && t > q) continue;
          compare(h.eri_chemist(p,q,r,t),
                  sign*g.sign[r]*g.sign[t]*h.eri_chemist(
                      g.image[p],g.image[q],g.image[r],g.image[t]));
        }
    }
  std::vector<double> errors(world.size+1, 0.);
  errors[world.rank] = error;
  errors.back() = finite ? 0. : 1.;
  parallel::sum_in_place(errors);
  require(errors.back() == 0., "non-finite PM symmetry Hamiltonian");
  errors.pop_back();
  return *std::max_element(errors.begin(), errors.end());
}
} // namespace

void validate_pm_symmetry_group(const PmSymmetry& symmetry) {
  check_group(symmetry);
}

void validate_pm_spatial_amplitudes(const PmSymmetry& s,
                                   const Matrix& t1,const Tensor4& t2) {
  check_group(s);
  const auto o=s.occupied,v=s.virtuals;
  require(t1.dimensions()==std::array<std::size_t,2>{o,v} &&
          t2.dimensions()==std::array<std::size_t,4>{o,o,v,v},
          "PM spatial amplitude dimensions mismatch");
  const auto same=[](double x,double y) {
    return std::isfinite(x) && std::isfinite(y) &&
           std::abs(x-y)<=1e-10*std::max({1.0,std::abs(x),std::abs(y)});
  };
  for (const auto& g:s.operations) {
    for (std::size_t i=0;i<o;++i) for (std::size_t a=0;a<v;++a)
      require(same(t1(i,a),g.sign[i]*g.sign[o+a]*t1(g.image[i],g.image[o+a]-o)),
              "PM spatial T1 covariance failed");
    for (std::size_t i=0;i<o;++i) for (std::size_t j=0;j<o;++j)
      for (std::size_t a=0;a<v;++a) for (std::size_t b=0;b<v;++b)
        require(same(t2(i,j,a,b),g.sign[i]*g.sign[j]*g.sign[o+a]*g.sign[o+b]*
                t2(g.image[i],g.image[j],g.image[o+a]-o,g.image[o+b]-o)),
                "PM spatial T2 covariance failed");
  }
}

std::vector<int> find_cyclic_water_cycle(
    const std::vector<std::array<double,3>>& xyz,
    const std::vector<int>& numbers) {
  require((xyz.size() == 9 || xyz.size() == 12) && numbers.size() == xyz.size(),
          "cyclic PM symmetry requires exactly three or four waters");
  const auto order = xyz.size()/3;
  for (std::size_t p=0; p<xyz.size(); ++p) {
    require(numbers[p] == (p%3 == 0 ? 8 : 1),
            "cyclic PM symmetry requires O,H,H geometry ordering");
    for (double x : xyz[p]) require(std::isfinite(x), "non-finite geometry");
  }
  auto distance = [&](std::size_t p, std::size_t q) {
    double d=0.;
    for (std::size_t k=0;k<3;++k) d += std::pow(xyz[p][k]-xyz[q][k],2);
    return std::sqrt(d);
  };
  std::vector<int> cycle(order);
  std::iota(cycle.begin(),cycle.end(),0);
  double best_error=std::numeric_limits<double>::infinity();
  do {
    int p=0;
    bool full_cycle=true;
    for (std::size_t step=1;step<=order;++step) {
      p=cycle[static_cast<std::size_t>(p)];
      if ((step<order && p==0) || (step==order && p!=0)) full_cycle=false;
    }
    if (!full_cycle) continue;
    for (unsigned swaps=0;swaps<(1U<<order);++swaps) {
      std::vector<std::size_t> image(xyz.size());
      for (std::size_t m=0;m<order;++m) {
        const auto target=3*static_cast<std::size_t>(cycle[m]);
        image[3*m]=target;
        image[3*m+1]=target+1+((swaps>>m)&1U);
        image[3*m+2]=target+2-((swaps>>m)&1U);
      }
      double error=0.;
      for (std::size_t a=0;a<xyz.size();++a)
        for (std::size_t b=0;b<a;++b)
          error=std::max(error,std::abs(distance(a,b)-distance(image[a],image[b])));
      bool atom_closure=true;
      for (std::size_t a=0;a<xyz.size();++a) {
        auto mapped=a;
        for (std::size_t power=0;power<order;++power) mapped=image[mapped];
        atom_closure=atom_closure && mapped==a;
      }
      if (atom_closure) best_error=std::min(best_error,error);
      if (atom_closure && error <= 1.e-9) return cycle;
    }
  } while (std::next_permutation(cycle.begin(),cycle.end()));
  std::ostringstream message;
  message<<std::scientific<<std::setprecision(12)<<"W"<<order
      <<" geometry has no verified order-"<<order
      <<" water cycle at 1e-9 bohr (including hydrogens); best maximum distance error_bohr="
      <<best_error<<"; coordinates were not symmetrized";
  throw std::runtime_error(message.str());
}

std::array<int,4> find_w4_water_cycle(
    const std::vector<std::array<double,3>>& xyz,
    const std::vector<int>& numbers) {
  require(xyz.size()==12,"W4 PM symmetry requires exactly four waters");
  const auto cycle=find_cyclic_water_cycle(xyz,numbers);
  return {cycle[0],cycle[1],cycle[2],cycle[3]};
}

PmSymmetry find_w4_pm_symmetry(const SpatialRhfHamiltonian& h,
    const std::vector<int>& ol, const std::vector<int>& vl,
    const std::array<int,4>& cycle) {
  require(h.nocc>0 && h.nvirt>0 && ol.size()==h.nocc && vl.size()==h.nvirt,
          "PM symmetry assignment dimensions mismatch");
  const auto orbitals=h.nocc+h.nvirt;
  require(h.fock.dimensions()==std::array<std::size_t,2>{orbitals,orbitals} &&
              h.hcore.dimensions()==std::array<std::size_t,2>{orbitals,orbitals} &&
              h.eri_chemist.dimensions()==std::array<std::size_t,4>{orbitals,orbitals,orbitals,orbitals},
          "PM symmetry mapping Hamiltonian dimensions mismatch");
  auto sorted_cycle=cycle;
  std::sort(sorted_cycle.begin(),sorted_cycle.end());
  require(sorted_cycle==std::array<int,4>{0,1,2,3},"invalid water permutation");
  int water=0;
  for (int power=1;power<=4;++power) {
    water=cycle[static_cast<std::size_t>(water)];
    require((power==4)==(water==0),"water permutation must be a four-cycle");
  }
  std::vector<int> labels=ol;
  labels.insert(labels.end(),vl.begin(),vl.end());
  // Accept either zero-based or one-based consecutive monomer labels.
  const int first=*std::min_element(labels.begin(),labels.end());
  for (int m:labels) require(m>=first && m<first+4,"PM symmetry monomer labels must be four consecutive integers");
  const auto n=labels.size();
  PmSymmetry result{h.nocc,h.nvirt,{},0.};
  PmSymmetryOperation identity{std::vector<std::size_t>(n),std::vector<int>(n,1)};
  std::iota(identity.image.begin(),identity.image.end(),0);
  auto generator=identity;
  // Fingerprints only propose a mapping. The exhaustive Hamiltonian test
  // below, not the matching cutoff, decides whether it is admissible.
  for (std::size_t p=0;p<n;++p) {
    double best=1.e100, second=1.e100;
    std::size_t match=n;
    for (std::size_t q=0;q<n;++q) {
      if ((p<h.nocc)!=(q<h.nocc) || labels[q] != first+cycle[static_cast<std::size_t>(labels[p]-first)]) continue;
      const double score=std::abs(h.fock(p,p)-h.fock(q,q))+
          std::abs(h.hcore(p,p)-h.hcore(q,q))+
          std::abs(h.eri_chemist(p,p,p,p)-h.eri_chemist(q,q,q,q));
      require(std::isfinite(score),"non-finite PM symmetry fingerprint");
      if (score<best) {second=best; best=score; match=q;}
      else second=std::min(second,score);
    }
    require(match<n && best<1.e-5 && second-best>1.e-8,
            "PM orbitals lack an unambiguous symmetry permutation; symmetry-adapted PM localization is required (no approximate matching enabled)");
    generator.image[p]=match;
  }
  // Infer orbital phases along a maximum-weight spanning tree of one-body
  // couplings. All ERIs are subsequently checked, including odd powers.
  std::vector<bool> known(n,false);
  known[0]=true;
  for (std::size_t step=1;step<n;++step) {
    double best=0.; std::size_t a=n,b=n; int relation=1;
    for (std::size_t p=0;p<n;++p) if (known[p])
      for (std::size_t q=0;q<n;++q) if (!known[q]) {
        const double x=h.hcore(p,q), y=h.hcore(generator.image[p],generator.image[q]);
        const double weight=std::min(std::abs(x),std::abs(y));
        if(weight>best) {best=weight; a=p; b=q; relation=std::signbit(x)==std::signbit(y)?1:-1;}
      }
    require(best>1.e-12,"PM orbital phases cannot be connected reliably; symmetry reuse refused");
    generator.sign[b]=generator.sign[a]*relation; known[b]=true;
  }
  result.operations.push_back(identity);
  for (std::size_t power=1;power<4;++power) {
    auto next=identity;
    const auto& previous=result.operations.back();
    for(std::size_t p=0;p<n;++p) {
      next.image[p]=generator.image[previous.image[p]];
      next.sign[p]=previous.sign[p]*generator.sign[previous.image[p]];
    }
    result.operations.push_back(std::move(next));
  }
  result.maximum_hamiltonian_error=hamiltonian_error(h,result);
  if (result.maximum_hamiltonian_error>covariance_tolerance)
    throw std::runtime_error("PM signed permutation fails exhaustive Hamiltonian covariance at 1e-10 Eh; maximum error / tolerance="+
        std::to_string(result.maximum_hamiltonian_error/covariance_tolerance)+
        "; symmetry reuse refused");
  validate_pm_symmetry_labels(result,ol,vl);
  return result;
}

void validate_pm_symmetry(const SpatialRhfHamiltonian& h,const PmSymmetry& s) {
  require(hamiltonian_error(h,s)<=covariance_tolerance,
          "PM symmetry Hamiltonian covariance failed at 1e-10 Eh");
}
std::size_t pm_spin_image(const PmSymmetryOperation& g,std::size_t p,std::size_t offset) {
  return 2*(g.image.at(offset+p/2)-offset)+p%2;
}
int pm_spin_sign(const PmSymmetryOperation& g,std::size_t p,std::size_t offset) {
  return g.sign.at(offset+p/2);
}
std::pair<std::array<std::size_t,3>,int> pm_map_triple(
    const PmSymmetryOperation& g,std::array<std::size_t,3> t,std::size_t offset) {
  int sign=1;
  for(auto& p:t) {sign*=pm_spin_sign(g,p,offset); p=pm_spin_image(g,p,offset);}
  for(std::size_t p=0;p<3;++p) for(std::size_t q=p+1;q<3;++q)
    if(t[p]>t[q]) sign=-sign;
  std::sort(t.begin(),t.end());
  return {t,sign};
}
void validate_pm_symmetry_labels(const PmSymmetry& s,const std::vector<int>& ol,const std::vector<int>& vl) {
  check_group(s);
  require(ol.size()==s.occupied && vl.size()==s.virtuals,"PM symmetry label dimensions mismatch");
  std::vector<int> labels=ol; labels.insert(labels.end(),vl.begin(),vl.end());
  for(const auto& g:s.operations)
    for(std::size_t p=0;p<labels.size();++p)
      for(std::size_t q=0;q<labels.size();++q)
        require((labels[p]==labels[q])==(labels[g.image[p]]==labels[g.image[q]]),
                "PM symmetry does not preserve external monomer-span classes");
}
void validate_pm_symmetry_amplitudes(const PmSymmetry& s,const Matrix& t1,const Tensor4& t2) {
  check_group(s);
  const auto o=2*s.occupied,v=2*s.virtuals;
  require(t1.dimensions()==std::array<std::size_t,2>{o,v} &&
              t2.dimensions()==std::array<std::size_t,4>{o,o,v,v},
          "PM symmetry amplitude dimensions mismatch");
  const auto& g=s.operations[1];
  const auto world=parallel::world();
  double error=0.; bool finite=true;
  auto compare=[&](double a,double b) {finite=finite&&std::isfinite(a)&&std::isfinite(b); error=std::max(error,std::abs(a-b)/(1.+std::max(std::abs(a),std::abs(b))));};
  for(std::size_t i=world.rank;i<o;i+=world.size) for(std::size_t a=0;a<v;++a) {
    const auto I=pm_spin_image(g,i,0),A=pm_spin_image(g,a,s.occupied);
    const int sign=pm_spin_sign(g,i,0)*pm_spin_sign(g,a,s.occupied);
    compare(t1(i,a),sign*t1(I,A));
    for(std::size_t j=0;j<o;++j) for(std::size_t b=0;b<v;++b)
      compare(t2(i,j,a,b),sign*pm_spin_sign(g,j,0)*pm_spin_sign(g,b,s.occupied)*
          t2(I,pm_spin_image(g,j,0),A,pm_spin_image(g,b,s.occupied)));
  }
  std::vector<double> errors(world.size+1,0.);
  errors[world.rank]=error;
  errors.back()=finite?0.:1.;
  parallel::sum_in_place(errors);
  const bool all_finite=errors.back()==0.;
  errors.pop_back();
  require(all_finite && *std::max_element(errors.begin(),errors.end())<=1.e-10,
          "CCSD amplitudes break PM symmetry; task reuse refused rather than projecting amplitudes");
}
} // namespace localized_cc
