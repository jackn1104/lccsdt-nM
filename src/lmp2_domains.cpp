#include "localized_cc/lmp2_domains.hpp"
#include "modernqc/linalg/eigensolver.hpp"
#include "modernqc/linalg/matrix.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

namespace localized_cc {
namespace {
using Dense = modernqc::linalg::Matrix;
std::size_t mul(std::size_t a, std::size_t b) {
  if (a && b>std::numeric_limits<std::size_t>::max()/a)
    throw std::overflow_error("LMP2 workspace size overflow");
  return a*b;
}
std::size_t add(std::size_t a, std::size_t b) {
  if(b>std::numeric_limits<std::size_t>::max()-a)
    throw std::overflow_error("LMP2 workspace size overflow");
  return a+b;
}
void finite(double x) {
  if(!std::isfinite(x)) throw std::invalid_argument("nonfinite LMP2 input/result");
}
Dense checked_fock(const Matrix& f, std::size_t n) {
  if(!n || f.dimensions()!=std::array<std::size_t,2>{n,n})
    throw std::invalid_argument("LMP2 Fock/label dimensions mismatch");
  Dense r(n,n);
  for(std::size_t i=0;i<n;++i) for(std::size_t j=0;j<n;++j) {
    finite(f(i,j));
    if(std::abs(f(i,j)-f(j,i))>1e-12)
      throw std::invalid_argument("LMP2 Fock asymmetry exceeds 1e-12 Eh");
    // Explicitly use the Hermitian part only within the admitted roundoff.
    r(i,j)=0.5*f(i,j)+0.5*f(j,i);
  }
  return r;
}
double inner(const std::vector<double>& a,const std::vector<double>& b) {
  long double s=0;
  for(std::size_t k=0;k<a.size();++k) s+=static_cast<long double>(a[k])*b[k];
  const auto r=static_cast<double>(s); finite(r); return r;
}
// Apply a one-body matrix along one tensor axis. Only two packed matrices
// coexist. Dense linear algebra remains behind the ModernQC BLAS interface.
void axis_add(const Dense& f, const std::array<std::size_t,4>& dims,
              std::size_t axis, double sign, const std::vector<double>& x,
              std::vector<double>& out) {
  std::size_t before=1,after=1;
  for(std::size_t k=0;k<axis;++k) before*=dims[k];
  for(std::size_t k=axis+1;k<4;++k) after*=dims[k];
  const auto d=dims[axis], columns=before*after;
  Dense packed(d,columns);
  for(std::size_t l=0;l<before;++l) for(std::size_t k=0;k<d;++k)
    for(std::size_t r=0;r<after;++r)
      packed(k,l*after+r)=x[(l*d+k)*after+r];
  const auto product=modernqc::linalg::multiply(f,packed);
  for(std::size_t l=0;l<before;++l) for(std::size_t k=0;k<d;++k)
    for(std::size_t r=0;r<after;++r)
      out[(l*d+k)*after+r]+=sign*product(k,l*after+r);
}
std::ofstream output(const std::filesystem::path& p) {
  std::ofstream f(p); f.exceptions(std::ios::failbit|std::ios::badbit);
  f<<std::setprecision(17); return f;
}
} // namespace

Lmp2DomainResult select_lmp2_domains(const Matrix& foo,const Matrix& fvv,
    const Mp2Integral& eri,const std::vector<int>& ol,const std::vector<int>& vl,
    const Lmp2DomainOptions& opt) {
  const auto o=ol.size(),v=vl.size();
  for(double x:{opt.residual_l2_tolerance,opt.solve_energy_error_budget,
                opt.omission_budget_hartree}) finite(x);
  if(!eri || !o || !v || !opt.max_iterations ||
      opt.residual_l2_tolerance<=0 || opt.solve_energy_error_budget<=0 ||
      opt.omission_budget_hartree<0 ||
      opt.omission_budget_hartree+opt.solve_energy_error_budget>1e-6)
    throw std::invalid_argument("invalid LMP2 selector options (combined budget <=1e-6 Eh)");
  for(const auto& labels:{ol,vl}) for(int m:labels)
    if(m<0) throw std::invalid_argument("LMP2 requires assigned PM labels");
  const auto count=mul(mul(o,o),mul(v,v));
  // Payload upper bound including CG vectors, packing, matrix copies and
  // conservative per-element map/output allowance. Excludes caller ERIs.
  const auto bound=add(mul(count,1024),mul(add(mul(o,o),mul(v,v)),128));
  if(bound>opt.maximum_workspace_bytes)
    throw std::runtime_error("LMP2 domain-selection workspace exceeds configured cap; no allocation attempted");
  if(count>static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("LMP2 domain-selection tensor exceeds BLAS integer range");
  const auto fo=checked_fock(foo,o),fv=checked_fock(fvv,v);
  const auto eo=modernqc::linalg::diagonalize_symmetric(fo);
  const auto ev=modernqc::linalg::diagonalize_symmetric(fv);
  const double gap=2*(ev.eigenvalues.front()-eo.eigenvalues.back());
  if(!std::isfinite(gap)||gap<=1e-10)
    throw std::invalid_argument("LMP2 requires a positive occupied/virtual spectral gap");
  const std::array<std::size_t,4> dims{o,o,v,v};
  Lmp2DomainResult result;
  result.workspace_upper_bytes=bound; result.minimum_denominator=gap;
  result.amplitudes=Tensor4(dims);
  auto& t=result.amplitudes.data();
  std::vector<double> g(count),w(count),diag(count),r(count),z(count),p(count),ap(count);
  auto idx=[=](std::size_t i,std::size_t j,std::size_t a,std::size_t b) {
    return ((i*o+j)*v+a)*v+b;
  };
  for(std::size_t i=0;i<o;++i) for(std::size_t j=0;j<o;++j)
    for(std::size_t a=0;a<v;++a) for(std::size_t b=0;b<v;++b) {
      const auto k=idx(i,j,a,b); g[k]=eri(i,j,a,b); finite(g[k]);
      diag[k]=fv(a,a)+fv(b,b)-fo(i,i)-fo(j,j);
      finite(diag[k]);
      if(diag[k]<=0) throw std::runtime_error("invalid LMP2 preconditioner");
    }
  for(std::size_t i=0;i<o;++i) for(std::size_t j=0;j<o;++j)
    for(std::size_t a=0;a<v;++a) for(std::size_t b=0;b<v;++b) {
      const auto k=idx(i,j,a,b);
      if(std::abs(g[k]-g[idx(j,i,b,a)])>1e-12*(1+std::abs(g[k])))
        throw std::invalid_argument("LMP2 provider violates (ia|jb) pair symmetry");
      w[k]=2*g[k]-g[idx(i,j,b,a)]; finite(w[k]);
      r[k]=-g[k]; z[k]=r[k]/diag[k]; p[k]=z[k];
    }
  auto apply=[&](const std::vector<double>& x,std::vector<double>& y) {
    std::fill(y.begin(),y.end(),0.);
    axis_add(fo,dims,0,-1,x,y); axis_add(fo,dims,1,-1,x,y);
    axis_add(fv,dims,2,1,x,y); axis_add(fv,dims,3,1,x,y);
  };
  const double wn=std::sqrt(inner(w,w));
  auto measure=[&]() {
    result.residual_l2=std::sqrt(inner(r,r));
    result.solve_energy_error_bound=wn*result.residual_l2/gap;
    finite(result.solve_energy_error_bound);
    return result.residual_l2<=opt.residual_l2_tolerance &&
           result.solve_energy_error_bound<=opt.solve_energy_error_budget;
  };
  double rz=inner(r,z);
  bool converged=measure();
  for(std::size_t iteration=1;!converged && iteration<=opt.max_iterations;++iteration) {
    apply(p,ap);
    const double pap=inner(p,ap);
    if(!(pap>0) || !(rz>0)) throw std::runtime_error("LMP2 CG lost positive definiteness");
    const double alpha=rz/pap; finite(alpha);
    for(std::size_t k=0;k<count;++k) {t[k]+=alpha*p[k];r[k]-=alpha*ap[k];}
    result.iterations=iteration;
    const bool candidate=measure();
    // Recompute the actual equation residual before acceptance and periodically
    // restart CG to avoid accepting accumulated recurrence roundoff.
    const bool refresh=candidate || iteration%25==0 || iteration==opt.max_iterations;
    if(refresh) {apply(t,ap);for(std::size_t k=0;k<count;++k) r[k]=-g[k]-ap[k];}
    converged=measure();
    if(opt.observer) opt.observer(iteration,result.residual_l2,result.solve_energy_error_bound);
    if(converged) break;
    for(std::size_t k=0;k<count;++k) z[k]=r[k]/diag[k];
    const double next=inner(r,z),beta=next/rz;
    for(std::size_t k=0;k<count;++k) p[k]=z[k]+(refresh?0.:beta*p[k]);
    rz=next;
  }
  if(!converged) throw std::runtime_error("LMP2 domain pilot did not converge; no selection admitted");
  std::map<std::array<int,4>,Lmp2BlockScore> scores;
  for(std::size_t i=0;i<o;++i) for(std::size_t j=0;j<o;++j)
    for(std::size_t a=0;a<v;++a) for(std::size_t b=0;b<v;++b) {
      const auto k=idx(i,j,a,b); finite(t[k]);
      result.pair_symmetry_error=std::max(result.pair_symmetry_error,
          std::abs(t[k]-t[idx(j,i,b,a)]));
      const std::array<int,4> key{std::min(ol[i],ol[j]),std::max(ol[i],ol[j]),
                                std::min(vl[a],vl[b]),std::max(vl[a],vl[b])};
      auto& s=scores[key];s.labels=key;s.elements++;
      s.amplitude_norm=std::hypot(s.amplitude_norm,t[k]);
      s.energy_weight_norm=std::hypot(s.energy_weight_norm,w[k]);
      const double e=w[k]*t[k];finite(e);
      s.signed_energy+=e;s.absolute_energy_sum+=std::abs(e);
    }
  if(result.pair_symmetry_error>1e-10)
    throw std::runtime_error("LMP2 spatial amplitude pair symmetry failed");
  for(auto& [key,s]:scores) {
    s.monomer_span=std::set<int>(key.begin(),key.end()).size();
    s.cauchy_bound=s.amplitude_norm*s.energy_weight_norm;finite(s.cauchy_bound);
    finite(s.signed_energy);finite(s.absolute_energy_sum);
    // Guard summation roundoff; this is a fixed-amplitude MP2 energy bound,
    // not a bound on a re-solved restricted MP2, CCSD, or (T) model.
    s.cauchy_bound=std::max(s.cauchy_bound,s.absolute_energy_sum)*(1+1e-12);
    result.blocks.push_back(s);
  }
  std::vector<std::size_t> order(result.blocks.size());
  std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](auto a,auto b) {
    return result.blocks[a].cauchy_bound<result.blocks[b].cauchy_bound;
  });
  for(auto k:order) {
    auto& s=result.blocks[k];
    if(opt.omission_budget_hartree>0 && s.monomer_span>1 &&
       s.cauchy_bound<=opt.omission_budget_hartree-result.omission_bound) {
      s.retained=false;result.omission_bound+=s.cauchy_bound;
    }
  }
  for(const auto& s:result.blocks) {
    const auto n=s.monomer_span-1;result.full_nm[n]+=s.signed_energy;
    if(s.retained) {result.retained_nm[n]+=s.signed_energy;result.retained_elements+=s.elements;}
    else result.omitted_absolute_nm[n]+=s.absolute_energy_sum;
  }
  return result;
}

void write_lmp2_domain_audit(const std::filesystem::path& directory,
    const Lmp2DomainResult& r,const Lmp2DomainOptions& opt,
    const std::vector<int>& ol,const std::vector<int>& vl,const std::string& provenance) {
  if(!std::filesystem::create_directory(directory))
    throw std::runtime_error("refusing to overwrite LMP2 domain audit directory");
  auto blocks=output(directory/"blocks.csv");
  blocks<<"occ_m1,occ_m2,vir_m1,vir_m2,nM,elements,retained,t2_norm,energy_weight_norm,signed_energy_hartree,absolute_energy_hartree,cauchy_bound_hartree\n";
  std::map<std::pair<int,int>,std::set<int>> domains;
  for(const auto& b:r.blocks) {
    for(int m:b.labels) blocks<<m<<',';
    blocks<<b.monomer_span<<','<<b.elements<<','<<int(b.retained)<<','<<b.amplitude_norm<<','<<b.energy_weight_norm<<','<<b.signed_energy<<','<<b.absolute_energy_sum<<','<<b.cauchy_bound<<'\n';
    if(b.retained) {auto& d=domains[{b.labels[0],b.labels[1]}];d.insert(b.labels[2]);d.insert(b.labels[3]);}
  }
  blocks.close();
  auto d=output(directory/"pair_domains.csv");d<<"occ_m1,occ_m2,virtual_monomer\n";
  for(const auto& [pair,labels]:domains) for(int m:labels) d<<pair.first<<','<<pair.second<<','<<m<<'\n';
  d.close();
  auto labels=output(directory/"orbital_labels.csv");labels<<"space,index,monomer\n";
  for(std::size_t i=0;i<ol.size();++i) labels<<"occupied,"<<i<<','<<ol[i]<<'\n';
  for(std::size_t a=0;a<vl.size();++a) labels<<"virtual,"<<a<<','<<vl[a]<<'\n';
  labels.close();
  auto e=output(directory/"energy.csv");e<<"nM,full_mp2_hartree,retained_fixed_amplitude_hartree,omitted_absolute_hartree\n";
  for(std::size_t n=0;n<4;++n) e<<n+1<<','<<r.full_nm[n]<<','<<r.retained_nm[n]<<','<<r.omitted_absolute_nm[n]<<'\n';
  e.close();
  auto m=output(directory/"metadata.txt");
  m<<"schema=lmp2_pm_domains_v1\nstatus=candidate_only_not_ccsd_admitted\n"
   <<"integrals=four_center_no_df\nrepresentation=fixed_PM_spatial\n"
   <<"implementation=bounded_dense_amplitude_reference_pilot\n"
   <<"production_mask_applied=0\nccsd_t_accuracy_validated=0\n"
   <<"full_nm_eligible=1\nlabel_origin=unchanged_zero_based_PM_assignment\n"
   <<"solver=full_Fock_PCG\niterations="<<r.iterations<<"\nresidual_l2="<<r.residual_l2
   <<"\nminimum_denominator_hartree="<<r.minimum_denominator
   <<"\nsolve_energy_error_bound_hartree="<<r.solve_energy_error_bound
   <<"\nomission_bound_hartree="<<r.omission_bound
   <<"\nomission_budget_hartree="<<opt.omission_budget_hartree
   <<"\npair_symmetry_error="<<r.pair_symmetry_error
   <<"\nworkspace_upper_bytes="<<r.workspace_upper_bytes
   <<"\nretained_elements="<<r.retained_elements
   <<"\nfull_elements="<<r.amplitudes.size()<<"\nprovenance="<<provenance<<'\n';m.close();
  auto done=output(directory/"COMPLETE");done<<"LMP2 domain pilot complete; domains not yet validated for CCSD(T) accuracy\n";done.close();
}
} // namespace localized_cc
