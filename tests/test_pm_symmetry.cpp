#include "localized_cc/pm_symmetry.hpp"
#include "localized_cc/ccsd.hpp"
#include "localized_cc/spatial_ccsd.hpp"
#include "localized_cc/partition.hpp"
#include "localized_cc/parallel.hpp"
#include "localized_cc/checkpoint.hpp"
#include "../src/pm_orbit_plan.hpp"
#include <chrono>
#include <filesystem>
#ifdef PM_SYMMETRY_TRIPLES_TEST
#include "localized_cc/laplace_triples.hpp"
#include "localized_cc/pm_adapted.hpp"
#include "modernqc/molecule/molecule.hpp"
#endif
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <string>

using namespace localized_cc;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
#ifdef PM_SYMMETRY_TRIPLES_TEST
void test_saved_pm_inversion() {
  modernqc::linalg::Matrix s(4,4),a(4,4),o(4,2),v(4,2);
  for(std::size_t p=0;p<4;++p) {s(p,p)=1.;a(p^1,p)=p<2 ? 1. : -1.;}
  o(0,0)=o(1,1)=v(2,0)=v(3,1)=1.;
  const auto map=match_pm_inversion(o,v,s,a,{});
  require(map.operations.size()==2,"inversion order");
  for(std::size_t p=0;p<4;++p) {
    require(map.operations[1].image[p]==(p^1),"inversion signed map image");
    require(map.operations[1].sign[p]==(p<2?1:-1),"inversion orbital parity");
  }
  auto mixed=o;
  mixed(0,0)=mixed(1,1)=std::cos(0.2);
  mixed(0,1)=-std::sin(0.2);mixed(1,0)=std::sin(0.2);
  bool rejected=false;
  try {(void)match_pm_inversion(mixed,v,s,a,{});} catch(const std::runtime_error&) {rejected=true;}
  require(rejected,"inversion refuses orbital mixing instead of approximating a permutation");
}
void test_adapted_localization() {
  modernqc::linalg::Matrix identity(4,4),generator(4,4),canonical(4,4);
  for(std::size_t i=0;i<4;++i) {identity(i,i)=1.;generator((i+1)%4,i)=1.;}
  // A noncanonical complete space must give the same symmetry-related seed.
  canonical=identity;
  const double angle=0.37;
  canonical(0,0)=canonical(1,1)=std::cos(angle);
  canonical(0,1)=std::sin(angle);canonical(1,0)=-std::sin(angle);
  const auto result=adapt_w4_pm(canonical,identity,{0,1,2,3},identity,
      generator,identity,{0,3,6,9},{});
  require(std::abs(result.objective-4.)<1.e-12,"adapted PM known objective");
  for(std::size_t i=0;i<4;++i) for(std::size_t j=0;j<4;++j)
    require(std::abs(result.coefficients(i,j)-identity(i,j))<1.e-12,
        "adapted PM rotated-space orbital equivalence");
  bool rejected=false;
  try { (void)adapt_w4_pm(canonical,identity,{0,0,2,3},identity,
      generator,identity,{0,3,6,9},{}); }
  catch(const std::runtime_error&) {rejected=true;}
  require(rejected,"adapted PM rejects incomplete orbit seeds");
  const auto map=w4_adapted_permutation(4,4);
  require(map.operations[1].image[0]==1 && map.operations[1].image[7]==4,
      "adapted PM occupied/virtual map");
  modernqc::linalg::Matrix mixed(4,4),skew(4,4);
  for(std::size_t i=0;i<4;++i) for(std::size_t j=0;j<4;++j)
    skew(i,j)=generator(i,j)-generator(j,i);
  for(std::size_t i=0;i<4;++i) for(std::size_t j=0;j<4;++j) {
    mixed(i,j)=identity(i,j)+0.5*std::sin(0.2)*skew(i,j);
    for(std::size_t k=0;k<4;++k)
      mixed(i,j)+=0.25*(1.-std::cos(0.2))*skew(i,k)*skew(k,j);
  }
  const auto optimized=adapt_w4_pm(identity,mixed,{0,1,2,3},identity,
      generator,identity,{0,3,6,9},{});
  require(std::abs(optimized.objective-4.)<1.e-10,
      "adapted PM optimizes a nonstationary symmetry-compatible seed");
  require(optimized.gradient_norm<1.e-7,"adapted PM retains gradient threshold");
  // Synthetic weak-curvature objective: rescaling the population map by 0.1
  // scales both curvature and gradient by 1e-4. This is an optimizer stress
  // fixture, not a physical meta-Lowdin AO basis or a relaxed molecular gate.
  auto weak_metric=identity;
  for(std::size_t i=0;i<4;++i) weak_metric(i,i)=0.1;
  const auto weak=adapt_w4_pm(identity,mixed,{0,1,2,3},identity,
      generator,weak_metric,{0,3,6,9},{});
  require(weak.gradient_norm<1.e-7 && weak.iterations<100,
      "adapted PM Newton resolves weak curvature without unit-step plateau");
  require(std::abs(weak.objective-4.e-4)<1.e-8,
      "adapted PM weak-curvature known objective");
}
#endif
void close(double a,double b,const char* message) {
  require(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=1.e-12*(1.+std::max(std::abs(a),std::abs(b))),message);
}
template<class F> void rejects(F&& f) {bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,"invalid symmetry input was accepted");}
PmSymmetry group() {
  PmSymmetry s{4,4,{},0.};
  const std::vector<std::size_t> generator{1,2,3,0,5,6,7,4};
  const std::vector<int> phases{-1,1,-1,1,1,-1,1,-1};
  s.operations.push_back({{0,1,2,3,4,5,6,7},{1,1,1,1,1,1,1,1}});
  for(std::size_t power=1;power<4;++power) {
    auto op=s.operations.back();
    for(std::size_t p=0;p<8;++p) {op.sign[p]*=phases[op.image[p]];op.image[p]=generator[op.image[p]];}
    s.operations.push_back(op);
  }
  return s;
}
SpatialRhfHamiltonian hamiltonian(const PmSymmetry& s) {
  SpatialRhfHamiltonian h;
  const auto n=s.occupied+s.virtuals;
  const double weight=1./static_cast<double>(s.operations.size());
  h.nocc=s.occupied;h.nvirt=s.virtuals;h.fock=Matrix({n,n});h.hcore=Matrix({n,n});h.eri_chemist=SymmetricEri4(n);
  SymmetricEri4 seed(n);
  for(std::size_t x=0;x<seed.size();++x) seed.data()[x]=.002*std::sin(static_cast<double>(x+1));
  for(std::size_t p=0;p<n;++p) for(std::size_t q=0;q<n;++q) {
    for(const auto& g:s.operations) {
      double f=.001*std::cos(static_cast<double>((g.image[p]+1)*(g.image[q]+1)));
      if(p==q) f+=p<s.occupied?-1.:.5;
      h.fock(p,q)+=weight*g.sign[p]*g.sign[q]*f;
      h.hcore(p,q)+=weight*g.sign[p]*g.sign[q]*(f+.03*std::sin(static_cast<double>(g.image[p]+g.image[q]+2)));
    }
    for(std::size_t r=0;r<n;++r) for(std::size_t t=0;t<n;++t) {
      double value=0.;
      for(const auto& g:s.operations) value+=weight*g.sign[p]*g.sign[q]*g.sign[r]*g.sign[t]*seed(g.image[p],g.image[q],g.image[r],g.image[t]);
      h.eri_chemist(p,q,r,t)=value;
    }
  }
  return h;
}
SpatialRccsdAmplitudes amplitudes(const PmSymmetry& s,double scale) {
  const auto o=s.occupied,v=s.virtuals;
  const double weight=1./static_cast<double>(s.operations.size());
  SpatialRccsdAmplitudes t{Matrix({o,v}),Tensor4({o,o,v,v})};
  for(std::size_t i=0;i<o;++i) for(std::size_t a=0;a<v;++a) {
    for(const auto& g:s.operations) {
      const auto I=g.image[i],A=g.image[o+a]-o;
      t.t1(i,a)+=weight*scale*g.sign[i]*g.sign[o+a]*std::sin(static_cast<double>(v*I+A+1));
      for(std::size_t j=0;j<o;++j) for(std::size_t b=0;b<v;++b) {
        const auto J=g.image[j],B=g.image[o+b]-o;
        t.t2(i,j,a,b)+=weight*scale*g.sign[i]*g.sign[j]*g.sign[o+a]*g.sign[o+b]*std::sin(static_cast<double>((v*I+A+1)*(v*J+B+1)));
      }
    }
  }
  return t;
}
void test_c3() {
  auto s=std::make_shared<PmSymmetry>();
  s->occupied=3;s->virtuals=3;
  const std::array<int,6> gauge{-1,1,1,1,-1,1};
  for(std::size_t power=0;power<3;++power) {
    PmSymmetryOperation op;
    for(std::size_t p=0;p<6;++p) {
      const auto q=(p/3)*3+(p%3+power)%3;
      op.image.push_back(q);op.sign.push_back(gauge[p]*gauge[q]);
    }
    s->operations.push_back(std::move(op));
  }
  validate_pm_symmetry_group(*s);
  validate_pm_symmetry_labels(*s,{0,1,2},{0,1,2});
  auto invalid=*s;invalid.operations.pop_back();
  rejects([&]{validate_pm_symmetry_group(invalid);});
  invalid=*s;invalid.operations[1].sign[0]*=-1;
  rejects([&]{validate_pm_symmetry_group(invalid);});
  detail::PmOrbitPlan<2> plan(s.get(),{3,3},{0,3});
  require(plan.tasks()==3,"C3 pair representative count");
  std::vector<double> values(9);
  for(std::size_t task=0;task<plan.tasks();++task)
    plan.scatter(plan.representative(task),1.+static_cast<double>(task),values);
  for(std::size_t g=0;g<3;++g) for(std::size_t k=0;k<9;++k) {
    const auto [mapped,sign]=plan.map(g,k);
    close(values[k],sign*values[mapped],"C3 signed scatter");
  }
  // All three same-spin occupied indices form a fixed triple, not a
  // three-member orbit. Triples tests below exercise its weight of one.
  require(pm_map_triple(s->operations[1],{0,2,4},0).first==
      std::array<std::size_t,3>{0,2,4},"C3 occupied-triple stabilizer");
  std::vector<std::array<double,3>> xyz;
  std::vector<int> numbers;
  for(int m=0;m<3;++m) {
    const double angle=2.*std::acos(-1.)*m/3.;
    for(const auto p:std::array<std::array<double,3>,3>{{{3.,0.,0.},{3.6,.5,.2},{3.5,-.5,-.2}}}) {
      xyz.push_back({std::cos(angle)*p[0]-std::sin(angle)*p[1],
          std::sin(angle)*p[0]+std::cos(angle)*p[1],p[2]});
      numbers.push_back(numbers.size()%3==0?8:1);
    }
  }
  require(find_cyclic_water_cycle(xyz,numbers).size()==3,"synthetic C3 geometry");
#ifdef PM_SYMMETRY_TRIPLES_TEST
  {
    std::vector<modernqc::molecule::Atom> atoms;
    for(std::size_t p=0;p<xyz.size();++p)
      atoms.push_back({numbers[p]==8?"O":"H",numbers[p],xyz[p]});
    const modernqc::molecule::Molecule molecule(atoms,0,1);
    const modernqc::integrals::Libint2IntegralProvider integrals(molecule,"sto-3g",true);
    const auto one=integrals.compute_one_electron();
    const auto action=cyclic_ao_symmetry_action(integrals,molecule,one.overlap,
        find_cyclic_water_cycle(xyz,numbers));
    require(action.rows()==one.overlap.rows(),"C3 AO action dimensions");
    // Independently check the one-electron scalar overlap covariance; shell
    // collocation and action cubed=I are checked inside the action builder.
    for(std::size_t p=0;p<action.rows();++p) for(std::size_t q=0;q<action.rows();++q) {
      double rotated=0.;
      for(std::size_t r=0;r<action.rows();++r) for(std::size_t t=0;t<action.rows();++t)
        rotated+=action(r,p)*one.overlap(r,t)*action(t,q);
      require(std::abs(rotated-one.overlap(p,q))<1e-10,"C3 AO overlap covariance");
    }
  }
#endif
  xyz[1][2]+=.001;
  rejects([&]{(void)find_cyclic_water_cycle(xyz,numbers);});
  // Unchanged oxygen triangle is insufficient when a hydrogen breaks C3.
  const auto h=hamiltonian(*s);
  validate_pm_symmetry(h,*s);
  auto bad=h;bad.fock(0,0)+=1e-5;
  rejects([&]{validate_pm_symmetry(bad,*s);});
  CcsdOptions oracle;
  oracle.amplitude_backend=CcsdAmplitudeBackend::replicated;
  oracle.wabef_backend=CcsdWabefBackend::tiled_cpu;
  oracle.wabef_packing_backend=CcsdWabefPackingBackend::cached_invariants;
  oracle.wabef_invariant_cache_max_bytes=16*1024*1024;
  oracle.wabef_tile_max_bytes=4*1024*1024;
  auto reduced=oracle;
  reduced.use_direct_spatial_singles=true;
  reduced.use_direct_spatial_doubles=true;
  reduced.use_tiled_spatial_pairs=true;
  reduced.use_pm_residual_orbits=true;
  reduced.pm_symmetry=s;
  for(double scale:{1e-4,1e-2,1e-1}) {
    const auto t=amplitudes(*s,scale);
    const auto a=evaluate_spatial_rccsd_residual(h,t,oracle);
    const auto b=evaluate_spatial_rccsd_residual(h,t,reduced);
    for(std::size_t k=0;k<a.r1.size();++k) close(a.r1.data()[k],b.r1.data()[k],"C3 R1 oracle");
    for(std::size_t k=0;k<a.r2.size();++k) close(a.r2.data()[k],b.r2.data()[k],"C3 R2 oracle");
#ifdef PM_SYMMETRY_TRIPLES_TEST
    if(scale==1e-2) {
      const auto t1=expand_spatial_rccsd_t1_to_spin_orbital(t.t1);
      const auto t2=expand_spatial_rccsd_t2_to_spin_orbital(t.t2);
      StreamedLaplaceTriplesOptions options;
      options.moment_backend=CanonicalTriplesBackend::tiled_cpu;
      options.virtual_tile_size=2;
      options.requested_energy_error_hartree=1e-7;
      for(const auto labels: {std::vector<int>{0,1,2},std::vector<int>{3,4,5}}) {
        options.pm_symmetry=nullptr;options.symmetry_propagation=false;
        options.sector_streaming=false;options.shared_pair_packing=false;
        const auto reference=solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2},labels,options);
        options.pm_symmetry=s;options.symmetry_propagation=true;
        options.shared_pair_packing=true;
        for(bool compact:{false,true}) {
          options.sector_streaming=compact;
          const auto result=solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2},labels,options);
          close(reference.parenthetical_energy,result.parenthetical_energy,"C3 triples total");
          require(reference.many_body_terms==result.many_body_terms,"C3 full excitation counts");
          for(std::size_t m=0;m<6;++m)
            close(reference.many_body_energies[m],result.many_body_energies[m],"C3 triples nM");
          auto restart=options;
          restart.initial_point_many_body_energies=reference.point_many_body_energies;
          restart.completed_points.assign(reference.completed_points.size(),0);
          restart.completed_points[0]=1;
          const auto resumed=solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2},labels,restart);
          close(reference.parenthetical_energy,resumed.parenthetical_energy,"C3 triples point restart");
        }
      }
    }
#endif
  }
  oracle.max_iterations=reduced.max_iterations=150;
  oracle.residual_tolerance=reduced.residual_tolerance=1e-12;
  oracle.energy_tolerance=reduced.energy_tolerance=1e-13;
  const auto a=solve_spatial_rccsd_reference(h,oracle);
  const auto b=solve_spatial_rccsd_reference(h,reduced);
  require(a.converged && b.converged,"C3 CCSD nonlinear solves");
  close(a.correlation_energy,b.correlation_energy,"C3 converged CCSD energy");
  const auto labels=expand_spatial_labels({0,1,2});
  auto partition=[&](const SpatialRccsdAmplitudes& t) {
    return partition_ccsd(h,expand_spatial_rccsd_t1_to_spin_orbital(t.t1),
        expand_spatial_rccsd_t2_to_spin_orbital(t.t2),labels,labels);
  };
  const auto pa=partition(a.amplitudes),pb=partition(b.amplitudes);
  close(pa.sum,a.correlation_energy,"C3 oracle partition closure");
  close(pb.sum,b.correlation_energy,"C3 reduced partition closure");
  for(const auto& [label,value]:pa.many_body_classes)
    close(value,pb.many_body_classes.at(label),"C3 CCSD nM agreement");
  reduced.initial_t1=expand_spatial_rccsd_t1_to_spin_orbital(b.amplitudes.t1);
  reduced.initial_t2=expand_spatial_rccsd_t2_to_spin_orbital(b.amplitudes.t2);
  const auto restarted=solve_spatial_rccsd_reference(h,reduced);
  require(restarted.converged,"C3 CCSD amplitude restart");
  close(b.correlation_energy,restarted.correlation_energy,"C3 CCSD restart energy");
#ifdef PM_SYMMETRY_TRIPLES_TEST
  modernqc::linalg::Matrix identity(3,3),generator(3,3),mixed(3,3),skew(3,3);
  for(std::size_t i=0;i<3;++i) {identity(i,i)=1.;generator((i+1)%3,i)=1.;}
  for(std::size_t i=0;i<3;++i) for(std::size_t j=0;j<3;++j)
    skew(i,j)=generator(i,j)-generator(j,i);
  for(std::size_t i=0;i<3;++i) for(std::size_t j=0;j<3;++j) {
    mixed(i,j)=identity(i,j)+std::sin(.2)/std::sqrt(3.)*skew(i,j);
    for(std::size_t k=0;k<3;++k) mixed(i,j)+=(1.-std::cos(.2))/3.*skew(i,k)*skew(k,j);
  }
  const auto adapted=adapt_cyclic_pm(identity,mixed,{0,1,2},identity,generator,identity,{0,3,6},3,{});
  require(adapted.gradient_norm<1e-7 && std::abs(adapted.objective-3.)<1e-10,"C3 adapted PM optimizer");
  const auto permutation=cyclic_adapted_permutation(3,3,3);
  validate_pm_symmetry_group(permutation);
  rejects([&]{(void)cyclic_adapted_permutation(4,3,3);});
  rejects([&]{(void)adapt_cyclic_pm(identity,mixed,{0,0,2},identity,generator,identity,{0,3,6},3,{});});
#endif
  if(parallel::world().rank==0) std::cout<<"C3 signed PM CCSD/full-nM equation gates passed\n";
}

void check_geometry_file(const std::string& path, std::size_t order) {
  std::ifstream file(path);
  std::size_t n=0;file>>n;
  require(file && n==3*order,"cyclic geometry preflight atom count mismatch");
  std::string line;std::getline(file,line);std::getline(file,line);
  std::vector<std::array<double,3>> xyz(n);
  std::vector<int> numbers(n);
  for(std::size_t p=0;p<n;++p) {
    std::string symbol;file>>symbol>>xyz[p][0]>>xyz[p][1]>>xyz[p][2];
    require(file && (symbol=="O" || symbol=="H"),"invalid cyclic geometry record");
    numbers[p]=symbol=="O"?8:1;
    for(auto& coordinate:xyz[p]) coordinate*=1.8897261254578281;
  }
  std::string trailing;
  require(!(file>>trailing),"unexpected extra geometry records");
  require(find_cyclic_water_cycle(xyz,numbers).size()==order,"cyclic geometry order mismatch");
  if(parallel::world().rank==0)
    std::cout<<"Full-atom cyclic geometry preflight passed; order="<<order<<"; tolerance=1e-9 bohr; path="<<path<<'\n';
}

void test_geometry() {
  std::ifstream file(std::string(PM_SYMMETRY_SOURCE_DIR)+"/examples/benchmarks/reference_geometries/w4_ring_tce_avdz.xyz");
  std::size_t n=0;file>>n;std::string line;std::getline(file,line);std::getline(file,line);
  std::vector<std::array<double,3>> xyz(n);std::vector<int> numbers(n);
  for(std::size_t p=0;p<n;++p) {
    std::string element;file>>element>>xyz[p][0]>>xyz[p][1]>>xyz[p][2];numbers[p]=element=="O"?8:1;
    for(auto& x:xyz[p]) x*=1.8897261254578281;
  }
  require(static_cast<bool>(file),"missing W4 geometry");
  const auto cycle=find_w4_water_cycle(xyz,numbers);
  int m=0;for(int k=0;k<4;++k)m=cycle[static_cast<std::size_t>(m)];require(m==0,"water cycle closure");
  xyz[1][0]+=.001;
  rejects([&]{(void)find_w4_water_cycle(xyz,numbers);});
}
void test_equations() {
  auto s=std::make_shared<PmSymmetry>(group());
  const detail::PmOrbitPlan<2> plan(s.get(),{4,4},{0,4});
  require(plan.tasks()==4,"signed pair orbit representative count");
  std::vector<double> scattered(16);
  for(std::size_t task=0;task<plan.tasks();++task)
    plan.scatter(plan.representative(task),static_cast<double>(task+1),scattered);
  for(std::size_t g=0;g<4;++g) for(std::size_t k=0;k<16;++k) {
    const auto [mapped,sign]=plan.map(g,k);
    close(scattered[k],sign*scattered[mapped],"signed orbit reconstruction");
  }
  rejects([&]{detail::PmOrbitPlan<2> too_small(s.get(),{4,4},{0,4},false,1);});
  auto fixed=*s;
  for(auto& op:fixed.operations) op=fixed.operations[0];
  const detail::PmOrbitPlan<2> stabilizers(&fixed,{4,4},{0,4});
  require(stabilizers.tasks()==16,"fixed-index stabilizers retain every representative");
  std::size_t count=0;
  stabilizers.images(0,[&](std::size_t,std::size_t,int){++count;});
  require(count==1,"stabilizers do not multiply orbit weight by group order");
  auto h=hamiltonian(*s);
  validate_pm_symmetry(h,*s);
  validate_pm_symmetry_labels(*s,{0,1,2,3},{0,1,2,3});
  const auto inferred=find_w4_pm_symmetry(h,{0,1,2,3},{0,1,2,3},{1,2,3,0});
  validate_pm_symmetry(h,inferred);
  auto bad=h;bad.eri_chemist(0,0,0,1)+=1.e-5;
  rejects([&]{validate_pm_symmetry(bad,*s);});
  auto broken=*s;broken.operations[1].sign[0]=0;
  rejects([&]{validate_pm_symmetry(h,broken);});
  auto nonfinite=h;nonfinite.fock(0,0)=std::numeric_limits<double>::quiet_NaN();
  rejects([&]{validate_pm_symmetry(nonfinite,*s);});
  rejects([&]{(void)find_w4_pm_symmetry(h,{0,1,2,3},{0,1,2,3},{0,1,2,3});});
  rejects([&]{validate_pm_symmetry_labels(*s,{0,1,2,3},{0,0,2,3});});
  CcsdOptions reference;
  reference.amplitude_backend=CcsdAmplitudeBackend::replicated;
  reference.wabef_backend=CcsdWabefBackend::tiled_cpu;
  reference.wabef_packing_backend=CcsdWabefPackingBackend::cached_invariants;
  reference.wabef_invariant_cache_max_bytes=16*1024*1024;
  for(double scale:{1.e-4,1.e-2,1.e-1}) {
    const auto t=amplitudes(*s,scale);
    const auto t1=expand_spatial_rccsd_t1_to_spin_orbital(t.t1);
    const auto t2=expand_spatial_rccsd_t2_to_spin_orbital(t.t2);
    validate_pm_symmetry_amplitudes(*s,t1,t2);
    auto native=reference;
    native.use_direct_spatial_singles=true;
    native.use_direct_spatial_doubles=true;
    native.use_tiled_spatial_pairs=true;
    native.wabef_tile_max_bytes=4*1024*1024;
    const auto oracle=evaluate_spatial_rccsd_residual(h,t,reference);
    for(bool symmetry:{false,true}) {
      native.pm_symmetry=symmetry?s:nullptr;
      const auto spatial=evaluate_spatial_rccsd_residual(h,t,native);
      for(std::size_t k=0;k<oracle.r1.size();++k)
        close(oracle.r1.data()[k],spatial.r1.data()[k],"native PM R1 oracle");
      for(std::size_t k=0;k<oracle.r2.size();++k)
        close(oracle.r2.data()[k],spatial.r2.data()[k],"native PM R2 oracle");
      require(spatial.timing.wabef_output_tasks==(symmetry?4:16),
          "native PM spatial external pair task reduction");
      if (symmetry) {
        native.use_pm_residual_orbits=true;
        const auto reduced=evaluate_spatial_rccsd_residual(h,t,native);
        for(std::size_t k=0;k<oracle.r1.size();++k)
          close(oracle.r1.data()[k],reduced.r1.data()[k],"PM representative R1 oracle");
        for(std::size_t k=0;k<oracle.r2.size();++k)
          close(oracle.r2.data()[k],reduced.r2.data()[k],"PM representative R2 oracle");
        require(reduced.timing.intermediate_mixed_pair_tasks<spatial.timing.intermediate_mixed_pair_tasks,
            "PM intermediate representative task reduction");
        const auto singles=evaluate_spatial_rccsd_singles_direct(h,t,native);
        const auto doubles=evaluate_spatial_rccsd_doubles_direct(h,t,native);
        require(singles.global_output_tasks==4 && doubles.global_output_tasks==64,
            "PM R1/R2 external orbit coverage");
        require(singles.global_intermediate_tasks==48,"PM small F intermediate orbit coverage");
        auto bad=t;bad.t1(0,0)+=1e-4;
        rejects([&]{(void)evaluate_spatial_rccsd_residual(h,bad,native);});
        native.use_pm_residual_orbits=false;
      }
    }
    auto bad_t1=t1;bad_t1(0,0)+=1.e-4;
    rejects([&]{validate_pm_symmetry_amplitudes(*s,bad_t1,t2);});
    for(std::size_t cap:{std::size_t{1},std::size_t{1024*1024}}) {
      reference.wabef_tile_max_bytes=cap;
      auto options=reference;options.pm_symmetry=s;
      auto a=evaluate_spatial_rccsd_residual(h,t,reference);
      auto b=evaluate_spatial_rccsd_residual(h,t,options);
      for(std::size_t k=0;k<a.r1.size();++k) close(a.r1.data()[k],b.r1.data()[k],"symmetry R1 oracle");
      for(std::size_t k=0;k<a.r2.size();++k) close(a.r2.data()[k],b.r2.data()[k],"symmetry R2 oracle");
      const auto tasks=parallel::sum_group(b.timing.wabef_output_tasks,0);
      require(tasks==16,"C4 external virtual-pair representative count");
    }
#ifdef PM_SYMMETRY_TRIPLES_TEST
    if(scale==1.e-2) {
      StreamedLaplaceTriplesOptions options;
      options.moment_backend=CanonicalTriplesBackend::tiled_cpu;
      options.virtual_tile_size=3;
      options.requested_energy_error_hartree=1.e-7;
      const auto a=solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2,3},{0,1,2,3},options);
      options.pm_symmetry=s;
      const auto b=solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2,3},{0,1,2,3},options);
      close(a.parenthetical_energy,b.parenthetical_energy,"symmetry Laplace triples oracle");
      require(b.symmetry_unique_moment_rows<a.occupied_triples,"triples symmetry failed to eliminate tasks");
      require(a.many_body_terms==b.many_body_terms,"triples symmetry nM term coverage");
      for(std::size_t m=0;m<6;++m) close(a.many_body_energies[m],b.many_body_energies[m],"symmetry triples nM oracle");
      options.sector_streaming=true;
      options.shared_pair_packing=true;
      options.propagation_row_batch=1;
      const auto compact=solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2,3},{0,1,2,3},options);
      close(a.parenthetical_energy,compact.parenthetical_energy,"compact symmetry triples total");
      require(compact.symmetry_unique_moment_rows==b.symmetry_unique_moment_rows,"compact symmetry orbit coverage");
      for(std::size_t m=0;m<6;++m) close(a.many_body_energies[m],compact.many_body_energies[m],"compact symmetry nM oracle");
      options.symmetry_propagation=true;
      for(bool streamed:{false,true}) {
        options.sector_streaming=streamed;
        const auto reduced=solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2,3},{0,1,2,3},options);
        close(a.parenthetical_energy,reduced.parenthetical_energy,"PM propagated total oracle");
        require(a.many_body_terms==reduced.many_body_terms,"PM propagated excitation multiplicities");
        for(std::size_t m=0;m<6;++m) close(a.many_body_energies[m],reduced.many_body_energies[m],"PM propagated nM oracle");
        // Reuse one old-path quadrature point; remaining points use the new
        // executor. No denominator, label or numerical tolerance changes.
        auto restart_options=options;
        restart_options.initial_point_many_body_energies=b.point_many_body_energies;
        restart_options.completed_points.assign(b.completed_points.size(),0);
        restart_options.completed_points[0]=1;
        const auto restarted=solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2,3},{0,1,2,3},restart_options);
        close(a.parenthetical_energy,restarted.parenthetical_energy,"PM propagation point restart oracle");
      }
      // Eight synthetic monomer labels exercise nonzero M5/M6; unlike W4
      // this is a classification oracle, not a molecular reference.
      options.sector_streaming=false;
      options.symmetry_propagation=false;
      const auto high=solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2,3},{4,5,6,7},options);
      require(std::abs(high.many_body_energies[4])>1e-30 && std::abs(high.many_body_energies[5])>1e-30,
          "synthetic PM fixture must exercise M5 and M6");
      options.symmetry_propagation=true;
      for(bool streamed:{false,true}) {
        options.sector_streaming=streamed;
        const auto reduced=solve_localized_triples_laplace_streamed(h,t1,t2,{0,1,2,3},{4,5,6,7},options);
        for(std::size_t m=0;m<6;++m) close(high.many_body_energies[m],reduced.many_body_energies[m],"PM high-span propagation oracle");
      }
    }
#endif
  }
  // Full nonlinear solves, nM closure, and amplitude-only restart. DIIS
  // history is deliberately not restored; only the final state must agree.
  reference.max_iterations=150;
  reference.residual_tolerance=1.e-12;
  reference.energy_tolerance=1.e-13;
  reference.wabef_tile_max_bytes=1024*1024;
  auto options=reference;options.pm_symmetry=s;
  if(parallel::world().rank==0) {
    auto saved_h=h;
    saved_h.core_indices.clear();
    saved_h.active_occ_indices={0,1,2,3};saved_h.virtual_indices={4,5,6,7};
    Matrix coefficients({8,4});
    for(std::size_t i=0;i<4;++i) coefficients(i,i)=1.;
    const auto directory=std::filesystem::temp_directory_path()/
        ("pm-hamiltonian-checkpoint-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    require(std::filesystem::create_directory(directory),"unique Hamiltonian checkpoint test directory");
    const auto path=directory/"hamiltonian.bin";
    write_hamiltonian_checkpoint(path,"synthetic-v1",coefficients,coefficients,h.fock,saved_h);
    const auto restored=read_hamiltonian_checkpoint(path,"synthetic-v1",coefficients,coefficients,h.fock,1024*1024);
    require(restored.eri_chemist.data()==h.eri_chemist.data() && restored.fock.data()==h.fock.data(),
        "Hamiltonian sidecar exact round trip");
    rejects([&]{(void)read_hamiltonian_checkpoint(path,"wrong",coefficients,coefficients,h.fock,1024*1024);});
    auto changed=coefficients;changed(0,0)+=1.e-12;
    rejects([&]{(void)read_hamiltonian_checkpoint(path,"synthetic-v1",changed,coefficients,h.fock,1024*1024);});
    rejects([&]{(void)read_hamiltonian_checkpoint(path,"synthetic-v1",coefficients,coefficients,h.fock,1);});
    rejects([&]{write_hamiltonian_checkpoint(path,"synthetic-v1",coefficients,coefficients,h.fock,saved_h);});
    const auto broken=directory/"truncated.bin";
    std::filesystem::copy_file(path,broken);
    std::filesystem::resize_file(broken,std::filesystem::file_size(broken)/2);
    rejects([&]{(void)read_hamiltonian_checkpoint(broken,"synthetic-v1",coefficients,coefficients,h.fock,1024*1024);});
    std::filesystem::remove(path);std::filesystem::remove(broken);std::filesystem::remove(directory);
  }
  std::size_t spatial_checkpoint_calls=0;
  bool observed_converged=false;
  options.checkpoint_interval=1;
  options.spatial_observer=[&](const SpatialRccsdProgress& progress) {
    ++spatial_checkpoint_calls;
    observed_converged=observed_converged||progress.converged;
    const auto t1=expand_spatial_rccsd_t1_to_spin_orbital(progress.amplitudes.t1);
    const auto t2=expand_spatial_rccsd_t2_to_spin_orbital(progress.amplitudes.t2);
    require(t1.dimensions()==std::array<std::size_t,2>{8,8} &&
            t2.dimensions()==std::array<std::size_t,4>{8,8,8,8},
            "PM spatial checkpoint expansion dimensions");
  };
  const auto a=solve_spatial_rccsd_reference(h,reference);
  const auto b=solve_spatial_rccsd_reference(h,options);
  require(a.converged && b.converged,"PM symmetry full CCSD convergence");
  require(spatial_checkpoint_calls>1 && observed_converged,
      "PM spatial checkpoint observer receives iterative and converged states");
  close(a.correlation_energy,b.correlation_energy,"PM symmetry converged energy");
  auto native_options=options;
  native_options.use_direct_spatial_singles=true;
  native_options.use_direct_spatial_doubles=true;
  native_options.use_tiled_spatial_pairs=true;
  native_options.use_pm_residual_orbits=true;
  native_options.use_root_replicated_diis=true;
  native_options.wabef_tile_max_bytes=4*1024*1024;
  const auto native=solve_spatial_rccsd_reference(h,native_options);
  require(native.converged,"native PM CCSD convergence");
  close(a.correlation_energy,native.correlation_energy,"native PM CCSD energy");
  auto native_restart_options=native_options;
  native_restart_options.initial_t1=expand_spatial_rccsd_t1_to_spin_orbital(native.amplitudes.t1);
  native_restart_options.initial_t2=expand_spatial_rccsd_t2_to_spin_orbital(native.amplitudes.t2);
  const auto native_restart=solve_spatial_rccsd_reference(h,native_restart_options);
  require(native_restart.converged,"PM representative amplitude restart");
  close(native.correlation_energy,native_restart.correlation_energy,"PM representative restart energy");
  for(std::size_t k=0;k<a.amplitudes.t1.size();++k)
    close(a.amplitudes.t1.data()[k],b.amplitudes.t1.data()[k],"PM symmetry converged T1");
  for(std::size_t k=0;k<a.amplitudes.t2.size();++k)
    close(a.amplitudes.t2.data()[k],b.amplitudes.t2.data()[k],"PM symmetry converged T2");
  options.initial_t1=expand_spatial_rccsd_t1_to_spin_orbital(a.amplitudes.t1);
  options.initial_t2=expand_spatial_rccsd_t2_to_spin_orbital(a.amplitudes.t2);
  const auto restart=solve_spatial_rccsd_reference(h,options);
  require(restart.converged,"PM symmetry amplitude restart convergence");
  close(a.correlation_energy,restart.correlation_energy,"PM symmetry restart energy");
  // partition_ccsd consumes spin-orbital amplitudes and therefore needs one
  // monomer label for each alpha/beta orbital, unlike the spatial triples API.
  const auto spin_labels=expand_spatial_labels({0,1,2,3});
  require(spin_labels==std::vector<int>({0,0,1,1,2,2,3,3}),
      "PM symmetry partition spin-label ordering");
  const auto pa=partition_ccsd(h,*options.initial_t1,*options.initial_t2,spin_labels,spin_labels);
  const auto pb=partition_ccsd(h,expand_spatial_rccsd_t1_to_spin_orbital(b.amplitudes.t1),
      expand_spatial_rccsd_t2_to_spin_orbital(b.amplitudes.t2),spin_labels,spin_labels);
  close(pa.sum,a.correlation_energy,"reference CCSD nM closure");
  close(pb.sum,b.correlation_energy,"symmetric CCSD nM closure");
  const auto pn=partition_ccsd(h,expand_spatial_rccsd_t1_to_spin_orbital(native.amplitudes.t1),
      expand_spatial_rccsd_t2_to_spin_orbital(native.amplitudes.t2),spin_labels,spin_labels);
  close(pn.sum,native.correlation_energy,"native PM CCSD nM closure");
  for(const auto& [label,energy]:pa.many_body_classes)
    close(energy,pn.many_body_classes.at(label),"native PM CCSD nM energy");
  for(const auto& [label,energy]:pa.many_body_classes)
    close(energy,pb.many_body_classes.at(label),"PM symmetry CCSD nM energy");
}
} // namespace
int main(int argc,char** argv) {
  parallel::Runtime runtime(argc,argv);
  try {
    if(argc==4 && std::string(argv[1])=="--check-water-cycle") {
      const std::string order=argv[3];
      require(order=="3" || order=="4","unsupported cyclic geometry order");
      check_geometry_file(argv[2],order=="3"?3:4);return 0;
    }
    require(argc==1,"usage: pm_symmetry_tests [--check-water-cycle geometry.xyz 3|4]");
#ifdef PM_SYMMETRY_TRIPLES_TEST
    test_saved_pm_inversion();
    test_adapted_localization();
#endif
    test_geometry();test_c3();test_equations();if(runtime.world().rank==0)std::cout<<"PM symmetry equation gates passed\n";return 0;}
  catch(const std::exception& e) {std::cerr<<"PM symmetry test failure: "<<e.what()<<'\n';runtime.abort(1);}
}
