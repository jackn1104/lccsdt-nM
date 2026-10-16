#include "localized_cc/lmp2_preintegral.hpp"
#include "localized_cc/parallel.hpp"
#include "modernqc/linalg/eigensolver.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace localized_cc {
namespace {
using Dense=modernqc::linalg::Matrix;
using Key=std::array<int,4>;
Key canonical(Key k) {
  if(k[0]>k[1]) std::swap(k[0],k[1]);
  if(k[2]>k[3]) std::swap(k[2],k[3]);
  return k;
}
std::size_t product(std::initializer_list<std::size_t> factors) {
  std::size_t n=1;
  for(auto x:factors) {
    if(x&&n>std::numeric_limits<std::size_t>::max()/x) throw std::overflow_error("LMP2 preintegral size overflow");
    n*=x;
  }
  return n;
}
void finite(double x) {if(!std::isfinite(x)) throw std::invalid_argument("nonfinite LMP2 preintegral quantity");}
std::map<int,std::vector<std::size_t>> groups(const std::vector<int>& labels) {
  std::map<int,std::vector<std::size_t>> g;
  for(std::size_t k=0;k<labels.size();++k) {
    if(labels[k]<0) throw std::invalid_argument("unassigned PM orbital");
    g[labels[k]].push_back(k);
  }
  return g;
}
Dense columns(const Dense& c,const std::vector<std::size_t>& ids) {
  Dense result(c.rows(),ids.size());
  for(std::size_t p=0;p<ids.size();++p) for(std::size_t mu=0;mu<c.rows();++mu) result(mu,p)=c(mu,ids[p]);
  return result;
}
std::size_t offset(const Lmp2OvovTile& t,std::size_t i,std::size_t j,std::size_t a,std::size_t b) {
  return t.offset+((i*t.indices[1].size()+j)*t.indices[2].size()+a)*t.indices[3].size()+b;
}
} // namespace

Lmp2PreintegralPlan plan_lmp2_preintegral_domains(const Matrix& q,
    const std::vector<int>& ol,const std::vector<int>& vl,double gap,double budget) {
  finite(gap);finite(budget);
  if(gap<=1e-10||budget<0||budget>2e-7||ol.empty()||vl.empty()||
     q.dimensions()!=std::array<std::size_t,2>{ol.size(),vl.size()})
    throw std::invalid_argument("invalid preintegral dimensions/gap/budget (maximum 2e-7 Eh)");
  const auto og=groups(ol),vg=groups(vl);
  Lmp2PreintegralPlan p;p.occupied_labels=ol;p.virtual_labels=vl;p.minimum_denominator=gap;p.budget=budget;
  p.full_elements=product({ol.size(),ol.size(),vl.size(),vl.size()});
  std::map<std::pair<int,int>,long double> pair_squares;
  long double total=0;
  for(std::size_t i=0;i<ol.size();++i) for(std::size_t a=0;a<vl.size();++a) {
    const double x=q(i,a);finite(x);if(x<0) throw std::invalid_argument("negative Schwarz bound");
    pair_squares[{ol[i],vl[a]}]+=static_cast<long double>(x)*x;
    total+=static_cast<long double>(x)*x;
  }
  p.full_source_norm_bound=static_cast<double>(total)*(1+1e-12);finite(p.full_source_norm_bound);
  std::map<Key,long double> squares;
  for(const auto& [im,ii]:og) for(const auto& [jm,jj]:og)
    for(const auto& [am,aa]:vg) for(const auto& [bm,bb]:vg) {
      const Key key=canonical({im,jm,am,bm});
      auto& b=p.blocks[key];b.labels=key;b.elements+=product({ii.size(),jj.size(),aa.size(),bb.size()});
      squares[key]+=pair_squares[{im,am}]*pair_squares[{jm,bm}];
    }
  std::vector<Key> order;
  for(auto& [key,b]:p.blocks) {
    b.monomer_span=std::set<int>(key.begin(),key.end()).size();
    b.source_norm_bound=std::sqrt(static_cast<double>(squares[key]))*(1+1e-12);finite(b.source_norm_bound);
    order.push_back(key);
  }
  std::stable_sort(order.begin(),order.end(),[&](const Key& a,const Key& b){return p.blocks.at(a).source_norm_bound<p.blocks.at(b).source_norm_bound;});
  long double omitted_square=0;
  for(const auto& key:order) {
    auto& b=p.blocks.at(key);
    const long double next=omitted_square+static_cast<long double>(b.source_norm_bound)*b.source_norm_bound;
    const double d=std::sqrt(static_cast<double>(next));
    const double error=3/gap*(2*p.full_source_norm_bound*d+d*d);finite(error);
    if(budget>0&&b.monomer_span>1&&error<=budget) {
      b.retained=false;omitted_square=next;p.energy_error_bound=error;
    }
    else p.retained_elements+=b.elements;
  }
  p.omitted_source_norm_bound=std::sqrt(static_cast<double>(omitted_square));
  return p;
}

Lmp2SelectedOvov allocate_lmp2_selected_ovov(const Lmp2PreintegralPlan& plan,std::size_t maximum_bytes) {
  // Reserve metadata too, rather than pretending the payload is the whole cache.
  const auto bytes=product({plan.retained_elements,sizeof(double)});
  const auto ng=groups(plan.occupied_labels),vg=groups(plan.virtual_labels);
  // Upper bound for copied index vectors, map nodes and tile structures.
  const auto orbital_count=plan.occupied_labels.size()+plan.virtual_labels.size();
  const auto per_tile=1024+product({std::size_t{2},sizeof(std::size_t),orbital_count});
  const auto metadata=product({ng.size(),ng.size(),vg.size(),vg.size(),per_tile});
  if(bytes>maximum_bytes||metadata>maximum_bytes-bytes)
    throw std::runtime_error("selected OVOV cache exceeds memory cap before allocation");
  Lmp2SelectedOvov r;r.plan=plan;r.occupied_local.resize(plan.occupied_labels.size());r.virtual_local.resize(plan.virtual_labels.size());
  for(const auto& [m,ids]:ng) for(std::size_t k=0;k<ids.size();++k) r.occupied_local[ids[k]]=k;
  for(const auto& [m,ids]:vg) for(std::size_t k=0;k<ids.size();++k) r.virtual_local[ids[k]]=k;
  std::size_t count=0;
  for(const auto& [im,ii]:ng) for(const auto& [jm,jj]:ng)
    for(const auto& [am,aa]:vg) for(const auto& [bm,bb]:vg) {
      const Key key{im,jm,am,bm};
      if(!plan.blocks.at(canonical(key)).retained) continue;
      r.lookup.emplace(key,r.tiles.size());r.tiles.push_back({key,{ii,jj,aa,bb},count});
      count+=product({ii.size(),jj.size(),aa.size(),bb.size()});
    }
  if(count!=plan.retained_elements) throw std::runtime_error("preintegral plan/store count mismatch");
  r.values.resize(count,0.);return r;
}

double Lmp2SelectedOvov::operator()(std::size_t i,std::size_t j,std::size_t a,std::size_t b) const {
  const Key key{plan.occupied_labels.at(i),plan.occupied_labels.at(j),plan.virtual_labels.at(a),plan.virtual_labels.at(b)};
  // Zero is allowed only for an explicitly bounded omission, never a cache miss.
  if(!plan.blocks.at(canonical(key)).retained) return 0;
  const auto& t=tiles.at(lookup.at(key));
  return values.at(offset(t,occupied_local.at(i),occupied_local.at(j),virtual_local.at(a),virtual_local.at(b)));
}

void complete_lmp2_selected_ao_block(
    const modernqc::integrals::BatchedFirstIndexAoBlock& block,
    const Dense& co,const Dense& cv,const std::vector<std::size_t>& owned,
    Lmp2SelectedOvov& out,std::size_t cap) {
  const auto nao=co.rows(),ni=owned.size(),nl=block.lambda_ao_count;
  if(co.columns()!=out.plan.occupied_labels.size()||cv.columns()!=out.plan.virtual_labels.size()||
     nao!=cv.rows()||block.ao_count!=nao||block.occupied_count!=ni||!ni||!nl||
     block.lambda_ao_first>nao||nl>nao-block.lambda_ao_first||block.point_values.size()!=1||
     block.point_values[0].size()!=product({nl,nao,ni,nao}))
    throw std::invalid_argument("invalid LMP2 first-index block shape");
  // Conservative simultaneous packing/work allowance, checked before packing.
  const auto peak=product({std::size_t{10},sizeof(double),nao,nl,ni,std::max(nao,cv.columns())});
  if(peak>cap) throw std::runtime_error("LMP2 completion workspace exceeds cap");
  Dense x(nao,product({nl,nao,ni}));x.values()=block.point_values[0];
  const auto vg=groups(out.plan.virtual_labels),og=groups(out.plan.occupied_labels);
  std::vector<std::size_t> owned_position(co.columns(),std::numeric_limits<std::size_t>::max());
  for(std::size_t k=0;k<ni;++k) {
    if(owned.at(k)>=co.columns()||owned_position.at(owned[k])!=std::numeric_limits<std::size_t>::max()) throw std::invalid_argument("invalid owned occupied indices");
    owned_position[owned[k]]=k;
  }
  for(const auto& [am,aa]:vg) {
    bool need_a=false;
    for(const auto& tile:out.tiles) if(tile.labels[2]==am)
      for(auto i:tile.indices[0]) need_a=need_a||owned_position[i]!=std::numeric_limits<std::size_t>::max();
    if(!need_a) continue;
    const auto second=modernqc::linalg::multiply(modernqc::linalg::transpose(columns(cv,aa)),x);
    Dense packed_sigma(nao,product({ni,aa.size(),nl}));
    for(std::size_t i=0;i<ni;++i) for(std::size_t a=0;a<aa.size();++a)
      for(std::size_t l=0;l<nl;++l) for(std::size_t s=0;s<nao;++s)
        packed_sigma(s,(i*aa.size()+a)*nl+l)=second(a,(l*nao+s)*ni+i);
    for(const auto& [bm,bb]:vg) {
      bool need_b=false;
      for(const auto& tile:out.tiles) if(tile.labels[2]==am&&tile.labels[3]==bm)
        for(auto i:tile.indices[0]) need_b=need_b||owned_position[i]!=std::numeric_limits<std::size_t>::max();
      if(!need_b) continue;
      const auto third=modernqc::linalg::multiply(modernqc::linalg::transpose(columns(cv,bb)),packed_sigma);
      Dense packed_lambda(nl,product({ni,aa.size(),bb.size()}));
      for(std::size_t i=0;i<ni;++i) for(std::size_t a=0;a<aa.size();++a)
        for(std::size_t b=0;b<bb.size();++b) for(std::size_t l=0;l<nl;++l)
          packed_lambda(l,(i*aa.size()+a)*bb.size()+b)=third(b,(i*aa.size()+a)*nl+l);
      for(const auto& [jm,jj]:og) {
        bool needed=false;
        for(auto i:owned) needed=needed||out.lookup.count({out.plan.occupied_labels[i],jm,am,bm});
        if(!needed) continue;
        Dense left(jj.size(),nl);
        for(std::size_t j=0;j<jj.size();++j) for(std::size_t l=0;l<nl;++l) left(j,l)=co(block.lambda_ao_first+l,jj[j]);
        const auto final=modernqc::linalg::multiply(left,packed_lambda);
        for(std::size_t i=0;i<ni;++i) {
          const auto found=out.lookup.find({out.plan.occupied_labels[owned[i]],jm,am,bm});
          if(found==out.lookup.end()) continue;
          const auto& tile=out.tiles[found->second];
          for(std::size_t j=0;j<jj.size();++j) for(std::size_t a=0;a<aa.size();++a) for(std::size_t b=0;b<bb.size();++b)
            out.values.at(offset(tile,out.occupied_local[owned[i]],j,a,b))+=final(j,(i*aa.size()+a)*bb.size()+b);
        }
      }
    }
  }
}

Lmp2SelectedOvov build_lmp2_selected_ovov(
    const modernqc::integrals::Libint2IntegralProvider& provider,
    const Dense& co,const Dense& cv,const Lmp2PreintegralPlan& plan,
    std::size_t cap,std::size_t threads,const std::function<void(const std::string&)>& progress) {
  const auto world=parallel::world();
  Lmp2SelectedOvov result;
  std::vector<std::size_t> owned;
  for(std::size_t i=world.rank;i<co.columns();i+=world.size) owned.push_back(i);
  std::string error;
  try {
    result=allocate_lmp2_selected_ovov(plan,cap/4);
    if(!owned.empty()) {
      modernqc::integrals::BatchedFirstIndexAoOptions opt{};
      opt.occupied_first=0;opt.occupied_count=owned.size();opt.target_lambda_ao_block_size=4;
      opt.threads=threads;opt.schwarz_threshold=0;opt.report_progress=false;
      opt.maximum_additional_memory_bytes=cap/4;
      // One AO traversal per nonempty rank, not one pass per selected tile.
      provider.for_each_batched_first_index_ao_block({columns(co,owned)},opt,
          [&](const auto& block) {
            complete_lmp2_selected_ao_block(block,co,cv,owned,result,cap/2);
            if(world.rank==0&&progress) progress("selected LMP2 AO block complete; lambda_first="+std::to_string(block.lambda_ao_first));
          });
    }
  } catch(const std::exception& e) {error=e.what();}
  std::vector<double> failed{error.empty()?0.:1.};parallel::sum_in_place(failed);
  if(failed[0]) throw std::runtime_error(error.empty()?"selected LMP2 transform failed on another rank":error);
  // Selected blocks are replicated only after all AO work is complete.
  // No MPI calls inside the integral-contraction loop.
  parallel::sum_in_place(result.values);
  for(double x:result.values) finite(x);
  return result;
}

void write_lmp2_preintegral_plan(const std::filesystem::path& dir,const Lmp2PreintegralPlan& p) {
  if(!std::filesystem::create_directory(dir)) throw std::runtime_error("preintegral plan output exists");
  std::ofstream f(dir/"blocks.csv");f.exceptions(std::ios::failbit|std::ios::badbit);f<<std::setprecision(17);
  f<<"occ_m1,occ_m2,vir_m1,vir_m2,nM,elements,retained,source_norm_bound\n";
  for(const auto& [key,b]:p.blocks) {
    for(int m:key) f<<m<<',';
    f<<b.monomer_span<<','<<b.elements<<','<<int(b.retained)<<','<<b.source_norm_bound<<'\n';
  }
  f.close();
  std::set<std::array<int,3>> domains;
  for(const auto& [key,b]:p.blocks) if(b.retained) {
    domains.insert({key[0],key[1],key[2]});domains.insert({key[0],key[1],key[3]});
  }
  std::ofstream d(dir/"pair_domains.csv");d.exceptions(std::ios::failbit|std::ios::badbit);
  d<<"occ_m1,occ_m2,virtual_monomer\n";
  for(const auto& key:domains) d<<key[0]<<','<<key[1]<<','<<key[2]<<'\n';
  d.close();
  std::ofstream m(dir/"metadata.txt");m.exceptions(std::ios::failbit|std::ios::badbit);m<<std::setprecision(17);
  m<<"selection_before_OVOV=1\nfull_spatial_MO_eri_bytes=0\nno_density_fitting=1\n"
   <<"full_source_norm_bound="<<p.full_source_norm_bound<<"\nomitted_source_norm_bound="<<p.omitted_source_norm_bound
   <<"\nminimum_denominator="<<p.minimum_denominator<<"\nenergy_error_bound_hartree="<<p.energy_error_bound
   <<"\nbudget_hartree="<<p.budget<<"\nfull_elements="<<p.full_elements<<"\nretained_elements="<<p.retained_elements
   <<"\nccsd_t_accuracy_validated=0\n";m.close();
}

void run_lmp2_preintegral_pilot(
    const modernqc::molecule::Molecule& molecule,
    const modernqc::integrals::Libint2IntegralProvider& provider,
    const Dense& co,const Dense& cv,const Dense& fock,
    const std::vector<int>& ol,const std::vector<int>& vl,
    const std::filesystem::path& checkpoint,const std::string& fingerprint,
    double budget,std::size_t cap,std::size_t threads,
    const std::function<void(const std::string&)>& progress) {
  const auto world=parallel::world();
  const auto start=std::chrono::steady_clock::now();
  const std::filesystem::path plan_dir=checkpoint.string()+".preintegral-plan";
  std::vector<int> status(1,0);std::string failure;
  Matrix q({co.columns(),cv.columns()}),fo({co.columns(),co.columns()}),fv({cv.columns(),cv.columns()});
  double gap=0;
  if(world.rank==0) try {
    if(std::filesystem::exists(plan_dir)) throw std::runtime_error("preintegral output exists; use a new checkpoint copy");
    progress("preintegral LMP2 shell-pair bounds started; full PM coefficient tails; no OVOV yet");
    auto foo=modernqc::linalg::multiply(modernqc::linalg::transpose(co),modernqc::linalg::multiply(fock,co));
    auto fvv=modernqc::linalg::multiply(modernqc::linalg::transpose(cv),modernqc::linalg::multiply(fock,cv));
    const auto eo=modernqc::linalg::diagonalize_symmetric(foo);
    const auto ev=modernqc::linalg::diagonalize_symmetric(fvv);
    gap=2*(ev.eigenvalues.front()-eo.eigenvalues.back());
    for(std::size_t i=0;i<co.columns();++i) for(std::size_t j=0;j<co.columns();++j) fo(i,j)=foo(i,j);
    for(std::size_t a=0;a<cv.columns();++a) for(std::size_t b=0;b<cv.columns();++b) fv(a,b)=fvv(a,b);
    q=lmp2_ao_schwarz_pair_bounds(molecule,provider,co,cv);
  } catch(const std::exception& e) {status[0]=1;failure=e.what();}
  parallel::broadcast_in_place(status);
  if(status[0]) throw std::runtime_error(world.rank==0?failure:"preintegral root bounds failed");
  std::vector<double> packed=q.data();packed.push_back(gap);
  parallel::broadcast_in_place(packed);
  std::copy_n(packed.begin(),q.size(),q.data().begin());gap=packed.back();
  const auto plan=plan_lmp2_preintegral_domains(q,ol,vl,gap,budget);
  if(world.rank==0) try {
    write_lmp2_preintegral_plan(plan_dir,plan);
    progress("preintegral plan saved BEFORE OVOV; retained="+std::to_string(plan.retained_elements)+"/"+std::to_string(plan.full_elements));
  } catch(const std::exception& e) {status[0]=1;failure=e.what();}
  parallel::broadcast_in_place(status);
  if(status[0]) throw std::runtime_error(world.rank==0?failure:"preintegral plan write failed");
  const auto before_transform=std::chrono::steady_clock::now();
  const auto cache=build_lmp2_selected_ovov(provider,co,cv,plan,cap,threads,progress);
  const auto after_transform=std::chrono::steady_clock::now();
  if(world.rank==0) try {
    // Audit actual completed integrals against the precomputed pair bound.
    // No unselected ERI is evaluated to perform this check.
    for(const auto& tile:cache.tiles) {
      auto k=tile.offset;
      for(auto i:tile.indices[0]) for(auto j:tile.indices[1])
        for(auto a:tile.indices[2]) for(auto b:tile.indices[3]) {
          const double upper=q(i,a)*q(j,b);
          if(std::abs(cache.values.at(k++))>upper*(1+1e-12)+1e-14)
            throw std::runtime_error("selected integral exceeds preintegral Schwarz bound");
        }
    }
    Lmp2DomainOptions opt;opt.maximum_workspace_bytes=cap;opt.omission_budget_hartree=0;
    opt.observer=[&](std::size_t k,double r,double e) {
      std::ostringstream s;s<<std::scientific<<std::setprecision(12)
        <<"preintegral LMP2 PCG iteration="<<k<<" residual_l2="<<r<<" solve_bound_Eh="<<e;progress(s.str());
    };
    const auto result=select_lmp2_domains(fo,fv,[&](auto i,auto j,auto a,auto b){return cache(i,j,a,b);},ol,vl,opt);
    if(plan.energy_error_bound+result.solve_energy_error_bound>1e-6)
      throw std::runtime_error("preintegral plus MP2 solve error budget exceeds 1e-6 Eh");
    write_lmp2_domain_audit(checkpoint.string()+".lmp2-domains",result,opt,ol,vl,fingerprint);
    std::ofstream ledger(plan_dir/"result.txt");ledger.exceptions(std::ios::failbit|std::ios::badbit);
    ledger<<std::setprecision(17)<<"mp2_source_screening_bound_hartree="<<plan.energy_error_bound
      <<"\nmp2_solve_bound_hartree="<<result.solve_energy_error_bound
      <<"\nselected_ovov_payload_bytes_per_rank="<<cache.values.size()*sizeof(double)
      <<"\nplan_seconds="<<std::chrono::duration<double>(before_transform-start).count()
      <<"\nselected_transform_seconds="<<std::chrono::duration<double>(after_transform-before_transform).count()
      <<"\nfull_spatial_MO_eri_bytes=0\nenergy_semantics=full_fock_mp2_with_screened_source\n";ledger.close();
    for(std::size_t n=0;n<4;++n) {
      std::ostringstream s;s<<std::setprecision(17)<<"preintegral LMP2 M"<<n+1<<"="<<result.full_nm[n]<<" Eh";progress(s.str());
    }
    std::ofstream done(plan_dir/"COMPLETE");done.exceptions(std::ios::failbit|std::ios::badbit);
    done<<"MP2 source selection complete; domains not yet validated for CCSD(T) accuracy\n";done.close();
  } catch(const std::exception& e) {status[0]=1;failure=e.what();}
  parallel::broadcast_in_place(status);
  if(status[0]) throw std::runtime_error(world.rank==0?failure:"preintegral MP2 root solve failed");
}
} // namespace localized_cc
