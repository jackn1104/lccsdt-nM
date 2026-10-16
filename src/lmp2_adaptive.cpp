#include "localized_cc/lmp2_adaptive.hpp"
#include "localized_cc/parallel.hpp"
#include <cstdint>
#include "modernqc/linalg/eigensolver.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace localized_cc {
namespace {
using Dense=modernqc::linalg::Matrix;
using Groups=std::vector<std::vector<std::size_t>>;
std::size_t mul(std::size_t a,std::size_t b) {
  if(a&&b>std::numeric_limits<std::size_t>::max()/a) throw std::overflow_error("adaptive MP2 size overflow");
  return a*b;
}
std::size_t mp2_checked_add(std::size_t a,std::size_t b) {
  if(b>std::numeric_limits<std::size_t>::max()-a) throw std::overflow_error("adaptive MP2 size sum overflow");
  return a+b;
}
double mp2_local_norm(const std::vector<double>& x) {
  long double n=0;for(double v:x) {if(!std::isfinite(v)) throw std::runtime_error("nonfinite adaptive MP2 vector");n+=static_cast<long double>(v)*v;}
  return std::sqrt(static_cast<double>(n));
}
double mp2_global_dot(const std::vector<double>& x,const std::vector<double>& y) {
  long double s=0;for(std::size_t k=0;k<x.size();++k) s+=static_cast<long double>(x[k])*y.at(k);
  std::vector<double> v{static_cast<double>(s)};parallel::sum_in_place(v);
  if(!std::isfinite(v[0])) throw std::runtime_error("nonfinite adaptive MP2 dot product");return v[0];
}
double mp2_global_norm(const std::vector<double>& x) {
  return std::sqrt(std::max(0.,mp2_global_dot(x,x)));
}
Groups groups(const std::vector<int>& labels,std::size_t width) {
  if(!width||labels.empty()) throw std::invalid_argument("empty adaptive MP2 space/tile");
  std::map<int,std::vector<std::size_t>> m;
  for(std::size_t p=0;p<labels.size();++p) {if(labels[p]<0) throw std::invalid_argument("unassigned PM orbital");m[labels[p]].push_back(p);}
  Groups g;for(const auto& [label,ids]:m) for(std::size_t k=0;k<ids.size();k+=width)
    g.emplace_back(ids.begin()+k,ids.begin()+std::min(ids.size(),k+width));
  return g;
}
Dense fock(const Matrix& x,std::size_t n) {
  if(x.dimensions()!=std::array<std::size_t,2>{n,n}) throw std::invalid_argument("adaptive Fock shape");
  Dense d(n,n);for(std::size_t i=0;i<n;++i) for(std::size_t j=0;j<n;++j) {
    if(!std::isfinite(x(i,j))||std::abs(x(i,j)-x(j,i))>1e-12) throw std::invalid_argument("adaptive Fock nonfinite/asymmetric");
    d(i,j)=0.5*x(i,j)+0.5*x(j,i);
  }return d;
}
Mp2TileKey orbit(Mp2TileKey k) {
  if(k[0]>k[1]) std::swap(k[0],k[1]);if(k[2]>k[3]) std::swap(k[2],k[3]);return k;
}
std::size_t elements(const Mp2AdaptiveTile& b) {
  std::size_t n=1;for(const auto& ids:b.indices) n=mul(n,ids.size());return n;
}
// Exact axis contraction between two selected orbital tiles, behind BLAS.
void axis_add(const Dense& f,const Mp2AdaptiveTile& dst,const Mp2AdaptiveTile& src,
    std::size_t axis,double sign,const std::vector<double>& x,std::size_t off,
    std::vector<double>& y,std::size_t out) {
  std::size_t before=1,after=1;
  for(std::size_t a=0;a<axis;++a) before*=src.indices[a].size();
  for(std::size_t a=axis+1;a<4;++a) after*=src.indices[a].size();
  const auto ns=src.indices[axis].size(),nd=dst.indices[axis].size();
  Dense left(nd,ns),packed(ns,before*after);
  for(std::size_t i=0;i<nd;++i) for(std::size_t j=0;j<ns;++j) left(i,j)=sign*f(dst.indices[axis][i],src.indices[axis][j]);
  for(std::size_t l=0;l<before;++l) for(std::size_t j=0;j<ns;++j) for(std::size_t r=0;r<after;++r)
    packed(j,l*after+r)=x.at(off+(l*ns+j)*after+r);
  const auto v=modernqc::linalg::multiply(left,packed);
  for(std::size_t l=0;l<before;++l) for(std::size_t i=0;i<nd;++i) for(std::size_t r=0;r<after;++r)
    y.at(out+(l*nd+i)*after+r)+=v(i,l*after+r);
}
}

std::size_t mp2_tile_owner(Mp2TileKey key,std::size_t ranks) {
  if(!ranks) throw std::invalid_argument("zero MP2 owner ranks");
  key=orbit(key);
  std::uint64_t h=1469598103934665603ULL;
  for(auto v:key) {h^=static_cast<std::uint64_t>(v);h*=1099511628211ULL;}
  h^=h>>30;h*=0xbf58476d1ce4e5b9ULL;h^=h>>27;
  h*=0x94d049bb133111ebULL;h^=h>>31;
  return static_cast<std::size_t>(h%ranks);
}

Mp2AdaptiveResult solve_adaptive_mp2(const Matrix& foo,const Matrix& fvv,const Matrix& q,
    const std::vector<int>& ol,const std::vector<int>& vl,const Mp2AdaptiveOptions& opt,
    const Mp2TileBuilder& builder,const std::function<void(const std::string&)>& progress) {
  if(!builder||!opt.max_iterations||!std::isfinite(opt.accuracy)||opt.accuracy<=0||opt.accuracy>1e-7)
    throw std::invalid_argument("adaptive MP2 requires accuracy <=1e-7 Eh and a builder");
  const auto world=parallel::world();
  const auto owned=[&](const Mp2AdaptiveTile& b){return mp2_tile_owner(b.key,world.size)==world.rank;};
  const auto og=groups(ol,opt.occupied_tile),vg=groups(vl,opt.virtual_tile);
  const auto fo=fock(foo,ol.size()),fv=fock(fvv,vl.size());
  if(q.dimensions()!=std::array<std::size_t,2>{ol.size(),vl.size()}) throw std::invalid_argument("adaptive pair-bound shape");
  Mp2AdaptiveResult result;
  result.gap=2*(modernqc::linalg::diagonalize_symmetric(fv).eigenvalues.front()-modernqc::linalg::diagonalize_symmetric(fo).eigenvalues.back());
  if(!std::isfinite(result.gap)||result.gap<=1e-10) throw std::invalid_argument("adaptive MP2 nonpositive gap");
  result.full_elements=mul(mul(ol.size(),ol.size()),mul(vl.size(),vl.size()));
  long double b2=0;for(double v:q.data()) {if(!std::isfinite(v)||v<0) throw std::invalid_argument("invalid pair bound");b2+=static_cast<long double>(v)*v;}
  result.full_source_bound=static_cast<double>(b2)*(1+1e-12);
  if(!std::isfinite(result.full_source_bound)) throw std::runtime_error("adaptive source norm overflow");
  const auto tile_count=mul(mul(og.size(),og.size()),mul(vg.size(),vg.size()));
  // Metadata enumerates tiles, not orbital quadruples. Not an O(N) planner yet.
  const auto index_bytes=mul(mul(2,std::max(opt.occupied_tile,opt.virtual_tile)),2*sizeof(std::size_t));
  if(index_bytes>std::numeric_limits<std::size_t>::max()-1024) throw std::overflow_error("adaptive metadata overflow");
  const auto metadata=mul(tile_count,1024+index_bytes);
  if(metadata>opt.maximum_bytes/4) throw std::runtime_error("adaptive MP2 metadata exceeds memory cap");
  std::map<Mp2TileKey,Mp2AdaptiveTile> all;
  std::set<Mp2TileKey> wanted;
  for(std::size_t i=0;i<og.size();++i) for(std::size_t j=0;j<og.size();++j)
    for(std::size_t a=0;a<vg.size();++a) for(std::size_t b=0;b<vg.size();++b) {
      Mp2AdaptiveTile t;t.key={i,j,a,b};t.indices={og[i],og[j],vg[a],vg[b]};
      t.span=std::set<int>{ol[og[i][0]],ol[og[j][0]],vl[vg[a][0]],vl[vg[b][0]]}.size();
      long double l=0,r=0;
      for(auto p:og[i]) for(auto v:vg[a]) l+=static_cast<long double>(q(p,v))*q(p,v);
      for(auto p:og[j]) for(auto v:vg[b]) r+=static_cast<long double>(q(p,v))*q(p,v);
      t.source_bound=std::sqrt(static_cast<double>(l*r))*(1+1e-12);
      if(!std::isfinite(t.source_bound)) throw std::runtime_error("adaptive source bound overflow");
      if(opt.full_reference||t.span==1) wanted.insert(orbit(t.key));
      all.emplace(t.key,std::move(t));
    }
  if(wanted.empty()) wanted.insert(orbit(all.begin()->first));
  const double B=result.full_source_bound;
  // Select source-bound exchange orbits BEFORE any general AO quartet work.
  // This is a scheduling heuristic, not acceptance: the full residual ledger
  // below still controls the final energy. Suffix sums avoid subtracting a
  // tiny omitted tail from a large total bound.
  if(opt.efficient_planning&&!opt.full_reference) {
    const auto initial_wanted=wanted;
    std::map<Mp2TileKey,long double> scores;
    for(const auto& [key,tile]:all) if(!wanted.count(orbit(key)))
      scores[orbit(key)]+=static_cast<long double>(tile.source_bound)*tile.source_bound;
    std::vector<std::pair<Mp2TileKey,long double>> sorted(scores.begin(),scores.end());
    std::stable_sort(sorted.begin(),sorted.end(),[](const auto& a,const auto& b){return a.second>b.second;});
    std::vector<long double> suffix(sorted.size()+1,0.L);
    for(std::size_t k=sorted.size();k-->0;) suffix[k]=suffix[k+1]+sorted[k].second;
    const long double goal=opt.accuracy*result.gap/(12*std::max(B,1.));
    for(std::size_t k=0;k<sorted.size()&&suffix[k]>goal*goal;++k) wanted.insert(sorted[k].first);
    std::size_t predicted=0;
    for(const auto& [key,tile]:all) if(wanted.count(orbit(key))) predicted+=elements(tile);
    if(static_cast<long double>(predicted)>=0.8L*result.full_elements) {
      for(const auto& [key,tile]:all) wanted.insert(orbit(key));
      result.full_domain_fallback=true;
    } else wanted=initial_wanted; // retain the initial sparse selection
  }
  for(const auto& [key,tile]:all) if(wanted.count(orbit(key))) result.initial_elements+=elements(tile);
  if(progress&&opt.efficient_planning) progress("adaptive MP2 preflight selected="+
      std::to_string(result.initial_elements)+"/"+std::to_string(result.full_elements)+
      "; full-domain scheduling fallback="+std::to_string(result.full_domain_fallback)+
      "; final residual/energy gate unchanged");
  // Integral budget is independent of the number of adaptive rounds. Each
  // ordered tile receives the same norm allowance once, including exchanges.
  const double integral_allow=opt.full_reference?0:std::min(1.,opt.accuracy*result.gap/(100*std::max(B,1.)));
  double integral_square=0;
  std::map<Mp2TileKey,std::size_t> active;
  while(true) {
    std::vector<Mp2AdaptiveTile> added;
    std::size_t total=result.active_elements;
    for(const auto& [key,t]:all) if(wanted.count(orbit(key))&&!active.count(key)) {total+=elements(t);added.push_back(t);}
    std::size_t local=result.local_elements,max_tile=result.halo_peak_elements;
    for(const auto& b:added) {if(owned(b)) local+=elements(b);max_tile=std::max(max_tile,elements(b));}
    // 16 local vectors plus bounded one-tile halo and BLAS packing workspace.
    const auto payload=mp2_checked_add(mul(local,16*sizeof(double)),mul(max_tile,8*sizeof(double)));
    std::vector<double> memory_failure{double(payload>opt.maximum_bytes-metadata)};
    parallel::sum_in_place(memory_failure);
    if(memory_failure[0]) throw std::runtime_error("adaptive owner payload exceeds a rank's memory cap");
    result.local_elements=local;result.halo_peak_elements=max_tile;
    result.workspace_bytes=metadata+payload;
    if(added.empty()) throw std::runtime_error("adaptive domain expansion made no progress");
    const double allowance=integral_allow*std::sqrt(double(added.size())/double(tile_count));
    const double error=builder(added,allowance);
    if(!std::isfinite(error)||error<0||error>allowance*(1+1e-10)+1e-30) throw std::runtime_error("AO integral omission budget exceeded");
    integral_square+=error*error;result.integral_error=std::sqrt(integral_square);
    std::vector<double> invalid{0.};
    for(const auto& b:added) {
      if(!owned(b)) {if(!b.integrals.empty()||!b.amplitudes.empty()) invalid[0]=1.;continue;}
      if(b.integrals.size()!=elements(b)) {invalid[0]=1.;continue;}
      bool finite=true;for(double v:b.integrals) finite=finite&&std::isfinite(v);
      if(!finite||mp2_local_norm(b.integrals)>b.source_bound+error+1e-12*std::max(1.,b.source_bound)) invalid[0]=1.;
    }
    parallel::sum_in_place(invalid);
    if(invalid[0]) throw std::runtime_error("adaptive builder ownership, shape or source-bound violation");
    for(auto& b:added) {
      b.amplitudes.assign(b.integrals.size(),0);
      active.emplace(b.key,result.tiles.size());result.tiles.push_back(std::move(b));
    }
    result.active_elements=total;++result.rounds;
    std::vector<std::size_t> offsets;std::size_t off=0;
    std::vector<double> g,t,diag;
    g.reserve(local);t.reserve(local);diag.reserve(local);
    for(const auto& tile:result.tiles) {
      offsets.push_back(off);
      if(!owned(tile)) continue;
      off+=tile.integrals.size();
      g.insert(g.end(),tile.integrals.begin(),tile.integrals.end());
      t.insert(t.end(),tile.amplitudes.begin(),tile.amplitudes.end());
      for(auto i:tile.indices[0]) for(auto j:tile.indices[1]) for(auto a:tile.indices[2]) for(auto b:tile.indices[3]) {
        const double d=fv(a,a)+fv(b,b)-fo(i,i)-fo(j,j);
        if(d<=0||!std::isfinite(d)) throw std::runtime_error("invalid adaptive preconditioner");diag.push_back(d);
      }
    }
    // Metadata is replicated, numerical vectors are not. Build exact Fock
    // dependencies once per round. Zero subblocks need no halo communication.
    struct Edge {std::size_t destination,axis;};
    using Consumers=std::map<std::size_t,std::vector<Edge>>;
    std::vector<double> rank_budgets(world.size,0.);
    rank_budgets[world.rank]=static_cast<double>(opt.maximum_bytes-result.workspace_bytes);
    parallel::sum_in_place(rank_budgets);
    const double dependency_budget=*std::min_element(rank_budgets.begin(),rank_budgets.end());
    std::size_t dependency_bytes=mul(result.tiles.size(),sizeof(Consumers));
    if(static_cast<double>(dependency_bytes)>dependency_budget) throw std::runtime_error("MP2 dependency metadata exceeds rank cap");
    std::vector<Consumers> dependencies(result.tiles.size());
    for(std::size_t n=0;n<result.tiles.size();++n) {
      const auto& src=result.tiles[n];
      for(std::size_t ax=0;ax<4;++ax) for(std::size_t k=0;k<(ax<2?og.size():vg.size());++k) {
        auto key=src.key;key[ax]=k;const auto it=active.find(key);if(it==active.end()) continue;
        const auto& dst=result.tiles[it->second];const auto& f=ax<2?fo:fv;
        bool nonzero=false;
        for(auto i:dst.indices[ax]) for(auto j:src.indices[ax]) nonzero=nonzero||f(i,j)!=0.;
        if(nonzero) {
          // Account before allocation; conservative map/vector overhead per edge.
          dependency_bytes=mp2_checked_add(dependency_bytes,128+2*sizeof(Edge));
          if(static_cast<double>(dependency_bytes)>dependency_budget)
            throw std::runtime_error("MP2 Fock dependency metadata exceeds rank cap");
          dependencies[n][mp2_tile_owner(key,world.size)].push_back({it->second,ax});
        }
      }
    }
    result.workspace_bytes+=dependency_bytes;
    auto apply=[&](const std::vector<double>& x,std::vector<double>& y) {
      std::fill(y.begin(),y.end(),0.);
      std::vector<double> buffer;buffer.reserve(max_tile);
      for(std::size_t n=0;n<result.tiles.size();++n) {
        const auto& src=result.tiles[n];const auto owner=mp2_tile_owner(src.key,world.size);
        // One bounded scratch tile; only actual consumers receive it.
        buffer.resize(elements(src));
        if(world.rank==owner) std::copy_n(x.begin()+offsets[n],buffer.size(),buffer.begin());
        for(const auto& [peer,edges]:dependencies[n]) {
          parallel::transfer_tile(buffer,owner,peer);
          if(peer!=owner) ++result.halo_transfers;
          if(world.rank!=peer) continue;
          for(const auto& edge:edges)
            axis_add(edge.axis<2?fo:fv,result.tiles[edge.destination],src,
              edge.axis,edge.axis<2?-1.:1.,buffer,0,y,offsets[edge.destination]);
        }
      }
    };
    std::vector<double> r(local),z(local),p(local),ap(local);
    apply(t,ap);for(std::size_t k=0;k<local;++k) {r[k]=-g[k]-ap[k];z[k]=r[k]/diag[k];p[k]=z[k];}
    long double omitted_source_square=0;
    for(const auto& [key,tile]:all) if(!active.count(key))
      omitted_source_square+=static_cast<long double>(tile.source_bound)*tile.source_bound;
    // Tighten the initial Schwarz source norm using already constructed tiles;
    // this does not evaluate a single additional integral.
    const double energy_source_bound=std::min(B,
      std::hypot(mp2_global_norm(g),std::sqrt(static_cast<double>(omitted_source_square)))*(1+1e-12)+result.integral_error);
    result.full_source_bound=energy_source_bound;
    const double strict_target=std::min(1e-11,opt.accuracy*result.gap/(100*std::max(3*energy_source_bound,1.)));
    // Our conservative certificate includes the positive omitted source term.
    // If THAT certificate cannot pass even with zero active residual, avoid
    // over-solving this preliminary domain. This is not a lower bound on the
    // physical error. Final acceptance always uses the unchanged strict target.
    const bool preliminary=opt.efficient_planning&&
      3*energy_source_bound/result.gap*std::sqrt(static_cast<double>(omitted_source_square))>opt.accuracy*0.8;
    const double solve_target=preliminary?std::max(strict_target,std::min(1e-4,0.01*mp2_global_norm(g))):strict_target;
    if(progress&&preliminary) progress("adaptive MP2 preliminary solve; domain cannot pass current omission certificate; strict final gate retained");
    double rz=mp2_global_dot(r,z);bool converged=mp2_global_norm(r)<=solve_target;
    for(std::size_t iteration=0;!converged&&iteration<opt.max_iterations;++iteration) {
      apply(p,ap);const double pap=mp2_global_dot(p,ap);
      if(pap<=0||rz<=0) throw std::runtime_error("adaptive CG lost positive definiteness");
      const double alpha=rz/pap;
      for(std::size_t k=0;k<local;++k) {t[k]+=alpha*p[k];r[k]-=alpha*ap[k];}
      const bool refresh=mp2_global_norm(r)<=solve_target||(iteration+1)%25==0;
      if(refresh) {apply(t,ap);for(std::size_t k=0;k<local;++k) r[k]=-g[k]-ap[k];}
      const double residual_norm=mp2_global_norm(r);
      converged=residual_norm<=solve_target;++result.iterations;
      if(preliminary) ++result.preliminary_iterations;
      if(progress&&(iteration+1)%10==0) {
        std::ostringstream message;message<<"adaptive MP2 projected iteration="<<iteration+1
          <<"; residual="<<residual_norm<<"; target="<<solve_target;progress(message.str());
      }
      if(converged) break;
      for(std::size_t k=0;k<local;++k) z[k]=r[k]/diag[k];
      const double next=mp2_global_dot(r,z),beta=next/rz;
      for(std::size_t k=0;k<local;++k) p[k]=z[k]+(refresh?0:beta*p[k]);rz=next;
    }
    if(!converged) throw std::runtime_error("adaptive MP2 projected solve failed; no result accepted");
    std::vector<double> tn(result.tiles.size());
    for(std::size_t n=0;n<result.tiles.size();++n) {
      if(!owned(result.tiles[n])) continue;
      auto& v=result.tiles[n].amplitudes;std::copy_n(t.begin()+offsets[n],v.size(),v.begin());tn[n]=mp2_local_norm(v);
    }
    parallel::sum_in_place(tn); // scalar norm per tile, never amplitude payload
    // Full-space residual bound WITHOUT storing omitted amplitudes/residuals.
    // Each off-domain axis contribution is bounded by ||F_sub||F ||T_tile||F.
    std::map<Mp2TileKey,long double> outside_orbits;
    long double outside=0;
    for(const auto& [key,dst]:all) if(!active.count(key)) {
      double upper=dst.source_bound;
      for(std::size_t ax=0;ax<4;++ax) for(std::size_t k=0;k<(ax<2?og.size():vg.size());++k) {
        auto src=key;src[ax]=k;const auto it=active.find(src);if(it==active.end()) continue;
        long double fn=0;const auto& f=ax<2?fo:fv;
        for(auto i:dst.indices[ax]) for(auto j:result.tiles[it->second].indices[ax]) fn+=static_cast<long double>(f(i,j))*f(i,j);
        upper+=std::sqrt(static_cast<double>(fn))*tn[it->second];
      }
      upper*=1+1e-12;outside_orbits[orbit(key)]+=static_cast<long double>(upper)*upper;outside+=static_cast<long double>(upper)*upper;
    }
    result.residual_bound=std::hypot(mp2_global_norm(r),std::sqrt(static_cast<double>(outside)))+result.integral_error;
    result.energy_error=3*energy_source_bound/result.gap*result.residual_bound+3*result.integral_error*mp2_global_norm(t);
    if(!std::isfinite(result.energy_error)) throw std::runtime_error("adaptive MP2 error bound overflow");
    if(progress) {std::ostringstream s;s<<"adaptive MP2 round="<<result.rounds<<" selected="<<total<<"/"<<result.full_elements<<" full_residual_bound="<<result.residual_bound<<" energy_error_bound_Eh="<<result.energy_error;progress(s.str());}
    if(result.energy_error<=opt.accuracy*0.8) {
      result.final_active_residual=mp2_global_norm(r);result.strict_residual_target=strict_target;
      if(result.final_active_residual>strict_target) throw std::runtime_error("adaptive MP2 final residual failed unchanged strict target");
      break;
    }
    if(outside_orbits.empty()) throw std::runtime_error("AO/solve error exceeds target; no unselected domain remains");
    std::vector<std::pair<Mp2TileKey,long double>> sorted(outside_orbits.begin(),outside_orbits.end());
    std::stable_sort(sorted.begin(),sorted.end(),[](const auto& a,const auto& b){return a.second>b.second;});
    // Expand at least the largest orbit; target a 90% reduction of the current
    // outside bound. Recheck all coupling response after every re-solve.
    std::vector<long double> suffix(sorted.size()+1,0.L);
    for(std::size_t k=sorted.size();k-->0;) suffix[k]=suffix[k+1]+sorted[k].second;
    long double goal=outside*0.01L;
    if(opt.efficient_planning) {
      const long double absolute_goal=opt.accuracy*result.gap/(12*std::max(energy_source_bound,1.));
      goal=std::min(goal,absolute_goal*absolute_goal);
    }
    for(std::size_t k=0;k<sorted.size();++k) {
      wanted.insert(sorted[k].first);if(suffix[k+1]<=goal) break;
    }
    if(opt.efficient_planning) {
      std::size_t predicted=0;
      for(const auto& [key,tile]:all) if(wanted.count(orbit(key))) predicted+=elements(tile);
      if(static_cast<long double>(predicted)>=0.8L*result.full_elements) {
        for(const auto& [key,tile]:all) wanted.insert(orbit(key));
        result.full_domain_fallback=true;
      }
    }
  }
  std::array<long double,4> energies{};
  for(const auto& tile:result.tiles) {
    if(!owned(tile)) continue;
    auto exchange=tile.key;std::swap(exchange[2],exchange[3]);
    const auto& ex=result.tiles.at(active.at(exchange));
    const auto ni=tile.indices[0].size(),nj=tile.indices[1].size(),na=tile.indices[2].size(),nb=tile.indices[3].size();
    for(std::size_t i=0;i<ni;++i) for(std::size_t j=0;j<nj;++j) for(std::size_t a=0;a<na;++a) for(std::size_t b=0;b<nb;++b) {
      const auto k=((i*nj+j)*na+a)*nb+b,e=((i*nj+j)*nb+b)*na+a;
      energies.at(tile.span-1)+=(2*tile.integrals[k]-ex.integrals[e])*tile.amplitudes[k];
    }
  }
  std::vector<double> bins(energies.begin(),energies.end());parallel::sum_in_place(bins);
  for(std::size_t n=0;n<4;++n) {result.energy[n]=bins[n];if(!std::isfinite(result.energy[n])) throw std::runtime_error("nonfinite MP2 energy");}
  return result;
}
} // namespace localized_cc
