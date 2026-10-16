#include "localized_cc/lmp2_adaptive.hpp"
#include "localized_cc/parallel.hpp"
#include <libint2/basis.h>
#include <libint2/engine.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <functional>
#include <map>
#include <deque>
#include <sstream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace localized_cc {
namespace {
using Dense=modernqc::linalg::Matrix;
std::size_t checked_mul(std::size_t a,std::size_t b) {
  if(a&&b>std::numeric_limits<std::size_t>::max()/a) throw std::overflow_error("domain AO workspace overflow");return a*b;
}
std::size_t ao_checked_add(std::size_t a,std::size_t b) {
  if(b>std::numeric_limits<std::size_t>::max()-a) throw std::overflow_error("AO workspace sum overflow");
  return a+b;
}
std::vector<libint2::Atom> atoms(const modernqc::molecule::Molecule& m) {
  std::vector<libint2::Atom> a;for(const auto& x:m.atoms()) a.push_back({x.atomic_number,x.position_bohr[0],x.position_bohr[1],x.position_bohr[2]});return a;
}
double ao_error_norm(const std::vector<double>& x) {long double s=0;for(double v:x) s+=static_cast<long double>(v)*v;return std::sqrt(static_cast<double>(s));}
}
struct Mp2DomainAoProvider::Impl {
  libint2::BasisSet basis;
  Dense co,cv,q,so,sv;
  std::vector<std::size_t> first;
  std::size_t cap,threads,max_shell{};
  bool pair_exchange_reuse;
  std::size_t cache_bytes;
  struct QuartetCache {
    std::map<std::array<std::size_t,4>,std::vector<double>> entries;
    std::deque<std::array<std::size_t,4>> fifo;
    std::size_t bytes{},peak{};
  };
  std::vector<QuartetCache> caches; // thread-private, bounded; persists across batches
  Mp2AoStatistics stats;
  Impl(const modernqc::molecule::Molecule& m,const modernqc::integrals::Libint2IntegralProvider& p,
      const Dense& o,const Dense& v,std::size_t c,std::size_t nt,bool reuse,std::size_t cache)
    :basis(p.basis_metadata().canonical_name,atoms(m),true),co(o),cv(v),
     q(basis.size(),basis.size()),so(basis.size(),o.columns()),sv(basis.size(),v.columns()),cap(c),threads(nt),pair_exchange_reuse(reuse),cache_bytes(cache) {
    if(cache_bytes>cap/8) throw std::invalid_argument("AO quartet cache must not exceed one eighth of AO workspace");
    const auto& meta=p.basis_metadata();basis.set_pure(meta.spherical);
    if(!nt||o.rows()!=meta.number_of_aos||v.rows()!=o.rows()||!o.columns()||!v.columns()||
        basis.size()!=meta.shells.size()||static_cast<std::size_t>(basis.nbf())!=o.rows()) throw std::invalid_argument("domain AO basis/dimensions mismatch");
    modernqc::linalg::require_finite(o,"domain occupied");modernqc::linalg::require_finite(v,"domain virtuals");
    first=basis.shell2bf();
    for(std::size_t s=0;s<basis.size();++s) {
      const auto& x=basis[s];const auto& y=meta.shells[s];
      if(x.contr.size()!=1||first[s]!=y.first_ao||x.size()!=y.function_count||x.contr[0].l!=y.angular_momentum||
          x.contr[0].pure!=y.spherical||std::vector<double>(x.alpha.begin(),x.alpha.end())!=y.exponents||
          std::vector<double>(x.contr[0].coeff.begin(),x.contr[0].coeff.end())!=y.contraction_coefficients)
        throw std::runtime_error("domain AO basis ordering/normalization mismatch");
      max_shell=std::max(max_shell,x.size());
      for(std::size_t mu=first[s];mu<first[s]+x.size();++mu) {
        for(std::size_t i=0;i<o.columns();++i) so(s,i)+=std::abs(o(mu,i));
        for(std::size_t a=0;a<v.columns();++a) sv(s,a)+=std::abs(v(mu,a));
      }
    }
    libint2::Engine engine{libint2::Operator::coulomb,basis.max_nprim(),static_cast<int>(basis.max_l()),0,0.0};
    for(std::size_t s=0;s<basis.size();++s) for(std::size_t t=0;t<=s;++t) {
      engine.compute(basis[s],basis[t],basis[s],basis[t]);const auto* buf=engine.results()[0];
      if(!buf) throw std::runtime_error("domain AO self quartet null with precision zero");
      double d=0;const auto ns=basis[s].size(),ntt=basis[t].size();
      for(std::size_t mu=0;mu<ns;++mu) for(std::size_t nu=0;nu<ntt;++nu) {
        const double x=buf[((mu*ntt+nu)*ns+mu)*ntt+nu];if(!std::isfinite(x)) throw std::runtime_error("nonfinite Schwarz integral");d=std::max(d,std::abs(x));
      }
      q(s,t)=q(t,s)=std::sqrt(d)*(1+1e-12);
    }
  }
};
Mp2DomainAoProvider::Mp2DomainAoProvider(const modernqc::molecule::Molecule& m,
    const modernqc::integrals::Libint2IntegralProvider& p,const Dense& o,const Dense& v,std::size_t cap,std::size_t threads,bool reuse,std::size_t cache)
  :impl_(std::make_unique<Impl>(m,p,o,v,cap,threads,reuse,cache)) {}
Mp2DomainAoProvider::~Mp2DomainAoProvider()=default;
Matrix Mp2DomainAoProvider::pair_bounds() const {
  const auto& x=*impl_;
  const auto b=modernqc::linalg::multiply(modernqc::linalg::transpose(x.so),modernqc::linalg::multiply(x.q,x.sv));
  Matrix r({x.co.columns(),x.cv.columns()});for(std::size_t i=0;i<r.dimensions()[0];++i) for(std::size_t a=0;a<r.dimensions()[1];++a) r(i,a)=b(i,a)*(1+1e-12);return r;
}
Mp2AoStatistics Mp2DomainAoProvider::statistics() const {return impl_->stats;}

double Mp2DomainAoProvider::build(std::vector<Mp2AdaptiveTile>& global_tiles,double budget,
    const std::function<void(const std::string&)>& progress) {
  auto& x=*impl_;const auto world=parallel::world();
  if(global_tiles.empty()||!std::isfinite(budget)||budget<0) throw std::invalid_argument("invalid domain AO build request");
  // Output ownership, not a replicated numerical all-reduce. A physical
  // quartet can be evaluated on different ranks for different owned blocks.
  std::vector<Mp2AdaptiveTile> tiles;
  std::vector<std::size_t> global_ids;
  // Real four-center Coulomb integrals obey (ia|jb)=(jb|ia).
  // Neither i<->j alone nor a<->b alone is this symmetry. Retain every
  // requested block, but contract one representative of each present pair.
  std::map<Mp2TileKey,std::size_t> by_key;
  std::map<std::pair<bool,std::size_t>,std::vector<std::size_t>> all_indices;
  for(std::size_t b=0;b<global_tiles.size();++b) {
    if(!global_tiles[b].integrals.empty()||!global_tiles[b].amplitudes.empty())
      throw std::invalid_argument("AO builder requires new empty tiles");
    if(!by_key.emplace(global_tiles[b].key,b).second) throw std::invalid_argument("duplicate AO target key");
    for(std::size_t axis=0;axis<4;++axis) {
      const auto& ids=global_tiles[b].indices[axis];
      if(ids.empty()) throw std::invalid_argument("empty AO target tile");
      for(auto p:ids) if(p>=(axis<2?x.co.columns():x.cv.columns())) throw std::invalid_argument("AO target index out of range");
      const auto [it,inserted]=all_indices.emplace(std::make_pair(axis<2,global_tiles[b].key[axis]),ids);
      if(!inserted&&it->second!=ids) throw std::invalid_argument("AO orbital tile key has inconsistent indices");
    }
  }
  std::vector<std::size_t> representative(global_tiles.size()),multiplicity(global_tiles.size(),0);
  std::size_t output_elements=0;
  for(std::size_t b=0;b<global_tiles.size();++b) {
    const auto& key=global_tiles[b].key;
    const Mp2TileKey exchanged{key[1],key[0],key[3],key[2]};
    const auto mate=by_key.find(exchanged);
    representative[b]=x.pair_exchange_reuse&&exchanged<key&&mate!=by_key.end()?mate->second:b;
    ++multiplicity[representative[b]];
    if(mp2_tile_owner(key,world.size)==world.rank) {
      std::size_t n=1;for(const auto& ids:global_tiles[b].indices) n=checked_mul(n,ids.size());
      output_elements=ao_checked_add(output_elements,n);
    }
  }
  for(std::size_t b=0;b<global_tiles.size();++b) {
    if(representative[b]==b&&mp2_tile_owner(global_tiles[b].key,world.size)==world.rank) {
      tiles.push_back(global_tiles[b]);global_ids.push_back(b);
    }
  }
  std::vector<std::size_t> offsets;std::size_t count=0,max_pair=0;
  for(const auto& b:tiles) {
    offsets.push_back(count);std::size_t n=1;for(std::size_t ax=0;ax<4;++ax) {
      if(b.indices[ax].empty()) throw std::invalid_argument("empty AO target tile");
      for(auto p:b.indices[ax]) if(p>=(ax<2?x.co.columns():x.cv.columns())) throw std::invalid_argument("AO target index out of range");
      n=checked_mul(n,b.indices[ax].size());
    }
    if(n>std::numeric_limits<std::size_t>::max()-count) throw std::overflow_error("AO payload overflow");count+=n;
    max_pair=std::max({max_pair,checked_mul(b.indices[0].size(),b.indices[2].size()),checked_mul(b.indices[1].size(),b.indices[3].size())});
  }
  std::map<std::pair<std::size_t,std::size_t>,std::size_t> left_pairs,right_pairs;
  std::map<std::pair<bool,std::size_t>,std::vector<std::size_t>> orbital_tiles;
  std::vector<std::size_t> left_slot,right_slot;
  for(const auto& tile:tiles) {
    for(std::size_t axis=0;axis<4;++axis) {
      const auto [it,inserted]=orbital_tiles.emplace(std::make_pair(axis<2,tile.key[axis]),tile.indices[axis]);
      if(!inserted&&it->second!=tile.indices[axis]) throw std::invalid_argument("AO orbital tile key has inconsistent indices");
    }
    const auto l=std::make_pair(tile.key[0],tile.key[2]),r=std::make_pair(tile.key[1],tile.key[3]);
    left_pairs.emplace(l,left_pairs.size());right_pairs.emplace(r,right_pairs.size());
    left_slot.push_back(left_pairs.at(l));right_slot.push_back(right_pairs.at(r));
  }
  const auto sp=checked_mul(x.max_shell,x.max_shell);
  const auto transient_work=checked_mul(64,checked_mul(std::max(sp,max_pair),std::max(sp,max_pair)));
  const auto cached_work=checked_mul(checked_mul(left_pairs.size()+right_pairs.size(),max_pair),checked_mul(sp,sizeof(double)));
  if(cached_work>std::numeric_limits<std::size_t>::max()-transient_work) throw std::overflow_error("AO pair cache overflow");
  const auto matrix_work=transient_work+cached_work;
  const auto payload=checked_mul(count,sizeof(double));
  const auto support_bytes=checked_mul(checked_mul(tiles.size(),x.basis.size()),4*sizeof(double));
  const auto ns=x.basis.size();
  const auto shell_pairs=checked_mul(ns,ns);
  std::size_t leaves=1;
  while(leaves<shell_pairs) leaves=checked_mul(leaves,2);
  const auto tree_count=left_pairs.size()+right_pairs.size();
  const auto tree_bytes=checked_mul(checked_mul(tree_count,checked_mul(2,leaves)),sizeof(long double));
  // Conservatively reserve metadata/recursive lists before allocating workers.
  const auto hierarchy_work=checked_mul(tiles.size(),4096);
  auto within=[](std::size_t a,std::size_t b,std::size_t cap) {return a<=cap&&b<=cap-a;};
  std::size_t base=0;
  bool fits=true;
  // Reserve full owner output as well as representative worker reductions.
  // Reuse lowers construction work, not the selected solver's payload count.
  for(auto bytes:{payload,support_bytes,tree_bytes,checked_mul(output_elements,sizeof(double)),x.cache_bytes}) {
    if(!within(base,bytes,x.cap)) {fits=false;break;}base+=bytes;
  }
  const auto per_worker=std::max(std::size_t{1},ao_checked_add(ao_checked_add(payload,matrix_work),hierarchy_work));
  std::vector<double> memory_failed{double(!fits||per_worker>x.cap-base)};
  parallel::sum_in_place(memory_failed);
  if(memory_failed[0]) throw std::runtime_error("domain AO owner batch/hierarchy exceeds a rank cap");
  std::size_t workers=std::max(std::size_t{1},std::min(x.threads,(x.cap-base)/per_worker));
#ifndef _OPENMP
  workers=1;
#endif
  if(workers>static_cast<std::size_t>(std::numeric_limits<int>::max())) throw std::overflow_error("AO thread count overflow");
  if(world.rank==0&&progress) progress("domain AO worker plan; maximum_threads_per_rank="+std::to_string(workers)+"; owned_batch_elements="+std::to_string(count));
  std::vector<std::vector<double>> values(workers,std::vector<double>(count)),errors(workers,std::vector<double>(tiles.size()));
  std::vector<std::string> failures(workers);
  std::vector<std::array<double,9>> counters(workers);
  const auto cache_limit=x.cache_bytes/x.threads;
  const bool use_cache=cache_limit>=2048;
  if(use_cache) x.caches.resize(workers);
  const long double quartets=static_cast<long double>(ns)*ns*ns*ns;
  const long double per_quartet=static_cast<long double>(budget)/std::sqrt(static_cast<long double>(global_tiles.size()))/quartets;
  // Full PM tails enter every bound. No coefficient is truncated.
  std::vector<std::array<std::vector<double>,4>> support(tiles.size());
  for(std::size_t b=0;b<tiles.size();++b) for(std::size_t ax=0;ax<4;++ax) {
    support[b][ax].resize(ns);
    for(std::size_t s=0;s<ns;++s) {long double sum=0;
      for(auto p:tiles[b].indices[ax]) {const auto v=ax<2?x.so(s,p):x.sv(s,p);sum+=static_cast<long double>(v)*v;}
      support[b][ax][s]=std::sqrt(static_cast<double>(sum))*(1+1e-12);
    }
  }
  // A sum tree bounds a whole rectangle of ordered shell-pair products.
  // Leaves retain ALL PM coefficient tails. Internal positive sums avoid
  // cancellation from prefix sums. No nshell^4 list is allocated or traversed.
  using Tree=std::vector<long double>;
  std::vector<Tree> left_tree(left_pairs.size()),right_tree(right_pairs.size());
  for(std::size_t b=0;b<tiles.size();++b) {
    for(bool left:{true,false}) {
      auto& tree=left?left_tree[left_slot[b]]:right_tree[right_slot[b]];
      if(!tree.empty()) continue;
      tree.assign(2*leaves,0.L);
      const auto ax=left?0U:1U;
      for(std::size_t pair=0;pair<shell_pairs;++pair)
        tree[leaves+pair]=static_cast<long double>(x.q(pair/ns,pair%ns))*
          support[b][ax][pair/ns]*support[b][ax+2][pair%ns]*(1+1e-12L);
      for(std::size_t k=leaves;k-->1;) tree[k]=(tree[2*k]+tree[2*k+1])*(1+1e-12L);
      if(!std::isfinite(tree[1])) throw std::runtime_error("AO tree bound overflow");
    }
  }
  struct Node {std::size_t node,begin,width;};
  std::vector<Node> frontier{{1,0,leaves}};
  while(frontier.size()<workers*4&&frontier.front().width>1) {
    std::vector<Node> next;
    for(const auto& n:frontier) {
      next.push_back({2*n.node,n.begin,n.width/2});
      next.push_back({2*n.node+1,n.begin+n.width/2,n.width/2});
    }
    frontier=std::move(next);
  }
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(workers))
#endif
  {
    std::size_t worker=0,team=1;
#ifdef _OPENMP
    worker=static_cast<std::size_t>(omp_get_thread_num());
    team=static_cast<std::size_t>(omp_get_num_threads());
#endif
    // Explicit thread-local partition avoids throwing across an OpenMP loop.
    try {
      libint2::Engine engine{libint2::Operator::coulomb,x.basis.max_nprim(),static_cast<int>(x.basis.max_l()),0,0.0};
      const auto started=std::chrono::steady_clock::now();auto last=started;
      // Static disjoint left-tree frontier; all MPI is outside this region.
      std::function<void(Node,Node,const std::vector<std::size_t>&)> visit;
      visit=[&](Node lnode,Node rnode,const std::vector<std::size_t>& candidates) {
        if(lnode.begin>=shell_pairs||rnode.begin>=shell_pairs||candidates.empty()) return;
        ++counters[worker][4];
        const auto nl=std::min(lnode.width,shell_pairs-lnode.begin);
        const auto nr=std::min(rnode.width,shell_pairs-rnode.begin);
        const long double area=static_cast<long double>(nl)*nr;
        const long double allowance=per_quartet*area;
        std::vector<std::size_t> retained;retained.reserve(candidates.size());
        for(auto b:candidates) {
          const long double bound=left_tree[left_slot[b]][lnode.node]*right_tree[right_slot[b]][rnode.node];
          if(!std::isfinite(bound)) throw std::runtime_error("AO rectangle bound overflow");
          if(bound<=allowance) {
            errors[worker][b]+=static_cast<double>(bound);
            counters[worker][3]+=static_cast<double>(area);
          } else retained.push_back(b);
        }
        if(retained.empty()) {
          counters[worker][2]+=static_cast<double>(area);++counters[worker][6];return;
        }
        if(lnode.width>1||rnode.width>1) {
          if(lnode.width>=rnode.width) {
            visit({2*lnode.node,lnode.begin,lnode.width/2},rnode,retained);
            visit({2*lnode.node+1,lnode.begin+lnode.width/2,lnode.width/2},rnode,retained);
          } else {
            visit(lnode,{2*rnode.node,rnode.begin,rnode.width/2},retained);
            visit(lnode,{2*rnode.node+1,rnode.begin+rnode.width/2,rnode.width/2},retained);
          }
          return;
        }
        ++counters[worker][5];
        std::vector<unsigned char> needed(tiles.size(),0);
        for(auto b:retained) needed[b]=1;
        const auto s=lnode.begin/ns,t=lnode.begin%ns,u=rnode.begin/ns,v=rnode.begin%ns;
          ++counters[worker][1];
          const std::array<std::size_t,4> shells{s,t,u,v};
          std::array<std::size_t,4> order{0,1,2,3};
          if(use_cache) {
            if(shells[order[0]]<shells[order[1]]) std::swap(order[0],order[1]);
            if(shells[order[2]]<shells[order[3]]) std::swap(order[2],order[3]);
            if(std::make_pair(shells[order[0]],shells[order[1]])<std::make_pair(shells[order[2]],shells[order[3]])) {
              std::swap(order[0],order[2]);std::swap(order[1],order[3]);
            }
          }
          const std::array<std::size_t,4> key{shells[order[0]],shells[order[1]],shells[order[2]],shells[order[3]]};
          const double* buf=nullptr;
          if(use_cache) {
            const auto found=x.caches[worker].entries.find(key);
            if(found!=x.caches[worker].entries.end()) {buf=found->second.data();++counters[worker][8];}
          }
          if(!buf) {
            ++counters[worker][7];engine.compute(x.basis[key[0]],x.basis[key[1]],x.basis[key[2]],x.basis[key[3]]);
            buf=engine.results()[0];if(!buf) throw std::runtime_error("unscreened requested AO quartet is null");
            if(use_cache) {
              auto& cache=x.caches[worker];
              std::size_t size=1;for(auto shell:key) size=checked_mul(size,x.basis[shell].size());
              const auto bytes=ao_checked_add(checked_mul(size,sizeof(double)),512);
              // 1024-byte container allowance and at most 64 shell quartets
              // per worker, independent of molecule size. Oversize entries
              // bypass the cache. No full AO four-index tensor is retained.
              if(bytes<=cache_limit-1024) {
                while(!cache.fifo.empty()&&(cache.entries.size()>=64||cache.bytes>cache_limit-1024-bytes)) {
                  const auto old=cache.fifo.front();cache.fifo.pop_front();
                  cache.bytes-=cache.entries.at(old).size()*sizeof(double)+512;cache.entries.erase(old);
                }
                auto [it,inserted]=cache.entries.emplace(key,std::vector<double>(buf,buf+size));
                if(!inserted) throw std::runtime_error("duplicate AO cache insertion");
                cache.fifo.push_back(key);cache.bytes+=bytes;
                cache.peak=std::max(cache.peak,cache.bytes+1024);buf=it->second.data();
              }
            }
          }
          const auto n0=x.basis[s].size(),n1=x.basis[t].size(),n2=x.basis[u].size(),n3=x.basis[v].size();
          Dense eri(n0*n1,n2*n3);
          if(!use_cache) {
            for(std::size_t l=0;l<n0*n1;++l) for(std::size_t r=0;r<n2*n3;++r) eri(l,r)=buf[l*n2*n3+r];
          } else {
            for(std::size_t a0=0;a0<n0;++a0) for(std::size_t a1=0;a1<n1;++a1)
              for(std::size_t a2=0;a2<n2;++a2) for(std::size_t a3=0;a3<n3;++a3) {
                const std::array<std::size_t,4> index{a0,a1,a2,a3};
                const auto k=((index[order[0]]*x.basis[key[1]].size()+index[order[1]])*
                    x.basis[key[2]].size()+index[order[2]])*x.basis[key[3]].size()+index[order[3]];
                eri(a0*n1+a1,a2*n3+a3)=buf[k];
              }
          }
          // Reuse (I,A) half transforms and (J,B) coefficient panels across
          // all output tiles of this quartet; no duplicate left DGEMMs.
          std::vector<Dense> left_cache(left_pairs.size()),right_cache(right_pairs.size());
          for(std::size_t b=0;b<tiles.size();++b) {
            if(!needed[b]) continue;
            const auto& ids=tiles[b].indices;const auto ni=ids[0].size(),nj=ids[1].size(),na=ids[2].size(),nb=ids[3].size();
            auto& half=left_cache[left_slot[b]];auto& right=right_cache[right_slot[b]];
            if(half.empty()) {
              Dense left(ni*na,n0*n1);
              for(std::size_t i=0;i<ni;++i) for(std::size_t a=0;a<na;++a) for(std::size_t mu=0;mu<n0;++mu) for(std::size_t nu=0;nu<n1;++nu)
                left(i*na+a,mu*n1+nu)=x.co(x.first[s]+mu,ids[0][i])*x.cv(x.first[t]+nu,ids[2][a]);
              half=modernqc::linalg::multiply(left,eri);
            }
            if(right.empty()) {
              right=Dense(n2*n3,nj*nb);
              for(std::size_t j=0;j<nj;++j) for(std::size_t a=0;a<nb;++a) for(std::size_t mu=0;mu<n2;++mu) for(std::size_t nu=0;nu<n3;++nu)
                right(mu*n3+nu,j*nb+a)=x.co(x.first[u]+mu,ids[1][j])*x.cv(x.first[v]+nu,ids[3][a]);
            }
            const auto result=modernqc::linalg::multiply(half,right);
            for(std::size_t i=0;i<ni;++i) for(std::size_t j=0;j<nj;++j) for(std::size_t a=0;a<na;++a) for(std::size_t c=0;c<nb;++c)
              values[worker][offsets[b]+((i*nj+j)*na+a)*nb+c]+=result(i*na+a,j*nb+c);
          }
        if(world.rank==0&&worker==0&&progress) {
          const auto now=std::chrono::steady_clock::now();
          if(std::chrono::duration<double>(now-last).count()>=30.) {
            std::ostringstream message;message<<"domain AO owner-zero hierarchy nodes="<<counters[worker][4]
              <<"; quartet leaves="<<counters[worker][5]<<"; seconds="<<std::chrono::duration<double>(now-started).count();
            progress(message.str());last=now;
          }
        }
      };
      std::vector<std::size_t> candidates(tiles.size());std::iota(candidates.begin(),candidates.end(),0);
      for(std::size_t task=worker;task<frontier.size();task+=team) {
        const auto& n=frontier[task];
        if(n.begin>=shell_pairs||tiles.empty()) continue;
        counters[worker][0]+=static_cast<double>(std::min(n.width,shell_pairs-n.begin))*shell_pairs;
        visit(n,{1,0,leaves},candidates);
      }
    } catch(const std::exception& e) {failures[worker]=e.what();}
  }
  std::vector<double> failed{0};for(const auto& e:failures) if(!e.empty()) failed[0]=1;
  parallel::sum_in_place(failed);
  if(failed[0]) {
    for(const auto& message:failures) if(!message.empty()) throw std::runtime_error("domain AO worker: "+message);
    throw std::runtime_error("domain AO worker failed on another rank; no integral batch accepted");
  }
  for(std::size_t worker=1;worker<workers;++worker) {
    for(std::size_t k=0;k<count;++k) values[0][k]+=values[worker][k];
    for(std::size_t k=0;k<tiles.size();++k) errors[0][k]+=errors[worker][k];
    for(std::size_t k=0;k<9;++k) counters[0][k]+=counters[worker][k];
  }
  // Only scalar statistics and the squared error norm cross ranks.
  // Each integral tile remains exclusively on its exchange-orbit owner.
  std::vector<double> c(counters[0].begin(),counters[0].end());parallel::sum_in_place(c);
  x.stats.quartets_considered+=c[0];x.stats.quartets_evaluated+=c[1];x.stats.quartets_skipped+=c[2];x.stats.tile_quartets_skipped+=c[3];
  x.stats.tree_nodes_visited+=c[4];x.stats.quartet_leaves_visited+=c[5];x.stats.pruned_subtrees+=c[6];
  x.stats.engine_evaluations+=c[7];x.stats.quartet_cache_hits+=c[8];
  std::vector<double> peaks(world.size,0.);
  for(const auto& cache:x.caches) peaks[world.rank]+=cache.peak;
  parallel::sum_in_place(peaks);
  x.stats.quartet_cache_peak_bytes=std::max(x.stats.quartet_cache_peak_bytes,*std::max_element(peaks.begin(),peaks.end()));
  for(std::size_t b=0;b<tiles.size();++b) {
    const auto end=b+1<tiles.size()?offsets[b+1]:count;
    global_tiles[global_ids[b]].integrals.assign(values[0].begin()+offsets[b],values[0].begin()+end);
    for(double value:global_tiles[global_ids[b]].integrals) if(!std::isfinite(value)) throw std::runtime_error("nonfinite selected AO integral");
    // The same approximation error appears in BOTH requested blocks; the
    // full source Frobenius bound must count both, not only representatives.
    errors[0][b]*=std::sqrt(static_cast<double>(multiplicity[global_ids[b]]));
  }
  for(std::size_t b=0;b<global_tiles.size();++b) {
    if(representative[b]==b||mp2_tile_owner(global_tiles[b].key,world.size)!=world.rank) continue;
    auto& target=global_tiles[b];const auto& source=global_tiles[representative[b]];
    const auto ni=target.indices[0].size(),nj=target.indices[1].size();
    const auto na=target.indices[2].size(),nb=target.indices[3].size();
    target.integrals.resize(checked_mul(checked_mul(ni,nj),checked_mul(na,nb)));
    if(source.integrals.size()!=target.integrals.size()) throw std::runtime_error("AO pair-exchange owner/extent mismatch");
    for(std::size_t i=0;i<ni;++i) for(std::size_t j=0;j<nj;++j)
      for(std::size_t a=0;a<na;++a) for(std::size_t bvirt=0;bvirt<nb;++bvirt)
        target.integrals[((i*nj+j)*na+a)*nb+bvirt]=source.integrals[((j*ni+i)*nb+bvirt)*na+a];
  }
  // Metadata is identical on all ranks; these counters already represent
  // global requested/contracted tiles and must not be summed over ranks.
  const auto contracted=std::count_if(multiplicity.begin(),multiplicity.end(),[](auto n){return n!=0;});
  x.stats.requested_tiles+=global_tiles.size();x.stats.contracted_tiles+=contracted;
  x.stats.pair_exchange_reused_tiles+=global_tiles.size()-contracted;
  const double local_error=ao_error_norm(errors[0]);
  std::vector<double> error_square{local_error*local_error};parallel::sum_in_place(error_square);
  return std::sqrt(error_square[0])*(1+1e-12);
}
} // namespace localized_cc
