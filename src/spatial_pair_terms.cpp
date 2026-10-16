#include "spatial_pair_terms.hpp"
#include "localized_cc/parallel.hpp"
#include "localized_cc/pm_symmetry.hpp"
#include "pm_orbit_plan.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace localized_cc {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now()-start).count();
}
std::size_t mul(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max()/b)
    throw std::overflow_error("spatial pair workspace size overflow");
  return a*b;
}
int bi(std::size_t n) {
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("spatial pair BLAS dimension overflow");
  return static_cast<int>(n);
}
std::size_t plus(std::size_t a,std::size_t b) {
  if (a>std::numeric_limits<std::size_t>::max()-b)
    throw std::overflow_error("spatial pair workspace sum overflow");
  return a+b;
}
extern "C" void dgemm_(const char*, const char*, const int*, const int*,
    const int*, const double*, const double*, const int*, const double*,
    const int*, const double*, double*, const int*);
// Row-major C=A B or A B^T, single-thread BLAS inside owned OpenMP tasks.
void gemm(const double* a, const double* b, double* c, int m, int n, int k,
          bool transpose_b=false, double alpha=1.0, double beta=0.0) {
  const char trans=transpose_b?'T':'N', normal='N';
  const int ldb=transpose_b?k:n;
  dgemm_(&trans,&normal,&n,&m,&k,&alpha,b,&ldb,a,&k,&beta,c,&n);
}
template<class F> void owned(std::size_t count, const F& f) {
  const auto w=parallel::world();
  const auto local=w.rank<count ? 1+(count-1-w.rank)/w.size : 0;
  if (local > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error("spatial pair task count overflow");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::int64_t k=0;k<static_cast<std::int64_t>(local);++k)
    f(w.rank+static_cast<std::size_t>(k)*w.size);
}
using PairPlan=detail::PmOrbitPlan<2>;
void scatter_row(Matrix& output,std::size_t row,const double* values,
                 const PairPlan& row_plan,const PairPlan& column_plan) {
  row_plan.images(row,[&](std::size_t g,std::size_t target,int sign) {
    for (std::size_t c=0;c<column_plan.size();++c) {
      const auto [destination,phase]=column_plan.map(g,c);
      output(target,destination)=sign*phase*values[c];
    }
  });
}
Matrix product(const Matrix& a, const Matrix& b, bool trans=false,
               const PairPlan* row_plan=nullptr,const PairPlan* column_plan=nullptr) {
  const auto m=a.dimensions()[0], k=a.dimensions()[1];
  const auto n=b.dimensions()[trans?0:1];
  if (b.dimensions()[trans?1:0]!=k)
    throw std::invalid_argument("spatial pair matrix product mismatch");
  const int ni=bi(n), ki=bi(k);
  Matrix c({m,n});
  constexpr std::size_t rows=16;
  if (row_plan && row_plan->reduced()) {
    if (!column_plan || row_plan->size()!=m || column_plan->size()!=n)
      throw std::invalid_argument("PM pair product dimensions mismatch");
    owned((row_plan->tasks()+rows-1)/rows,[&](std::size_t tile) {
      const auto first=tile*rows,count=std::min(rows,row_plan->tasks()-first);
      Matrix packed({count,k}),out({count,n});
      for (std::size_t r=0;r<count;++r)
        std::copy_n(a.data().data()+row_plan->representative(first+r)*k,k,packed.data().data()+r*k);
      gemm(packed.data().data(),b.data().data(),out.data().data(),static_cast<int>(count),ni,ki,trans);
      for (std::size_t r=0;r<count;++r)
        scatter_row(c,row_plan->representative(first+r),out.data().data()+r*n,*row_plan,*column_plan);
    });
  } else owned((m+rows-1)/rows,[&](std::size_t tile) {
    const auto first=tile*rows, count=std::min(rows,m-first);
    gemm(a.data().data()+first*k,b.data().data(),c.data().data()+first*n,
         static_cast<int>(count),ni,ki,trans);
  });
  parallel::sum_in_place(c.data());
  return c;
}
void add(Matrix& a, const Matrix& b, double factor=1.0) {
  for (std::size_t i=0;i<a.size();++i) a.data()[i]+=factor*b.data()[i];
}
}

Tensor4 spatial_tiled_pair_terms(const SpatialRhfHamiltonian& h,
    const SpatialRccsdAmplitudes& amplitudes, const CcsdOptions& options,
    CcsdIterationTiming& timing) {
  const auto o=h.nocc, v=h.nvirt, oo=mul(o,o), vv=mul(v,v), ov=mul(o,v);
  const int ooi=bi(oo), vvi=bi(vv);
  // Conservative bound on simultaneously live replicated pair matrices.
  // Does not include the Hamiltonian, amplitudes, solver/DIIS or small F blocks.
  const auto pair_elements=mul(ov,ov), occ_elements=mul(oo,oo);
  if (pair_elements > (std::numeric_limits<std::size_t>::max()-occ_elements)/16)
    throw std::overflow_error("spatial pair workspace size overflow");
  auto fixed_bytes=mul(16*pair_elements+occ_elements,sizeof(double));
  if (options.use_pm_residual_orbits) {
    std::size_t threads=1;
#if defined(_OPENMP)
    threads=static_cast<std::size_t>(omp_get_max_threads());
#endif
    // Extra corrected Woooo plus gathered BLAS input/output panels per
    // worker. Representative vectors may have capacity up to twice size.
    const auto scratch=mul(mul(threads,32),std::max({oo,ov,vv}));
    fixed_bytes=plus(fixed_bytes,mul(plus(occ_elements,scratch),sizeof(double)));
    fixed_bytes=plus(fixed_bytes,mul(plus(plus(oo,ov),vv),2*sizeof(std::size_t)));
    // Caller retains its scalar R2 orbit plan while adding pair terms.
    fixed_bytes=plus(fixed_bytes,mul(pair_elements,2*sizeof(std::size_t)));
  }
  if (fixed_bytes>options.wabef_invariant_cache_max_bytes)
    throw std::runtime_error("native spatial pair workspace exceeds CCSD_WABEF_INVARIANT_CACHE_MIB");
  const auto& t1=amplitudes.t1;
  const auto& t2=amplitudes.t2;
  const auto& eri=h.eri_chemist;
  if (options.use_pm_residual_orbits && !options.pm_symmetry)
    throw std::invalid_argument("PM residual orbits require a verified group");
  const auto* group=options.use_pm_residual_orbits?options.pm_symmetry.get():nullptr;
  const PairPlan oo_plan(group,{o,o},{0,0});
  const PairPlan ov_plan(group,{o,v},{0,o});
  const PairPlan vv_plan(group,{v,v},{o,o});
  if (options.pm_symmetry) {
    validate_pm_symmetry_group(*options.pm_symmetry);
    if(options.pm_symmetry->occupied!=o || options.pm_symmetry->virtuals!=v)
      throw std::invalid_argument("native spatial PM group dimension mismatch");
    // Check the spin executor's covariance conditions without spin expansion.
    for (const auto& g:options.pm_symmetry->operations) {
      for (std::size_t i=0;i<o;++i) for (std::size_t a=0;a<v;++a) {
        const auto gi=g.image[i], ga=g.image[o+a]-o;
        const double x=t1(i,a), y=g.sign[i]*g.sign[o+a]*t1(gi,ga);
        if (std::abs(x-y)>1e-10*std::max({1.0,std::abs(x),std::abs(y)}))
          throw std::runtime_error("native spatial T1 violates PM covariance");
      }
      for (std::size_t i=0;i<o;++i) for (std::size_t j=0;j<o;++j)
        for (std::size_t a=0;a<v;++a) for (std::size_t b=0;b<v;++b) {
          const double x=t2(i,j,a,b);
          const double y=g.sign[i]*g.sign[j]*g.sign[o+a]*g.sign[o+b]*
              t2(g.image[i],g.image[j],g.image[o+a]-o,g.image[o+b]-o);
          if (std::abs(x-y)>1e-10*std::max({1.0,std::abs(x),std::abs(y)}))
            throw std::runtime_error("native spatial T2 violates PM covariance");
        }
    }
  }
  auto begin=Clock::now();
  Matrix tau({oo,vv}), integrals({oo,vv});
  owned(oo,[&](std::size_t ij) {
    const auto i=ij/o,j=ij%o;
    for (std::size_t a=0;a<v;++a) for (std::size_t b=0;b<v;++b) {
      tau(ij,a*v+b)=t2(i,j,a,b)+t1(i,a)*t1(j,b);
      integrals(ij,a*v+b)=eri(i,o+a,j,o+b);
    }
  });
  parallel::sum_in_place(tau.data());
  parallel::sum_in_place(integrals.data());
  Matrix woooo=product(tau,integrals,true,&oo_plan,&oo_plan); // rows ij, columns mn
  if (group) {
    Matrix corrected({oo,oo});
    owned(oo_plan.tasks(),[&](std::size_t task) {
      const auto ij=oo_plan.representative(task),i=ij/o,j=ij%o;
      std::vector<double> row(oo);
      for (std::size_t m=0;m<o;++m) for (std::size_t n=0;n<o;++n) {
        double x=eri(m,i,n,j)+0.5*woooo(ij,m*o+n);
        for (std::size_t e=0;e<v;++e)
          x+=t1(j,e)*eri(m,i,n,o+e)+t1(i,e)*eri(m,o+e,n,j);
        row[m*o+n]=x;
      }
      scatter_row(corrected,ij,row.data(),oo_plan,oo_plan);
    });
    parallel::sum_in_place(corrected.data());
    woooo=std::move(corrected);
  } else {
  owned(oo,[&](std::size_t ij) {
    const auto i=ij/o,j=ij%o;
    for (std::size_t m=0;m<o;++m) for (std::size_t n=0;n<o;++n) {
      double x=eri(m,i,n,j)+0.5*woooo(ij,m*o+n);
      for (std::size_t e=0;e<v;++e)
        x+=t1(j,e)*eri(m,i,n,o+e)+t1(i,e)*eri(m,o+e,n,j);
      woooo(ij,m*o+n)=x;
    }
  });
  // product() was replicated: zero non-owned rows before the collective.
  const auto world=parallel::world();
  for (std::size_t ij=0;ij<oo;++ij) if (ij%world.size!=world.rank)
    std::fill_n(woooo.data().data()+ij*oo,oo,0.0);
  parallel::sum_in_place(woooo.data());
  }
  Matrix pair=product(woooo,tau,false,&oo_plan,&vv_plan);

  // Two SU(2) channels of W_mbej, A*delta_me*delta_bj and
  // B*delta_mj*delta_be. Evaluate the spin sums once per spatial element;
  // spin labels here are selection rules, not expanded tensor storage.
  Matrix wa({ov,ov}), wb({ov,ov}), a0({ov,ov}), b0({ov,ov});
  const SpinOrbitalEriView se(h);
  const auto s1=[&](std::size_t i,std::size_t a) {
    return i%2==a%2?t1(i/2,a/2):0.0;
  };
  const auto s2=[&](std::size_t i,std::size_t j,std::size_t a,std::size_t b) {
    double x=0;
    if(i%2==a%2 && j%2==b%2) x+=t2(i/2,j/2,a/2,b/2);
    if(i%2==b%2 && j%2==a%2) x-=t2(i/2,j/2,b/2,a/2);
    return x;
  };
  const auto w=[&](std::size_t m,std::size_t b,std::size_t e,std::size_t j) {
    double x=se(m,2*o+b,2*o+e,j);
    for(std::size_t f=0;f<2*v;++f)
      x+=s1(j,f)*se(m,2*o+b,2*o+e,2*o+f);
    for(std::size_t n=0;n<2*o;++n)
      x-=s1(n,b)*se(m,n,2*o+e,j);
    for(std::size_t n=0;n<2*o;++n) for(std::size_t f=0;f<2*v;++f)
      x-=(0.5*s2(j,n,f,b)+s1(j,f)*s1(n,b))*se(m,n,2*o+e,2*o+f);
    return x;
  };
  if (options.use_compact_spatial_doubles) {
    if (options.use_pm_residual_orbits)
      throw std::invalid_argument("compact mixed pairs does not use PM residual orbits");
    // Exact spin-summed channels, in chemists' ERI notation:
    // D_me,nf=(me|nf), X_me,nf=(mf|ne).
    // A_nf,jb=t_jn^bf - t_jn^fb/2 - t_j^f t_n^b,
    // B_nf,jb=-t_jn^bf/2, C_nf,jb=t_jn^fb/2 + t_j^f t_n^b.
    // Nonlinear Wa=D*A+X*B; nonlinear Wb=X*C. These are dense spatial
    // matrices, not density-fitting factors or screened approximations.
    // Release this scratch before the subsequent td/tx/ts/singles panels;
    // the existing conservative 16*ov^2 workspace bound still covers it.
    Matrix direct({ov,ov}), exchange({ov,ov}), coefficient({ov,ov});
    owned(ov,[&](std::size_t me) {
      const auto m=me/v,e=me%v;
      for(std::size_t n=0;n<o;++n) for(std::size_t f=0;f<v;++f) {
        direct(me,n*v+f)=eri(m,o+e,n,o+f);
        exchange(me,n*v+f)=eri(m,o+f,n,o+e);
      }
      for(std::size_t j=0;j<o;++j) for(std::size_t b=0;b<v;++b) {
        a0(me,j*v+b)=eri(m,o+e,o+b,j);
        b0(me,j*v+b)=-eri(m,j,o+b,o+e);
        double a=a0(me,j*v+b),bb=b0(me,j*v+b);
        for(std::size_t f=0;f<v;++f) {
          a+=t1(j,f)*eri(m,o+e,o+b,o+f);
          bb-=t1(j,f)*eri(m,o+f,o+b,o+e);
        }
        for(std::size_t n=0;n<o;++n) {
          a-=t1(n,b)*eri(m,o+e,n,j);
          bb+=t1(n,b)*eri(m,j,n,o+e);
        }
        wa(me,j*v+b)=a; wb(me,j*v+b)=bb;
      }
    });
    parallel::sum_in_place(direct.data());
    parallel::sum_in_place(exchange.data());
    parallel::sum_in_place(wa.data()); parallel::sum_in_place(wb.data());
    parallel::sum_in_place(a0.data()); parallel::sum_in_place(b0.data());
    for(int channel=0;channel<3;++channel) {
      // All rows have one MPI owner; overwrite owned rows and zero others
      // on each pass before the collective (no stale replicated values).
      std::fill(coefficient.data().begin(),coefficient.data().end(),0.0);
      owned(ov,[&](std::size_t nf) {
        const auto n=nf/v,f=nf%v;
        for(std::size_t j=0;j<o;++j) for(std::size_t b=0;b<v;++b) {
          const double bf=t2(j,n,b,f),fb=t2(j,n,f,b),s=t1(j,f)*t1(n,b);
          coefficient(nf,j*v+b)=channel==0 ? bf-0.5*fb-s :
              (channel==1 ? -0.5*bf : 0.5*fb+s);
        }
      });
      parallel::sum_in_place(coefficient.data());
      add(channel==2 ? wb : wa,product(channel==0 ? direct : exchange,coefficient));
    }
    timing.intermediate_blas_contractions+=3*((ov+15)/16);
  } else {
  owned(ov_plan.tasks(),[&](std::size_t task) {
    const auto me=ov_plan.representative(task);
    const auto m=me/v,e=me%v;
    for(std::size_t j=0;j<o;++j) for(std::size_t b=0;b<v;++b) {
      wa(me,j*v+b)=w(2*m,2*b+1,2*e,2*j+1);
      wb(me,j*v+b)=w(2*m,2*b+1,2*e+1,2*j);
      a0(me,j*v+b)=eri(m,o+e,o+b,j);
      b0(me,j*v+b)=-eri(m,j,o+b,o+e);
    }
    // Each row orbit is disjoint, so reconstruction has no write races.
    scatter_row(wa,me,wa.data().data()+me*ov,ov_plan,ov_plan);
    scatter_row(wb,me,wb.data().data()+me*ov,ov_plan,ov_plan);
    scatter_row(a0,me,a0.data().data()+me*ov,ov_plan,ov_plan);
    scatter_row(b0,me,b0.data().data()+me*ov,ov_plan,ov_plan);
  });
  parallel::sum_in_place(wa.data()); parallel::sum_in_place(wb.data());
  parallel::sum_in_place(a0.data()); parallel::sum_in_place(b0.data());
  }
  Matrix td({ov,ov}), tx({ov,ov}), ts({ov,ov}), singles({ov,ov});
  for(std::size_t i=0;i<o;++i) for(std::size_t a=0;a<v;++a)
    for(std::size_t m=0;m<o;++m) for(std::size_t e=0;e<v;++e) {
      td(i*v+a,m*v+e)=t2(i,m,a,e);
      tx(i*v+a,m*v+e)=t2(i,m,e,a);
      ts(i*v+a,m*v+e)=2*t2(i,m,a,e)-t2(i,m,e,a);
      singles(i*v+a,m*v+e)=t1(i,e)*t1(m,a);
    }
  Matrix q=product(ts,wa,false,&ov_plan,&ov_plan);
  add(q,product(td,wb,false,&ov_plan,&ov_plan)); add(q,product(singles,a0,false,&ov_plan,&ov_plan),-1.0);
  Matrix p=product(tx,wb,false,&ov_plan,&ov_plan);
  add(p,product(singles,b0,false,&ov_plan,&ov_plan));
  Tensor4 result({o,o,v,v});
  for(std::size_t i=0;i<o;++i) for(std::size_t j=0;j<o;++j)
    for(std::size_t a=0;a<v;++a) for(std::size_t b=0;b<v;++b)
      result(i,j,a,b)=pair(i*o+j,a*v+b)+q(i*v+a,j*v+b)+q(j*v+b,i*v+a)
          +p(j*v+a,i*v+b)+p(i*v+b,j*v+a);
  timing.intermediates_seconds+=seconds(begin);
  timing.intermediate_mixed_pair_tasks+=ov_plan.tasks()*ov;
  timing.intermediate_blas_contractions+=5*((ov_plan.tasks()+15)/16)+2*((oo_plan.tasks()+15)/16);

  // Build bounded spatial W_abef panels. Both the tau*OOVV dressing and
  // final tau*W contraction use DGEMM; never allocate full VVVV or spin T2.
  begin=Clock::now();
  std::vector<std::size_t> reps;
  std::vector<std::vector<std::size_t>> images(vv);
  for(std::size_t ab=0;ab<vv;++ab) {
    if(!options.pm_symmetry) { reps.push_back(ab); continue; }
    const auto a=ab/v,b=ab%v;
    std::size_t representative=ab;
    for(const auto& g:options.pm_symmetry->operations)
      representative=std::min(representative,(g.image[o+a]-o)*v+g.image[o+b]-o);
    if(ab!=representative) continue;
    reps.push_back(ab);
    std::vector<std::size_t> seen;
    for(std::size_t k=0;k<options.pm_symmetry->operations.size();++k) {
      const auto& g=options.pm_symmetry->operations[k];
      const auto target=(g.image[o+a]-o)*v+g.image[o+b]-o;
      if(std::find(seen.begin(),seen.end(),target)==seen.end()) {
        images[ab].push_back(k); seen.push_back(target);
      }
    }
  }
  std::size_t threads=1;
#if defined(_OPENMP)
  threads=static_cast<std::size_t>(omp_get_max_threads());
#endif
  const auto bytes_per_row=mul(vv+2*oo,sizeof(double));
  const auto per_worker=options.wabef_tile_max_bytes/threads;
  const auto rows=std::min<std::size_t>(16,per_worker/bytes_per_row);
  if(rows==0) throw std::runtime_error("native spatial W_abef tile cap cannot hold one row per OpenMP worker");
  Tensor4 wabef_result({o,o,v,v});
  owned((reps.size()+rows-1)/rows,[&](std::size_t tile) {
    const auto first=tile*rows, count=std::min(rows,reps.size()-first);
    Matrix panel({count,vv}), tab({count,oo}), out({count,oo});
    for(std::size_t k=0;k<count;++k) {
      const auto ab=reps[first+k],a=ab/v,b=ab%v;
      for(std::size_t m=0;m<o;++m) for(std::size_t n=0;n<o;++n)
        tab(k,m*o+n)=tau(m*o+n,ab);
      for(std::size_t e=0;e<v;++e) for(std::size_t f=0;f<v;++f) {
        double x=eri(o+a,o+e,o+b,o+f);
        for(std::size_t m=0;m<o;++m)
          x-=t1(m,b)*eri(o+a,o+e,m,o+f)+t1(m,a)*eri(m,o+e,o+b,o+f);
        panel(k,e*v+f)=x;
      }
    }
    gemm(tab.data().data(),integrals.data().data(),panel.data().data(),
         static_cast<int>(count),vvi,ooi,false,0.5,1.0);
    gemm(panel.data().data(),tau.data().data(),out.data().data(),
         static_cast<int>(count),ooi,vvi,true);
    for(std::size_t k=0;k<count;++k) {
      const auto ab=reps[first+k],a=ab/v,b=ab%v;
      for(std::size_t i=0;i<o;++i) for(std::size_t j=0;j<o;++j) {
        if(!options.pm_symmetry) { wabef_result(i,j,a,b)=out(k,i*o+j); continue; }
        for(const auto op:images[ab]) {
          const auto& g=options.pm_symmetry->operations[op];
          wabef_result(g.image[i],g.image[j],g.image[o+a]-o,g.image[o+b]-o)=
              g.sign[i]*g.sign[j]*g.sign[o+a]*g.sign[o+b]*out(k,i*o+j);
        }
      }
    }
  });
  parallel::sum_in_place(wabef_result.data());
  for(std::size_t k=0;k<result.size();++k) result.data()[k]+=wabef_result.data()[k];
  timing.wabef_seconds+=seconds(begin);
  timing.wabef_external_tiles+=(reps.size()+rows-1)/rows;
  timing.wabef_output_tasks+=reps.size();
  timing.wabef_blas_contractions+=2*((reps.size()+rows-1)/rows);
  return result;
}
}
