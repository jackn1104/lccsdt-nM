#include "localized_cc/triples.hpp"
#include "localized_cc/diis.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace localized_cc {
namespace {
constexpr std::array<std::array<std::size_t,3>,6> perms{{{{0,1,2}},{{0,2,1}},{{1,0,2}},{{1,2,0}},{{2,0,1}},{{2,1,0}}}};
int parity(const std::array<std::size_t,3>& p){int n=0;for(int i=0;i<3;++i)for(int j=i+1;j<3;++j)n+=p[static_cast<std::size_t>(i)]>p[static_cast<std::size_t>(j)];return n%2?-1:1;}
Tensor6 anti(const Tensor6& x){const auto d=x.dimensions();Tensor6 y(d);for(std::size_t i=0;i<d[0];++i)for(std::size_t j=0;j<d[1];++j)for(std::size_t k=0;k<d[2];++k)for(std::size_t a=0;a<d[3];++a)for(std::size_t b=0;b<d[4];++b)for(std::size_t c=0;c<d[5];++c){std::array<std::size_t,3> oi{i,j,k},va{a,b,c};double z=0;for(const auto& po:perms)for(const auto& pv:perms)z+=static_cast<double>(parity(po)*parity(pv))*x(oi[po[0]],oi[po[1]],oi[po[2]],va[pv[0]],va[pv[1]],va[pv[2]]);y(i,j,k,a,b,c)=z;}return y;}

template <typename Eri>
std::pair<Tensor6, Tensor6>
build_triples_moments_impl(const Matrix& f, const Eri& g, const Matrix& s,
                           const Tensor4& t) {
  const auto o = s.dimensions()[0], v = s.dimensions()[1];
  const auto orbitals = o + v;
  if (o == 0 || v == 0 ||
      f.dimensions() != std::array<std::size_t, 2>{orbitals, orbitals} ||
      g.dimensions() !=
          std::array<std::size_t, 4>{orbitals, orbitals, orbitals, orbitals} ||
      t.dimensions() != std::array<std::size_t, 4>{o, o, v, v})
    throw std::invalid_argument("triples moment dimensions mismatch");
  Tensor6 r({o, o, o, v, v, v}), l({o, o, o, v, v, v});
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t k = 0; k < o; ++k)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t b = 0; b < v; ++b)
            for (std::size_t c = 0; c < v; ++c) {
              double z = 0.0;
              for (std::size_t e = 0; e < v; ++e)
                z += t(j, k, a, e) * g(o + b, o + c, o + e, i);
              for (std::size_t m = 0; m < o; ++m)
                z -= t(i, m, b, c) * g(m, o + a, j, k);
              r(i, j, k, a, b, c) = z;
              l(i, j, k, a, b, c) =
                  z + s(i, a) * g(o + b, o + c, j, k) +
                  f(o + a, i) * t(j, k, b, c);
            }
  r = anti(r);
  l = anti(l);
  for (auto& x : r.data()) x *= 0.25;
  for (auto& x : l.data()) x *= 0.25;
  return {std::move(r), std::move(l)};
}
}

std::map<std::string, double>
estimate_triples_memory_mib(const std::size_t occupied,
                            const std::size_t virtuals,
                            const std::size_t diis_size) {
  if (occupied == 0 || virtuals == 0)
    throw std::invalid_argument(
        "triples memory estimate requires positive dimensions");
  const long double elements = static_cast<long double>(occupied) * occupied *
                               occupied * virtuals * virtuals * virtuals;
  const long double tensor_mib =
      elements * static_cast<long double>(sizeof(double)) / 1048576.0L;
  if (!std::isfinite(tensor_mib) ||
      tensor_mib > static_cast<long double>(std::numeric_limits<double>::max()))
    throw std::overflow_error("triples memory estimate exceeds double range");
  std::map<std::string, double> result{
      {"t1", static_cast<double>(occupied) * virtuals * sizeof(double) /
                 1048576.0},
      {"t2", static_cast<double>(occupied) * occupied * virtuals * virtuals *
                 sizeof(double) / 1048576.0},
      {"right_triples_moment", static_cast<double>(tensor_mib)},
      {"left_triples_moment", static_cast<double>(tensor_mib)},
      {"triples_response", static_cast<double>(tensor_mib)},
      {"triples_energy_contribution", static_cast<double>(tensor_mib)},
      {"temporary_triples_arrays",
       static_cast<double>(tensor_mib * (4.0L + diis_size))}};
  double total = 0.0;
  for (const auto& [name, value] : result) {
    (void)name;
    total += value;
  }
  result["total"] = total;
  return result;
}

Tensor6 triples_fock_action(const Tensor6& x,const Matrix& foo,const Matrix& fvv){const auto d=x.dimensions();const auto o=d[0],v=d[3];if(d!=std::array<std::size_t,6>{o,o,o,v,v,v}||foo.dimensions()!=std::array<std::size_t,2>{o,o}||fvv.dimensions()!=std::array<std::size_t,2>{v,v})throw std::invalid_argument("triples Fock-action dimensions mismatch");Tensor6 y(d);
for(std::size_t i=0;i<o;++i)for(std::size_t j=0;j<o;++j)for(std::size_t k=0;k<o;++k)for(std::size_t a=0;a<v;++a)for(std::size_t b=0;b<v;++b)for(std::size_t c=0;c<v;++c){double z=0;for(std::size_t q=0;q<v;++q)z+=fvv(a,q)*x(i,j,k,q,b,c)+fvv(b,q)*x(i,j,k,a,q,c)+fvv(c,q)*x(i,j,k,a,b,q);for(std::size_t l=0;l<o;++l)z-=foo(l,i)*x(l,j,k,a,b,c)+foo(l,j)*x(i,l,k,a,b,c)+foo(l,k)*x(i,j,l,a,b,c);y(i,j,k,a,b,c)=z;}return y;}

std::pair<Tensor6,Tensor6> build_triples_moments(const Matrix& f,const Tensor4& g,const Matrix& s,const Tensor4& t){return build_triples_moments_impl(f,g,s,t);}

std::pair<Tensor6, Tensor6>
build_triples_moments(const SpatialRhfHamiltonian& hamiltonian,
                      const Matrix& t1, const Tensor4& t2) {
  const auto spin_fock =
      expand_spatial_one_body_to_spin_orbital(hamiltonian.fock);
  const SpinOrbitalEriView spin_eri(hamiltonian);
  return build_triples_moments_impl(spin_fock, spin_eri, t1, t2);
}

TriplesResult solve_triples(const Matrix& f,const Tensor4& g,const Matrix& s,const Tensor4& t,const TriplesOptions& opt){const auto o=s.dimensions()[0],v=s.dimensions()[1];TriplesResult out;out.memory_mib=estimate_triples_memory_mib(o,v,opt.diis_size);const double total=out.memory_mib.at("total");if(total>opt.memory_limit_mib)throw std::runtime_error("dense localized triples requires "+std::to_string(total)+" MiB on rank zero, exceeding the configured limit of "+std::to_string(opt.memory_limit_mib)+" MiB; use plan mode to inspect the estimate (this dense solver does not use distributed streaming)");
  auto moments=build_triples_moments(f,g,s,t);out.right_moment=std::move(moments.first);out.left_moment=std::move(moments.second);Tensor6 d({o,o,o,v,v,v}),x({o,o,o,v,v,v});for(std::size_t i=0;i<o;++i)for(std::size_t j=0;j<o;++j)for(std::size_t k=0;k<o;++k)for(std::size_t a=0;a<v;++a)for(std::size_t b=0;b<v;++b)for(std::size_t c=0;c<v;++c){const double q=f(i,i)+f(j,j)+f(k,k)-f(o+a,o+a)-f(o+b,o+b)-f(o+c,o+c);if(std::abs(q)<opt.denominator_floor)throw std::runtime_error("small triples preconditioner");d(i,j,k,a,b,c)=q;x(i,j,k,a,b,c)=out.right_moment(i,j,k,a,b,c)/q;}
  if(opt.initial_response){if(opt.initial_response->dimensions()!=x.dimensions())throw std::invalid_argument("triples restart response dimensions mismatch");for(const auto value:opt.initial_response->data())if(!std::isfinite(value))throw std::invalid_argument("non-finite restarted triples response");x=*opt.initial_response;}
  Matrix foo({o,o}),fvv({v,v});for(std::size_t i=0;i<o;++i)for(std::size_t j=0;j<o;++j)foo(i,j)=f(i,j);for(std::size_t a=0;a<v;++a)for(std::size_t b=0;b<v;++b)fvv(a,b)=f(o+a,o+b);Diis diis(opt.diis_size);
  out.solver_history=opt.initial_solver_history;if(!out.solver_history.empty()&&out.solver_history.size()!=opt.completed_iterations)throw std::invalid_argument("triples restart history/iteration mismatch");
  const auto iteration_limit=opt.initial_response?std::max(opt.max_iterations,opt.completed_iterations+2):opt.max_iterations;
  for(std::size_t it=opt.completed_iterations+1;it<=iteration_limit;++it){const auto ax=triples_fock_action(x,foo,fvv);std::vector<double> residual(x.size());for(std::size_t q=0;q<x.size();++q)residual[q]=-out.right_moment.data()[q]-ax.data()[q];out.residual_norm=norm(residual);out.solver_history.push_back(out.residual_norm);out.iterations=it;if(out.residual_norm<=opt.residual_tolerance*std::max(1.0,norm(out.right_moment.data())))out.converged=true;if(!out.converged){std::vector<double> next=x.data();for(std::size_t q=0;q<next.size();++q)next[q]+=residual[q]/(-d.data()[q]);if(it>=opt.diis_start)next=diis.extrapolate(next,residual);x.data()=std::move(next);}if(opt.observer&&(out.converged||opt.checkpoint_interval==0||it%opt.checkpoint_interval==0))opt.observer(TriplesProgress{it,out.residual_norm,out.converged,x,out.solver_history});if(out.converged)break;}
  out.response=std::move(x);out.energy_contribution=Tensor6({o,o,o,v,v,v});for(std::size_t q=0;q<out.response.size();++q){out.energy_contribution.data()[q]=out.left_moment.data()[q]*out.response.data()[q]/36.0;out.energy+=out.energy_contribution.data()[q];}return out;}

double triples_antisymmetry_error(const Tensor6& x){const auto d=x.dimensions();double e=0;for(std::size_t i=0;i<d[0];++i)for(std::size_t j=0;j<d[1];++j)for(std::size_t k=0;k<d[2];++k)for(std::size_t a=0;a<d[3];++a)for(std::size_t b=0;b<d[4];++b)for(std::size_t c=0;c<d[5];++c)e=std::max({e,std::abs(x(i,j,k,a,b,c)+x(j,i,k,a,b,c)),std::abs(x(i,j,k,a,b,c)+x(i,k,j,a,b,c)),std::abs(x(i,j,k,a,b,c)+x(i,j,k,b,a,c)),std::abs(x(i,j,k,a,b,c)+x(i,j,k,a,c,b))});return e;}
} // namespace localized_cc
