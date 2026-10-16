#include "localized_cc/pm_adapted.hpp"
#define EIGEN_USE_BLAS
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <numeric>

namespace localized_cc {
namespace {
using E = Eigen::MatrixXd;
using M = modernqc::linalg::Matrix;
E eigen(const M& m) {
  E a(m.rows(),m.columns());
  for(std::size_t i=0;i<m.rows();++i)
    for(std::size_t j=0;j<m.columns();++j) a(i,j)=m(i,j);
  return a;
}
M matrix(const E& a) {
  M m(static_cast<std::size_t>(a.rows()),static_cast<std::size_t>(a.cols()));
  for(Eigen::Index i=0;i<a.rows();++i)
    for(Eigen::Index j=0;j<a.cols();++j) m(i,j)=a(i,j);
  return m;
}
void gate(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
double error(const E& a) {return a.allFinite()?a.cwiseAbs().maxCoeff():INFINITY;}
E polar(const E& a) {
  Eigen::SelfAdjointEigenSolver<E> eig(a.transpose()*a);
  gate(eig.info()==Eigen::Success && eig.eigenvalues().minCoeff()>1.e-8,
       "cyclic PM symmetry seed does not span the complete orbital space");
  return a*eig.eigenvectors()*eig.eigenvalues().array().rsqrt().matrix().asDiagonal()*eig.eigenvectors().transpose();
}
}

M inversion_ao_action(const modernqc::integrals::Libint2IntegralProvider& provider,
    const modernqc::molecule::Molecule& molecule,const M& overlap) {
  const auto& atoms=molecule.atoms();
  gate(!atoms.empty() && atoms.size()%3==0,"inversion audit requires O,H,H waters");
  gate(overlap.rows()>0 && overlap.rows()==overlap.columns(),"inversion AO overlap dimensions mismatch");
  Eigen::Vector3d center=Eigen::Vector3d::Zero();
  for(const auto& atom:atoms) for(int k=0;k<3;++k) center[k]+=atom.position_bohr[k]/static_cast<double>(atoms.size());
  std::vector<std::size_t> image(atoms.size(),atoms.size());
  for(std::size_t p=0;p<atoms.size();++p) {
    std::size_t matches=0;
    for(std::size_t q=0;q<atoms.size();++q) {
      if(atoms[p].atomic_number!=atoms[q].atomic_number) continue;
      double residual=0.;
      for(int k=0;k<3;++k) residual=std::max(residual,std::abs(atoms[p].position_bohr[k]+atoms[q].position_bohr[k]-2.*center[k]));
      if(residual<=1.e-9) {image[p]=q;++matches;}
    }
    gate(matches==1,"geometry has no unique inversion atom map at 1e-9 bohr; no coordinates changed");
  }
  for(std::size_t p=0;p<atoms.size();++p)
    gate(image[image[p]]==p,"inversion atom-map closure failed");
  for(std::size_t m=0;m<atoms.size()/3;++m)
    gate(atoms[3*m].atomic_number==8 && atoms[3*m+1].atomic_number==1 && atoms[3*m+2].atomic_number==1 &&
        image[3*m]/3==image[3*m+1]/3 && image[3*m]/3==image[3*m+2]/3,
        "inversion does not map complete O,H,H monomers");
  const auto& basis=provider.basis_metadata();
  E action=E::Zero(overlap.rows(),overlap.rows());
  for(const auto& shell:basis.shells) {
    const modernqc::basis::Shell* target=nullptr;
    std::size_t ordinal=0;
    for(const auto& s:basis.shells) {if(&s==&shell) break;if(s.atom_index==shell.atom_index) ++ordinal;}
    for(const auto& s:basis.shells) if(s.atom_index==image[shell.atom_index]) {
      if(ordinal==0) {target=&s;break;} --ordinal;
    }
    gate(target && target->function_count==shell.function_count && target->angular_momentum==shell.angular_momentum &&
        target->exponents==shell.exponents && target->contraction_coefficients==shell.contraction_coefficients,
        "inversion equivalent-atom basis mismatch");
    // Both Cartesian and real spherical functions have parity (-1)^l.
    const double parity=shell.angular_momentum%2 ? -1. : 1.;
    for(std::size_t f=0;f<shell.function_count;++f) action(target->first_ao+f,shell.first_ao+f)=parity;
  }
  const E s=eigen(overlap);
  gate(error(action*action-E::Identity(s.rows(),s.rows()))<=1.e-10 &&
       error(action.transpose()*s*action-s)<=1.e-10,
       "inversion AO metric/closure audit failed");
  return matrix(action);
}

PmSymmetry match_pm_inversion(const M& occupied,const M& virtuals,
    const M& overlap,const M& ao_action,const std::function<void(const std::string&)>& progress) {
  const E s=eigen(overlap),a=eigen(ao_action);
  gate(s.rows()>0 && s.rows()==s.cols() && a.rows()==s.rows() && a.cols()==s.cols() &&
       occupied.rows()==overlap.rows() && virtuals.rows()==overlap.rows() &&
       occupied.columns()>0 && virtuals.columns()>0,"inversion PM dimensions mismatch");
  const auto n=occupied.columns()+virtuals.columns();
  PmSymmetry result{occupied.columns(),virtuals.columns(),{},0.};
  PmSymmetryOperation identity{std::vector<std::size_t>(n),std::vector<int>(n,1)};
  std::iota(identity.image.begin(),identity.image.end(),0);
  auto generator=identity;
  const auto match=[&](const M& input,std::size_t offset,const char* name) {
    const E c=eigen(input),transformed=a*c;
    gate(error(c.transpose()*s*c-E::Identity(c.cols(),c.cols()))<=1.e-10,
         "PM inversion requires orthonormal saved orbitals");
    const E representation=c.transpose()*s*transformed;
    double maximum=0.;
    std::vector<bool> seen(static_cast<std::size_t>(c.cols()),false);
    for(Eigen::Index p=0;p<c.cols();++p) {
      Eigen::Index q=0; representation.col(p).cwiseAbs().maxCoeff(&q);
      const int sign=representation(q,p)<0.?-1:1;
      const E delta=transformed.col(p)-sign*c.col(q);
      const double squared=(delta.transpose()*s*delta)(0,0);
      gate(std::isfinite(squared) && squared>=-1.e-20,"invalid PM inversion metric residual");
      const double residual=std::sqrt(std::max(0.,squared));
      maximum=std::max(maximum,residual);
      if(residual>1.e-10 || seen[static_cast<std::size_t>(q)]) {
        std::ostringstream message; message<<std::scientific<<std::setprecision(12)
          <<"saved PM "<<name<<" orbital="<<p<<" inversion signed-permutation residual="<<residual
          <<"; limit=1e-10; reuse refused; no orbitals changed";
        throw std::runtime_error(message.str());
      }
      seen[static_cast<std::size_t>(q)]=true;
      generator.image[offset+static_cast<std::size_t>(p)]=offset+static_cast<std::size_t>(q);
      generator.sign[offset+static_cast<std::size_t>(p)]=sign;
    }
    if(progress) {std::ostringstream message; message<<std::scientific<<std::setprecision(12)
      <<"saved PM "<<name<<" inversion maximum metric residual="<<maximum;progress(message.str());}
  };
  match(occupied,0,"occupied"); match(virtuals,occupied.columns(),"virtual");
  result.operations={identity,generator};
  validate_pm_symmetry_group(result);
  return result;
}

M cyclic_ao_symmetry_action(const modernqc::integrals::Libint2IntegralProvider& provider,
    const modernqc::molecule::Molecule& molecule,const M& overlap,
    const std::vector<int>& cycle) {
  const auto& atoms=molecule.atoms();
  const auto order=cycle.size();
  gate((order==3 || order==4) && atoms.size()==3*order,
       "adapted PM requires a W3 or W4 cycle");
  std::vector<bool> seen(order,false);
  for(int m:cycle) {
    gate(m>=0 && static_cast<std::size_t>(m)<order && !seen[m],"invalid water cycle");
    seen[m]=true;
  }
  int water=0;
  for(std::size_t power=1;power<=order;++power) {
    water=cycle[water];
    gate((water==0)==(power==order),"water map is not a complete cycle");
  }
  E x(3,atoms.size());
  for(std::size_t a=0;a<atoms.size();++a) for(int k=0;k<3;++k) x(k,a)=atoms[a].position_bohr[k];
  Eigen::Vector3d center=x.rowwise().mean();
  x.colwise()-=center;
  Eigen::Matrix3d q=Eigen::Matrix3d::Identity();
  std::vector<std::size_t> image(atoms.size());
  bool found=false;
  for(unsigned swaps=0;swaps<(1U<<order) && !found;++swaps) {
    E y(3,atoms.size());
    for(std::size_t m=0;m<order;++m) {
      image[3*m]=3*static_cast<std::size_t>(cycle[m]);
      image[3*m+1]=image[3*m]+1+((swaps>>m)&1U);
      image[3*m+2]=image[3*m]+2-((swaps>>m)&1U);
    }
    bool atom_closure=true;
    for(std::size_t a=0;a<atoms.size();++a) {
      y.col(a)=x.col(image[a]);
      auto mapped=a;
      for(std::size_t power=0;power<order;++power) mapped=image[mapped];
      atom_closure=atom_closure && mapped==a;
    }
    Eigen::JacobiSVD<E> svd(y*x.transpose(),Eigen::ComputeFullU|Eigen::ComputeFullV);
    q=svd.matrixU()*svd.matrixV().transpose(); // Includes improper S4 operations.
    Eigen::Matrix3d closure=Eigen::Matrix3d::Identity();
    for(std::size_t power=0;power<order;++power) closure=(closure*q).eval();
    found=atom_closure && error(q*x-y)<1.e-9 &&
        error(closure-Eigen::Matrix3d::Identity())<1.e-10 &&
        (order!=3 || std::abs(q.determinant()-1.)<1.e-10);
  }
  gate(found,"cyclic Cartesian symmetry operation failed geometry/closure audit");
  const auto& basis=provider.basis_metadata();
  E action=E::Zero(overlap.rows(),overlap.rows());
  for(const auto& shell:basis.shells) {
    const modernqc::basis::Shell* target=nullptr;
    // Shell ordinal on each equivalent atom disambiguates repeated contractions.
    std::size_t ordinal=0;
    for(const auto& s:basis.shells) {if(&s==&shell) break; if(s.atom_index==shell.atom_index) ++ordinal;}
    for(const auto& s:basis.shells) if(s.atom_index==image[shell.atom_index]) {
      if(ordinal==0) {target=&s;break;} --ordinal;
    }
    gate(target && target->function_count==shell.function_count &&
         target->angular_momentum==shell.angular_momentum &&
         target->exponents==shell.exponents && target->contraction_coefficients==shell.contraction_coefficients,
         "cyclic symmetry requires identical equivalent-atom basis shells");
    auto values=[&](int offset) {
      std::vector<std::array<double,3>> p,r;
      for(int j=0;j<64;++j) {
        const double z=1.-2.*(static_cast<double>(j)+0.5)/64.;
        const double angle=(static_cast<double>(j)+offset)*2.399963229728653;
        const double radius=(0.8+0.1*(j%5))/std::sqrt(*std::max_element(shell.exponents.begin(),shell.exponents.end()));
        Eigen::Vector3d point;
        point<<radius*std::sqrt(1.-z*z)*std::cos(angle),radius*std::sqrt(1.-z*z)*std::sin(angle),radius*z;
        for(int k=0;k<3;++k) point[k]+=atoms[shell.atom_index].position_bohr[k];
        const Eigen::Vector3d rotated=q*(point-center)+center;
        p.push_back({point[0],point[1],point[2]});
        r.push_back({rotated[0],rotated[1],rotated[2]});
      }
      E source=eigen(provider.evaluate_basis_values(p)).middleCols(shell.first_ao,shell.function_count);
      E destination=eigen(provider.evaluate_basis_values(r)).middleCols(target->first_ao,target->function_count);
      const double norm=std::max(source.norm(),destination.norm());
      gate(std::isfinite(norm)&&norm>0.,"invalid cyclic shell collocation norm");
      return std::pair<E,E>{source/norm,destination/norm};
    };
    const auto [source,destination]=values(0);
    Eigen::ColPivHouseholderQR<E> qr(destination);
    gate(qr.rank()==static_cast<Eigen::Index>(shell.function_count),"rank-deficient cyclic shell collocation");
    const E block=qr.solve(source);
    const auto [check_source,check_destination]=values(7);
    gate(error(check_destination*block-check_source)<1.e-11,"cyclic AO action failed independent collocation audit");
    action.block(target->first_ao,shell.first_ao,shell.function_count,shell.function_count)=block;
  }
  const E s=eigen(overlap);
  E closure=E::Identity(s.rows(),s.rows());
  for(std::size_t power=0;power<order;++power) closure=(closure*action).eval();
  gate(error(action.transpose()*s*action-s)<1.e-10 &&
       error(closure-E::Identity(s.rows(),s.rows()))<1.e-10,
       "cyclic AO symmetry failed metric/group closure audit");
  return matrix(action);
}

AdaptedPmResult adapt_cyclic_pm(const M& canonical,const M& seed,
    const std::vector<int>& labels,const M& overlap,const M& ao_action,
    const M& meta_lowdin,const std::vector<std::size_t>& ao_atoms,std::size_t order,
    const std::function<void(const std::string&)>& progress) {
  const E c=eigen(canonical),s=eigen(overlap),a=eigen(ao_action);
  gate(order==3 || order==4,"adapted PM supports cyclic orders three and four");
  const Eigen::Index n=c.cols(),d=n/static_cast<Eigen::Index>(order);
  gate(n>0 && n%order==0 && labels.size()==static_cast<std::size_t>(n) &&
       ao_atoms.size()==static_cast<std::size_t>(c.rows()) &&
       s.rows()==c.rows() && s.cols()==c.rows() &&
       a.rows()==c.rows() && a.cols()==c.rows() &&
       seed.rows()==canonical.rows() && seed.columns()==canonical.columns() &&
       meta_lowdin.rows()==canonical.rows() && meta_lowdin.columns()==canonical.rows(),
       "cyclic PM adaptation dimensions mismatch");
  gate(c.allFinite() && s.allFinite() && a.allFinite() &&
       error(c.transpose()*s*c-E::Identity(n,n))<1.e-10,
       "invalid canonical space for cyclic PM adaptation");
  const E g=c.transpose()*s*a*c;
  gate(error(a*c-c*g)<1.e-9 && error(g.transpose()*g-E::Identity(n,n))<1.e-9,
       "cyclic active orbital space is not symmetry invariant");
  const int first=*std::min_element(labels.begin(),labels.end());
  std::vector<std::size_t> label_counts(order,0);
  for(int label:labels) {
    gate(label>=first && label-first<static_cast<int>(order),"invalid cyclic PM seed label");
    ++label_counts[static_cast<std::size_t>(label-first)];
  }
  for(auto number:label_counts)
    gate(number==static_cast<std::size_t>(d),"cyclic PM seed has unequal monomer populations");
  E seeds(n,d); Eigen::Index count=0;
  const E seed_rotation=c.transpose()*s*eigen(seed);
  for(Eigen::Index j=0;j<n;++j) if(labels[j]==first) {
    gate(count<d,"cyclic PM seed has unequal monomer populations"); seeds.col(count++)=seed_rotation.col(j);
  }
  gate(count==d,"cyclic PM seed has unequal monomer populations");
  E u(n,n),p=E::Zero(n,n);
  for(std::size_t power=0;power<order;++power) {
    u.middleCols(power*d,d)=seeds;
    seeds=g*seeds;
    p.block(((power+1)%order)*d,power*d,d,d).setIdentity();
  }
  u=polar(u);
  gate(error(g*u-u*p)<1.e-9,"cyclic PM orbit seed failed equivariance audit");
  const E populations=eigen(meta_lowdin).transpose()*s*c;
  std::vector<E> atom_populations(3*order,E::Zero(n,n));
  for(Eigen::Index row=0;row<populations.rows();++row) {
    gate(ao_atoms[row]<3*order,"invalid PM AO atom index");
    atom_populations[ao_atoms[row]].noalias()+=populations.row(row).transpose()*populations.row(row);
  }
  auto project=[&](const E& input) -> E {
    E sum=E::Zero(n,n),power=E::Identity(n,n);
    const E skew=0.5*(input-input.transpose());
    for(std::size_t j=0;j<order;++j) {sum.noalias()+=power.transpose()*skew*power;power=power*p;}
    return sum/static_cast<double>(order);
  };
  bool converged=false;
  double value=0.;
  double last_norm=INFINITY;
  int completed_iterations=0;
  for(int iteration=0;iteration<2000;++iteration) {
    completed_iterations=iteration;
    std::vector<E> blocks;
    E raw=E::Zero(n,n);
    value=0.;
    for(const auto& atom:atom_populations) {
      E b=u.transpose()*atom*u;
      b=(0.5*(b+b.transpose())).eval();
      const E diagonal=b.diagonal().asDiagonal();
      value+=b.diagonal().squaredNorm();
      raw.noalias()+=2.*(b*diagonal-diagonal*b);
      blocks.push_back(std::move(b));
    }
    const E k=project(raw);
    const double norm=k.norm();
    last_norm=norm;
    gate(std::isfinite(norm)&&std::isfinite(value),"nonfinite adapted PM objective/gradient");
    if(progress) {
      std::ostringstream message;
      message<<std::scientific<<std::setprecision(12)
          <<"symmetry-adapted PM iteration="<<iteration<<" objective="<<value
          <<" gradient="<<norm<<" target="<<1.e-7;
      progress(message.str());
    }
    if(norm<1.e-7) {converged=true;break;}
    // Analytic Riemannian Hessian action, no finite differences. For B=U^T A U,
    // B'=[B,X]. The connection term makes the action self-adjoint in the
    // Frobenius metric on the symmetry-preserving orthogonal subgroup.
    auto hessian=[&](const E& x) -> E {
      E h=E::Zero(n,n);
      for(const auto& b:blocks) {
        const E db=b*x-x*b;
        const E diagonal=b.diagonal().asDiagonal(),dd=db.diagonal().asDiagonal();
        h.noalias()+=2.*(db*diagonal-diagonal*db+b*dd-dd*b);
      }
      h.noalias()+=0.5*(x*k-k*x);
      return project(h);
    };
    // Truncated CG solves -H d = gradient. Stop at nonpositive curvature;
    // globalization below bounds the rotation, not the inverse-curvature scale.
    E direction=E::Zero(n,n),residual=k,search=k;
    double rr=residual.squaredNorm();
    int actions=0;
    for(int inner=0;inner<60;++inner) {
      const E applied=-hessian(search); ++actions;
      const double curvature=(search.array()*applied.array()).sum();
      if(!std::isfinite(curvature)||curvature<=1.e-14*search.squaredNorm()) break;
      const double alpha=rr/curvature;
      direction+=alpha*search;
      residual-=alpha*applied;
      const double next=residual.squaredNorm();
      if(next<=0.01*norm*norm) break;
      search=residual+(next/rr)*search;rr=next;
    }
    direction=project(direction);
    double slope=(k.array()*direction.array()).sum();
    if(!direction.allFinite()||!std::isfinite(slope)||slope<=0.) direction=k;
    bool accepted=false;
    double accepted_gain=0.,accepted_step=0.;
    int trials=0;
    for(int fallback=0;fallback<2 && !accepted;++fallback) {
      if(fallback) direction=k;
      slope=(k.array()*direction.array()).sum();
      double step=std::min(1.,0.2/direction.norm());
      for(int trial=0;trial<40;++trial) {
        ++trials;
        const E identity=E::Identity(n,n);
        // R-I=(I-X/2)^-1 X avoids subtracting nearly equal identity matrices.
        const E delta=(identity-0.5*step*direction).partialPivLu().solve(step*direction);
        long double gain=0.;
        for(const auto& b:blocks) {
          const E bd=b*delta;
          const E change=bd+bd.transpose()+delta.transpose()*bd;
          for(Eigen::Index j=0;j<n;++j) {
            const long double dq=change(j,j);
            gain+=dq*(2.L*static_cast<long double>(b(j,j))+dq);
          }
        }
        if(std::isfinite(gain)&&gain>0. && gain>=1.e-4L*step*slope) {
          u+= (u*delta).eval();accepted=true;
          accepted_gain=static_cast<double>(gain);accepted_step=step;break;
        }
        step*=0.5;
      }
    }
    if(progress) {
      std::ostringstream message;
      message<<std::scientific<<std::setprecision(12)
          <<"symmetry-adapted PM step; hessian_actions="<<actions
          <<" line_trials="<<trials<<" step="<<accepted_step
          <<" objective_gain="<<accepted_gain;
      progress(message.str());
    }
    gate(accepted,"symmetry-adapted PM line search failed; no unconverged orbitals accepted");
  }
  if(!converged) {
    std::ostringstream message;
    message<<std::scientific<<std::setprecision(12)
        <<"symmetry-adapted PM did not converge in 2000 iterations; gradient="
        <<last_norm<<" target="<<1.e-7;
    throw std::runtime_error(message.str());
  }
  gate(error(u.transpose()*u-E::Identity(n,n))<1.e-10 && error(g*u-u*p)<1.e-9,
       "adapted PM failed final orthogonality/equivariance audit");
  return {matrix(c*u),matrix(u),value,last_norm,completed_iterations};
}

PmSymmetry cyclic_adapted_permutation(std::size_t occupied,std::size_t virtuals,std::size_t order) {
  gate((order==3 || order==4) && occupied>0 && virtuals>0 && occupied%order==0 && virtuals%order==0,"invalid adapted PM space sizes");
  PmSymmetry result{occupied,virtuals,{},0.};
  for(std::size_t g=0;g<order;++g) {
    PmSymmetryOperation op;
    for(const auto size:{occupied,virtuals}) {
      const auto offset=op.image.size();
      for(std::size_t j=0;j<size;++j) {op.image.push_back(offset+(j+g*(size/order))%size);op.sign.push_back(1);}
    }
    result.operations.push_back(std::move(op));
  }
  return result;
}

M w4_ao_symmetry_action(const modernqc::integrals::Libint2IntegralProvider& provider,
    const modernqc::molecule::Molecule& molecule,const M& overlap,
    const std::array<int,4>& cycle) {
  return cyclic_ao_symmetry_action(provider,molecule,overlap,
      std::vector<int>(cycle.begin(),cycle.end()));
}
AdaptedPmResult adapt_w4_pm(const M& canonical,const M& seed,
    const std::vector<int>& labels,const M& overlap,const M& ao_action,
    const M& meta_lowdin,const std::vector<std::size_t>& ao_atoms,
    const std::function<void(const std::string&)>& progress) {
  return adapt_cyclic_pm(canonical,seed,labels,overlap,ao_action,meta_lowdin,ao_atoms,4,progress);
}
PmSymmetry w4_adapted_permutation(std::size_t occupied,std::size_t virtuals) {
  return cyclic_adapted_permutation(occupied,virtuals,4);
}
} // namespace localized_cc
