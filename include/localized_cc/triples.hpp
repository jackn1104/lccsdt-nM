#pragma once

#include "localized_cc/frozen_core.hpp"
#include "localized_cc/tensor.hpp"
#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <vector>

namespace localized_cc {

struct TriplesProgress {
  std::size_t completed_iterations{};
  double residual_norm{};
  bool converged{};
  const Tensor6& response;
  const std::vector<double>& residual_history;
};
using TriplesObserver = std::function<void(const TriplesProgress&)>;

struct TriplesOptions {
  double residual_tolerance{1.0e-10};
  double denominator_floor{1.0e-10};
  double memory_limit_mib{512.0};
  std::size_t max_iterations{200};
  std::size_t diis_start{2};
  std::size_t diis_size{8};
  std::optional<Tensor6> initial_response;
  std::size_t completed_iterations{};
  std::vector<double> initial_solver_history;
  std::size_t checkpoint_interval{1};
  TriplesObserver observer;
};
struct TriplesResult {
  double energy{};
  Tensor6 right_moment, left_moment, response, energy_contribution;
  double residual_norm{};
  std::size_t iterations{};
  std::vector<double> solver_history;
  bool converged{};
  std::map<std::string,double> memory_mib;
};

std::map<std::string, double>
estimate_triples_memory_mib(std::size_t occupied, std::size_t virtuals,
                            std::size_t diis_size);
Tensor6 triples_fock_action(const Tensor6& x,const Matrix& f_oo,const Matrix& f_vv);
std::pair<Tensor6,Tensor6> build_triples_moments(const Matrix& fock,const Tensor4& eri,
                                                 const Matrix& t1,const Tensor4& t2);
std::pair<Tensor6, Tensor6>
build_triples_moments(const SpatialRhfHamiltonian& hamiltonian,
                      const Matrix& t1, const Tensor4& t2);
TriplesResult solve_triples(const Matrix& fock,const Tensor4& eri,const Matrix& t1,
                            const Tensor4& t2,const TriplesOptions& options = {});
double triples_antisymmetry_error(const Tensor6& tensor);

} // namespace localized_cc
