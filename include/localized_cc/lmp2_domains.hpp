#pragma once

#include "localized_cc/tensor.hpp"
#include <array>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace localized_cc {

// Stage-1 admission interface: spatial G(i,j,a,b)=(ia|jb), real RHF,
// unchanged PM orbitals/labels. The provider must supply four-center ERIs.
// This bounded dense-amplitude pilot is not the production sparse solver.
using Mp2Integral = std::function<double(std::size_t,std::size_t,
                                         std::size_t,std::size_t)>;
struct Lmp2DomainOptions {
  double residual_l2_tolerance{1e-11};
  double solve_energy_error_budget{1e-9};
  double omission_budget_hartree{1e-7};
  std::size_t max_iterations{500};
  std::size_t maximum_workspace_bytes{512ULL*1024*1024};
  std::function<void(std::size_t,double,double)> observer;
};
struct Lmp2BlockScore {
  // Sorted occupied labels followed by sorted virtual labels. This key
  // groups every ordered block under separate occupied/virtual exchanges.
  std::array<int,4> labels{};
  std::size_t monomer_span{}, elements{};
  double amplitude_norm{}, energy_weight_norm{}, signed_energy{};
  double absolute_energy_sum{}, cauchy_bound{};
  bool retained{true};
};
struct Lmp2DomainResult {
  Tensor4 amplitudes; // T(i,j,a,b)=T(j,i,b,a); NOT spin antisymmetric
  std::vector<Lmp2BlockScore> blocks;
  std::array<double,4> full_nm{}, retained_nm{}, omitted_absolute_nm{};
  std::size_t iterations{}, workspace_upper_bytes{}, retained_elements{};
  double residual_l2{}, minimum_denominator{}, solve_energy_error_bound{};
  double omission_bound{}, pair_symmetry_error{};
};

// Solves K T = -G using preconditioned CG, with complete F_oo and F_vv.
// No canonical MO integral tensor, diagonal-denominator approximation,
// Laplace fit, CCSD solve, or triples construction is performed.
Lmp2DomainResult select_lmp2_domains(
    const Matrix& fock_occupied, const Matrix& fock_virtual,
    const Mp2Integral& integrals, const std::vector<int>& occupied_labels,
    const std::vector<int>& virtual_labels,
    const Lmp2DomainOptions& options = {});

// Creates a new directory exclusively. Never overwrites prior audit files.
// Completion marker is written last. No selector mask is applied to CCSD.
void write_lmp2_domain_audit(const std::filesystem::path& directory,
    const Lmp2DomainResult& result, const Lmp2DomainOptions& options,
    const std::vector<int>& occupied_labels,
    const std::vector<int>& virtual_labels, const std::string& provenance);
} // namespace localized_cc
