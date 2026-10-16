#pragma once

#include "modernqc/integrals/integral_provider.hpp"
#include "modernqc/linalg/matrix.hpp"
#include "modernqc/molecule/molecule.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace modernqc::scf {

struct RhfOptions {
  int maximum_iterations{100};
  double energy_tolerance{1.0e-10};
  double density_rms_tolerance{1.0e-8};
  double commutator_rms_tolerance{1.0e-7};
  double linear_dependency_tolerance{1.0e-8};
  bool diis{true};
  int diis_start{2};
  int diis_subspace{8};
  double damping{0.0};
  double level_shift{0.0};
  double eri_schwarz_threshold{0.0};
  bool density_screen{false};
  double fock_contribution_threshold{0.0};
  std::size_t mpi_rank{0};
  std::size_t mpi_ranks{1};
  std::size_t fock_threads{1};
};

struct RhfIteration {
  int iteration;
  double electronic_energy;
  double total_energy;
  double energy_change;
  double density_rms;
  double commutator_rms;
};

using RhfObserver = std::function<void(const RhfIteration&)>;
using RhfDensityObserver =
    std::function<void(const RhfIteration&, const linalg::Matrix&)>;

struct RhfResult {
  bool converged;
  int iteration_count;
  double nuclear_repulsion_energy;
  double electronic_energy;
  double total_energy;
  double final_energy_change;
  double final_density_rms;
  double final_commutator_rms;
  linalg::Matrix overlap;
  linalg::Matrix core_hamiltonian;
  linalg::Matrix orthogonalizer;
  linalg::Matrix density;
  linalg::Matrix fock;
  linalg::Matrix coefficients;
  std::vector<double> orbital_energies;
  std::vector<double> occupations;
  std::size_t removed_overlap_vectors;
  std::uint64_t unique_shell_quartets_per_fock;
  std::uint64_t evaluated_shell_quartets_total;
  std::uint64_t screened_shell_quartets_total;
  std::uint64_t schwarz_screened_shell_quartets_total;
  std::uint64_t density_screened_shell_quartets_total;
  double eri_schwarz_threshold;
  bool density_screen;
  double fock_contribution_threshold;
  double final_screened_fock_maximum_error_bound;
  std::size_t mpi_ranks;
  std::size_t fock_threads;
  std::uint64_t minimum_owned_shell_quartets_per_fock;
  std::uint64_t maximum_owned_shell_quartets_per_fock;
  double direct_fock_wall_seconds_total;
  double minimum_direct_fock_wall_seconds_last;
  double maximum_direct_fock_wall_seconds_last;
  std::vector<RhfIteration> history;
};

[[nodiscard]] RhfResult run_rhf(
    const molecule::Molecule& molecule,
    const integrals::Libint2IntegralProvider& integral_provider,
    const RhfOptions& options,
    const std::optional<linalg::Matrix>& initial_density = std::nullopt,
    const RhfObserver& observer = {},
    const RhfDensityObserver& density_observer = {});

}  // namespace modernqc::scf
