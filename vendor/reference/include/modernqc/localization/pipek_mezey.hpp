#pragma once

#include "modernqc/basis/basis_set.hpp"
#include "modernqc/linalg/matrix.hpp"
#include "modernqc/localization/orbital_localization.hpp"
#include "modernqc/molecule/molecule.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <vector>

namespace modernqc::localization {

enum class PipekMezeyPopulationMethod {
  lowdin,
  meta_lowdin,
};

enum class PipekMezeyOptimizer {
  jacobi,
  pyscf_ciah,
};

struct PipekMezeySweep {
  int completed_sweeps{0};
  bool converged{false};
  double objective{0.0};
  double sweep_gain{0.0};
  double gradient_norm{0.0};
  int keyframes{0};
  int hessian_actions{0};
};

struct PipekMezeyRestartState {
  linalg::Matrix rotation;
  int completed_sweeps{0};
  std::vector<double> objective_history;
};

using PipekMezeyProgressCallback = std::function<void(
    const PipekMezeySweep&, const linalg::Matrix&,
    const std::vector<double>&)>;

struct PipekMezeyOptions {
  int maximum_sweeps{200};
  // Absolute change in the dimensionless PM objective per complete sweep.
  double objective_tolerance{1.0e-12};
  PipekMezeyOptimizer optimizer{PipekMezeyOptimizer::jacobi};
  // PySCF 2.11 CIAH defaults. A nonpositive gradient tolerance derives
  // sqrt(objective_tolerance * 0.1), matching pyscf.lo.boys.kernel.
  double gradient_tolerance{0.0};
  double maximum_rotation_step{0.05};
  int maximum_micro_iterations{20};
  int keyframe_interval{5};
  double keyframe_trust_region{5.0};
  double augmented_hessian_start_tolerance{1.0e9};
  int augmented_hessian_start_cycle{1};
  double augmented_hessian_level_shift{0.0};
  double augmented_hessian_convergence_tolerance{1.0e-12};
  double augmented_hessian_linear_dependency{1.0e-14};
  int augmented_hessian_maximum_iterations{40};
  double augmented_hessian_trust_region{3.0};
  // An assignment is ambiguous when best-minus-second population is below
  // this dimensionless threshold.
  double ambiguity_population_margin{0.05};
  // The supplied meta-Lowdin AO columns must be orthonormal in the AO metric
  // and retain the target AO atom ordering. It is required for meta_lowdin
  // and rejected for lowdin.
  PipekMezeyPopulationMethod population_method{
      PipekMezeyPopulationMethod::lowdin};
  linalg::Matrix meta_lowdin_orthogonal_ao;
  // Match the atomic projected-AO initial rotation used by the supplied
  // Python PM workflow. Restarts and explicit initial rotations take
  // precedence and are mutually exclusive with this generated guess.
  bool atomic_initial_guess{false};
  // Empty means identity. Use restart_state for an interrupted Jacobi sweep
  // sequence; initial_rotation and restart_state are mutually exclusive.
  linalg::Matrix initial_rotation;
  std::optional<PipekMezeyRestartState> restart_state;
  PipekMezeyProgressCallback progress;
  // CIAH atom contributions are owned cyclically across ranks and summed
  // through this backend-neutral collective. Jacobi PM remains serial.
  LocalizationParallelOptions parallel;
};

struct OrbitalAssignment {
  std::size_t atom_index;
  double atom_population;
  double second_atom_population;
  double atom_population_margin;
  bool atom_ambiguous;
  int monomer_id;
  double monomer_population;
  double second_monomer_population;
  double monomer_population_margin;
  bool monomer_ambiguous;
};

struct PipekMezeyResult {
  linalg::Matrix coefficients;
  linalg::Matrix rotation;
  // [localized orbital, atom]
  linalg::Matrix lowdin_populations;
  std::vector<OrbitalAssignment> assignments;
  std::vector<double> objective_history;
  int completed_sweeps;
  double final_sweep_gain;
  double rotation_orthogonality_maximum_error;
  double orbital_orthonormality_maximum_error;
  double projector_maximum_error;
  PipekMezeyPopulationMethod population_method;
  PipekMezeyOptimizer optimizer{PipekMezeyOptimizer::jacobi};
  double final_gradient_norm{0.0};
  int total_keyframes{0};
  int total_hessian_actions{0};
};

struct PipekMezeyDiagnosticPaths {
  std::filesystem::path rotation;
  std::filesystem::path coefficients;
  std::filesystem::path monomer_populations;
  std::filesystem::path objective_history;
  std::filesystem::path summary;
};

// Construct the ANO-preorthogonalized meta-Lowdin AO basis used by the
// supplied Python PM workflow. The reference overlap must be
// <target AO|ANO-RCC AO> on the same H/O geometry and spherical convention.
[[nodiscard]] linalg::Matrix build_meta_lowdin_orthogonal_ao(
    const molecule::Molecule& molecule,
    const basis::BasisSet& target_basis,
    const linalg::Matrix& target_overlap,
    const basis::BasisSet& ano_reference_basis,
    const linalg::Matrix& target_ano_overlap);

[[nodiscard]] PipekMezeyResult localize_pipek_mezey(
    const molecule::Molecule& molecule, const basis::BasisSet& basis,
    const linalg::Matrix& overlap,
    const linalg::Matrix& canonical_coefficients,
    const std::vector<std::size_t>& orbital_indices,
    const PipekMezeyOptions& options = {});

// Write a deterministic, zero-based diagnostic record for comparison with an
// independent localization implementation. All target files are checked
// before any file is opened, and existing files are rejected unless overwrite
// is explicitly requested. The supplied orbital indices identify the columns
// of the canonical coefficient matrix used to form this localized space.
[[nodiscard]] PipekMezeyDiagnosticPaths write_pipek_mezey_diagnostics(
    const PipekMezeyResult& result,
    const molecule::Molecule& molecule,
    const std::vector<std::size_t>& orbital_indices,
    const std::filesystem::path& prefix, bool overwrite = false);

}  // namespace modernqc::localization
