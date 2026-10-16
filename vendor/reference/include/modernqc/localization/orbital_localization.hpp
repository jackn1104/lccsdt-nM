#pragma once

#include "modernqc/linalg/matrix.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace modernqc::localization {

enum class LocalizationMethod {
  pipek_mezey,
  fourth_moment,
};

// Cartesian AO position-moment integrals in row-major Cartesian index order.
// Matrix elements are <mu|r_a ... r_d|nu>, with a,b,c,d in {x,y,z}.
struct CartesianMomentIntegrals {
  std::array<linalg::Matrix, 3> first;
  std::array<linalg::Matrix, 9> second;
  std::array<linalg::Matrix, 27> third;
  std::array<linalg::Matrix, 81> fourth;
};

struct OrbitalSpatialDiagnostic {
  std::size_t orbital;
  std::array<double, 3> centroid_bohr;
  double omega2_bohr2;
  double sigma2_bohr;
  double omega4_bohr4;
  double sigma4_bohr;
};

// Backend-neutral distributed-memory boundary. Each rank owns the cyclic
// orbital columns p = rank, rank + ranks, ... . The callback must replace the
// supplied buffer by its elementwise sum over exactly these ranks. It is
// called only by the calling thread, never inside an OpenMP region.
struct LocalizationParallelOptions {
  std::size_t rank{0};
  std::size_t ranks{1};
  std::function<void(std::vector<double>&)> sum_in_place;
  // Replace the supplied buffer on every rank by rank zero's buffer. PM/CIAH
  // uses this to avoid repeating global dense rotation work on every rank.
  std::function<void(std::vector<double>&)> broadcast_from_root;
};

// Complete state needed to resume the outer trust-region iterations. The
// objective gradient is deliberately recomputed from the saved rotation so a
// checkpoint never depends on an internal matrix-cache representation.
struct LocalizationRestartState {
  linalg::Matrix rotation;
  double trust_radius{0.25};
  int completed_iterations{0};
  int accepted_steps{0};
  int rejected_steps{0};
  std::vector<double> objective_history;
};

struct LocalizationIteration {
  int iteration{0};
  int accepted_steps{0};
  int rejected_steps{0};
  bool step_accepted{false};
  bool converged{false};
  double objective{0.0};
  double gradient_norm{0.0};
  double trust_radius{0.0};
  std::string convergence_reason;
};

using LocalizationProgressCallback = std::function<void(
    const LocalizationIteration&, const linalg::Matrix&,
    const std::vector<double>&)>;

struct TrustRegionLocalizationOptions {
  int maximum_iterations{200};
  double gradient_tolerance{1.0e-8};
  double objective_tolerance{1.0e-12};
  double initial_trust_radius{0.25};
  double maximum_trust_radius{2.0};
  double acceptance_threshold{0.1};
  double minimum_trust_radius{1.0e-12};
  double finite_difference_step{1.0e-6};
  double gradient_check_tolerance{5.0e-6};
  int gradient_check_directions{3};
  std::uint64_t gradient_check_seed{20260730};
  double invariant_tolerance{2.0e-10};
  // Empty means identity. A supplied matrix restarts from that rotation.
  linalg::Matrix initial_rotation;
  // Use restart_state, rather than initial_rotation, when continuing an
  // interrupted optimizer. Supplying both is rejected.
  std::optional<LocalizationRestartState> restart_state;
  LocalizationParallelOptions parallel;
  LocalizationProgressCallback progress;
};

struct LocalizationResult {
  linalg::Matrix coefficients;
  linalg::Matrix rotation;
  LocalizationMethod method;
  int fm_power;
  double objective_initial;
  double objective_final;
  double absolute_objective_reduction;
  double relative_objective_reduction;
  double gradient_norm_initial;
  double gradient_norm_final;
  int iterations;
  int accepted_steps;
  int rejected_steps;
  bool converged;
  std::string convergence_reason;
  double final_trust_radius;
  double rotation_orthogonality_maximum_error;
  double orbital_orthonormality_maximum_error;
  double projector_maximum_error;
  double gradient_check_maximum_relative_error;
  std::vector<double> objective_history;
  std::vector<OrbitalSpatialDiagnostic> diagnostics;
};

// Backend-independent Löwdin-population PM localization. AO center lists must
// form a disjoint, exhaustive partition of [0, nao).
[[nodiscard]] LocalizationResult localize_pipek_mezey_trust_region(
    const linalg::Matrix& canonical_coefficients,
    const linalg::Matrix& overlap,
    const std::vector<std::vector<std::size_t>>& center_ao_indices,
    const TrustRegionLocalizationOptions& options = {});

// Rows are AO centers and columns are orbitals. Each column sums to one for
// an AO-metric orthonormal orbital set. This is the same Lowdin population
// definition used by the trust-region Pipek--Mezey objective.
[[nodiscard]] linalg::Matrix lowdin_center_populations(
    const linalg::Matrix& coefficients, const linalg::Matrix& overlap,
    const std::vector<std::vector<std::size_t>>& center_ao_indices);

// Backend-independent fourth-central-moment localization. power selects FM1
// through FM4 and must be in [1,4].
[[nodiscard]] LocalizationResult localize_fourth_moment(
    const linalg::Matrix& canonical_coefficients,
    const linalg::Matrix& overlap,
    const CartesianMomentIntegrals& moments, int power,
    const TrustRegionLocalizationOptions& options = {});

// Generic dispatcher. Pipek-Mezey is the default method. The unused data
// pointer for the selected method may be null.
[[nodiscard]] LocalizationResult localize_orbitals(
    const linalg::Matrix& canonical_coefficients,
    const linalg::Matrix& overlap,
    LocalizationMethod method = LocalizationMethod::pipek_mezey,
    const std::vector<std::vector<std::size_t>>& center_ao_indices = {},
    const CartesianMomentIntegrals* moments = nullptr, int fm_power = 2,
    const TrustRegionLocalizationOptions& options = {});

// Convenience overload retaining Pipek-Mezey as the default behavior.
[[nodiscard]] LocalizationResult localize_orbitals(
    const linalg::Matrix& canonical_coefficients,
    const linalg::Matrix& overlap,
    const std::vector<std::vector<std::size_t>>& center_ao_indices,
    const TrustRegionLocalizationOptions& options = {});

[[nodiscard]] std::vector<OrbitalSpatialDiagnostic>
evaluate_spatial_diagnostics(
    const linalg::Matrix& coefficients,
    const CartesianMomentIntegrals& moments);

// Writes <prefix>_C_localized.npy, <prefix>_U_localization.npy,
// <prefix>_orbital_diagnostics.csv, and
// <prefix>_localization_summary.json. Existing files are rejected unless
// overwrite is explicitly true.
void save_localization_outputs(const LocalizationResult& result,
                               const std::filesystem::path& prefix,
                               bool overwrite = false);

}  // namespace modernqc::localization
