#pragma once

#include "localized_cc/ccsd.hpp"
#include "localized_cc/frozen_core.hpp"
#include "localized_cc/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace localized_cc {

// Optional immutable AO-to-MO sidecar. The contract includes numerical
// settings; exact AO coefficient and RHF Fock equality prevents gauge reuse.
void write_hamiltonian_checkpoint(const std::filesystem::path& path,
    const std::string& contract, const Matrix& occupied, const Matrix& virtuals,
    const Matrix& rhf_fock, const SpatialRhfHamiltonian& h);
SpatialRhfHamiltonian read_hamiltonian_checkpoint(const std::filesystem::path& path,
    const std::string& contract, const Matrix& occupied, const Matrix& virtuals,
    const Matrix& rhf_fock, std::size_t maximum_bytes);

// The checkpoint is deliberately backend-independent.  Matrices use the
// same row-major Tensor storage as the CC code, and the molecular driver
// converts ModernQC matrices at its boundary.
struct RhfCheckpointState {
  double nuclear_repulsion_energy{};
  double electronic_energy{};
  double total_energy{};
  Matrix overlap;
  Matrix core_hamiltonian;
  Matrix density;
  Matrix fock;
  Matrix coefficients;
  std::vector<double> orbital_energies;
  std::vector<double> occupations;
};

struct LocalizationCheckpointState {
  std::string method; // "pm", "fm2", or "fm4"
  Matrix occupied_coefficients;
  Matrix virtual_coefficients;
  Matrix occupied_rotation;
  Matrix virtual_rotation;
  std::vector<int> occupied_monomers;
  std::vector<int> virtual_monomers;
  std::vector<double> occupied_confidence;
  std::vector<double> virtual_confidence;
  double assignment_threshold{0.8};
};

// Restart for one independently optimized orbital space. A converged state
// may be reused while the other space is still in progress. Recomputing the
// objective gradient from rotation keeps this record independent of ModernQC
// internal caches.
struct LocalizationOptimizationCheckpointState {
  std::string method; // "pm", "fm2", or "fm4"
  std::string space;  // "occupied" or "virtual"
  bool converged{};
  Matrix rotation;
  double trust_radius{};
  std::size_t completed_iterations{};
  std::size_t accepted_steps{};
  std::size_t rejected_steps{};
  std::vector<double> objective_history;
};

struct CcsdCheckpointState {
  bool localized{};
  bool converged{};
  std::size_t completed_iterations{};
  double correlation_energy{};
  double r1_norm{};
  double r2_norm{};
  Matrix t1;
  Tensor4 t2;
  std::vector<CcsdIteration> history;
};

// Backend-independent restart for the pre-construction selected CCSD
// executor. T2 is block-major in the deterministic manifold-plan order.
// Labels, maximum span, and block sizes are the structural restore contract;
// omitted amplitudes are structural zeros and are never serialized densely.
struct SparseCcsdCheckpointState {
  bool converged{};
  std::size_t completed_iterations{};
  std::size_t maximum_monomers{};
  double correlation_energy{};
  double r1_norm{};
  double r2_norm{};
  Matrix t1;
  std::vector<int> occupied_monomers;
  std::vector<int> virtual_monomers;
  std::vector<std::uint64_t> t2_block_sizes;
  std::vector<double> t2_block_values;
  std::vector<CcsdIteration> history;
};

struct CanonicalTriplesCheckpointState {
  std::vector<double> bracket_task_energies;
  std::vector<double> parenthetical_task_energies;
  std::vector<std::uint8_t> completed_tasks;
};

struct LaplaceTriplesCheckpointState {
  std::size_t quadrature_points{};
  // Point-major M1--M6 contributions and one binary completion flag per
  // quadrature point. The converged CCSD amplitudes remain in either the
  // dense localized_ccsd state or the block-native localized_sparse_ccsd
  // state selected by the calculation fingerprint.
  std::vector<double> point_many_body_energies;
  std::vector<std::uint8_t> completed_points;
  std::string quadrature_scheme{"minimax"};
};

struct LocalizedTriplesCheckpointState {
  bool converged{};
  std::size_t completed_iterations{};
  double energy{};
  double residual_norm{};
  Tensor6 response;
  std::vector<double> residual_history;
};

// Provenance and numerical checks for a same-geometry small-to-large basis
// projection. Triples are intentionally never projected.
struct BasisProjectionCheckpointState {
  std::string source_checkpoint;
  std::string source_fingerprint;
  std::string source_basis;
  std::string target_basis;
  std::size_t source_ao_count{};
  std::size_t target_ao_count{};
  std::size_t target_metric_rank{};
  double minimum_projection_gram_eigenvalue{};
  double occupied_orthonormality_error{};
  double spin_summed_electron_count{};
  bool canonical_amplitudes_projected{};
  bool localization_seed_projected{};
  bool localized_amplitudes_projected{};
};

struct CalculationCheckpoint {
  std::string fingerprint;
  std::optional<Matrix> rhf_restart_density;
  std::optional<RhfCheckpointState> rhf;
  std::optional<LocalizationCheckpointState> localization;
  std::optional<LocalizationOptimizationCheckpointState>
      occupied_localization_optimization;
  std::optional<LocalizationOptimizationCheckpointState>
      virtual_localization_optimization;
  std::optional<CcsdCheckpointState> canonical_ccsd;
  std::optional<CanonicalTriplesCheckpointState> canonical_triples;
  std::optional<CcsdCheckpointState> localized_ccsd;
  std::optional<SparseCcsdCheckpointState> localized_sparse_ccsd;
  std::optional<LocalizedTriplesCheckpointState> localized_triples;
  std::optional<BasisProjectionCheckpointState> basis_projection;
  // Task restart for the exact semicanonical streamed total evaluated from
  // converged localized amplitudes. Kept distinct from canonical validation
  // tasks and from the retained dense noncanonical reference response.
  std::optional<CanonicalTriplesCheckpointState> localized_streamed_triples;
  std::optional<LaplaceTriplesCheckpointState> localized_laplace_triples;
};

// Restore a transactionally saved converged CCSD state without re-entering
// the fixed-point solver. The saved energy change, residuals, tensor
// dimensions, finite values, and T2 antisymmetry are revalidated against the
// current run contract before the amplitudes are returned for (T).
class CcsdCheckpointConvergenceError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
// Same validation without copying amplitudes. Only a valid checkpoint that
// fails the requested convergence gate raises CcsdCheckpointConvergenceError.
void validate_converged_ccsd_checkpoint(
    const CcsdCheckpointState& state, bool expected_localized,
    std::size_t occupied, std::size_t virtuals, double energy_tolerance,
    double residual_tolerance, CcsdResidualMetric residual_metric);

[[nodiscard]] CcsdResult restore_converged_ccsd(
    const CcsdCheckpointState& state, bool expected_localized,
    std::size_t occupied, std::size_t virtuals, double energy_tolerance,
    double residual_tolerance, CcsdResidualMetric residual_metric,
    CcsdAmplitudeBackend amplitude_backend, CcsdWabefBackend wabef_backend);

// Replaces path only after a complete temporary file has been flushed,
// closed, read back, and fingerprint-validated.
void write_checkpoint_transactional(const std::filesystem::path& path,
                                    const CalculationCheckpoint& checkpoint);

[[nodiscard]] CalculationCheckpoint read_checkpoint(
    const std::filesystem::path& path,
    const std::string& expected_fingerprint);

} // namespace localized_cc
