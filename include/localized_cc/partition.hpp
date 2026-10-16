#pragma once

#include "localized_cc/frozen_core.hpp"
#include "localized_cc/tensor.hpp"
#include <array>
#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace localized_cc {
class SelectedDoublesView;
class SelectedIntegralStore;
struct PartitionScreeningOptions {
  // This is an exact post-contraction contribution screen, not an AO Schwarz
  // screen. Contributions with |e| <= threshold are omitted from the reported
  // classes, and the calculation is accepted only when their accumulated
  // absolute bound remains at or below max_omitted_absolute.
  double contribution_threshold{};
  double max_omitted_absolute{1.0e-6};
};

struct EnergyPartition {
  // Detailed keys retain the participating monomers, hole/particle labels,
  // charge-transfer character, and exact additive contribution.
  std::map<std::string,double> classes;
  std::map<std::string,std::size_t> class_term_counts;
  // Coarse additive totals M1, M2, ... by the number of distinct monomers.
  std::map<std::string,double> many_body_classes;
  std::map<std::string,std::size_t> many_body_term_counts;
  // sum is the energy in fully assigned M classes. Parent reconstruction is
  // sum + unassigned_energy + omitted_energy.
  double singles_fia_t1{}, sum{}, parent_energy{}, sum_error{};
  double unassigned_energy{}, omitted_energy{}, omitted_absolute_bound{};
  double contribution_threshold{}, max_omitted_absolute{};
  std::size_t assigned_terms{}, unassigned_terms{}, retained_terms{};
  std::size_t omitted_terms{};
};

struct CcsdEnergyTermAuditRow {
  std::string criterion;
  double tolerance_hartree{};
  double maximum_omitted_term_hartree{};
  double omitted_signed_energy_hartree{};
  double omitted_absolute_bound_hartree{};
  std::size_t omitted_terms{};
  std::array<std::size_t, 4> omitted_many_body_terms{};
};

struct CcsdEnergyTermAudit {
  double tolerance_hartree{};
  double singles_energy_hartree{};
  double doubles_energy_hartree{};
  std::size_t total_doubles_terms{};
  std::size_t exact_zero_terms{};
  std::vector<CcsdEnergyTermAuditRow> rows;
};

// Post-convergence diagnostic for the spin-orbital CCSD doubles-energy terms
//   E_ijab = 1/4 <ij||ab> t_ij^ab
//          + 1/2 <ij||ab> t_i^a t_j^b.
// It does not screen CCSD iterations. The absolute criterion is certified by
// the triangle inequality; the signed criterion is empirical and may rely on
// cancellation. Rows are evaluated only at complete equal-|E_ijab| groups so
// maximum_omitted_term_hartree defines a reproducible scalar threshold.
CcsdEnergyTermAudit audit_ccsd_energy_terms(
    const SpatialRhfHamiltonian& hamiltonian, const Matrix& t1,
    const Tensor4& t2, const std::vector<int>& occupied_labels,
    const std::vector<int>& virtual_labels, double tolerance_hartree);

void write_ccsd_energy_term_audit_csv(
    const std::filesystem::path& path,
    const CcsdEnergyTermAudit& audit);

EnergyPartition partition_ccsd(const Matrix& fock,const Tensor4& eri,const Matrix& t1,const Tensor4& t2,std::size_t nocc,const std::vector<int>& occupied_labels,const std::vector<int>& virtual_labels,const PartitionScreeningOptions& screening = {});
EnergyPartition partition_ccsd(const Matrix& fock,
                               const SelectedIntegralStore& eri,
                               const Matrix& t1, const Tensor4& t2,
                               std::size_t spatial_nocc,
                               const std::vector<int>& occupied_labels,
                               const std::vector<int>& virtual_labels,
                               const PartitionScreeningOptions& screening = {});
EnergyPartition partition_ccsd(const SpatialRhfHamiltonian& hamiltonian,
                               const Matrix& t1, const Tensor4& t2,
                               const std::vector<int>& occupied_labels,
                               const std::vector<int>& virtual_labels,
                               const PartitionScreeningOptions& screening = {});
EnergyPartition partition_ccsd(const SpatialRhfHamiltonian& hamiltonian,
                               const Matrix& t1,
                               const SelectedDoublesView& t2,
                               const std::vector<int>& occupied_labels,
                               const std::vector<int>& virtual_labels,
                               const PartitionScreeningOptions& screening = {});
EnergyPartition partition_triples(const Tensor6& energy_contribution,const std::vector<int>& occupied_labels,const std::vector<int>& virtual_labels,const PartitionScreeningOptions& screening = {});
// Exact scalar shortcut when every occupied and virtual orbital belongs to one
// confidently assigned monomer. This is sufficient for W1 and deliberately
// rejects contribution screening because individual six-index terms are not
// available at this boundary.
EnergyPartition partition_triples_single_monomer(
    double parent_energy, const std::vector<int>& occupied_labels,
    const std::vector<int>& virtual_labels,
    const PartitionScreeningOptions& screening = {});
// Construct the additive M1--M6 report from a streamed orbital-invariant
// triples contraction. Term-level screening must already have been applied by
// the producer; this boundary therefore accepts only a zero report threshold.
EnergyPartition partition_triples_many_body_sums(
    double parent_energy, const std::vector<double>& many_body_energies,
    const std::vector<std::size_t>& many_body_terms,
    const PartitionScreeningOptions& screening = {});
std::vector<int> expand_spatial_labels(const std::vector<int>& labels);

void write_partition_csv(const std::filesystem::path& path,
                         const EnergyPartition& ccsd,
                         const EnergyPartition& triples);
void write_orbital_assignments_csv(
    const std::filesystem::path& path, const std::string& localizer,
    double confidence_threshold, const std::vector<int>& occupied_labels,
    const std::vector<double>& occupied_confidence,
    const std::vector<int>& virtual_labels,
    const std::vector<double>& virtual_confidence);
} // namespace localized_cc
