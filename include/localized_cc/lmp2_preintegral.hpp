#pragma once
#include "localized_cc/lmp2_domains.hpp"
#include "modernqc/integrals/integral_provider.hpp"
#include "modernqc/molecule/molecule.hpp"
#include <map>

namespace localized_cc {
struct Lmp2PreintegralBlock {
  std::array<int,4> labels{}; // independently sorted hole and particle labels
  std::size_t elements{}, monomer_span{};
  double source_norm_bound{};
  bool retained{true};
};
struct Lmp2PreintegralPlan {
  std::vector<int> occupied_labels, virtual_labels;
  std::map<std::array<int,4>,Lmp2PreintegralBlock> blocks;
  double full_source_norm_bound{}, omitted_source_norm_bound{};
  double minimum_denominator{}, energy_error_bound{}, budget{};
  std::size_t full_elements{}, retained_elements{};
};

// Pair bounds q(i,a)>=sqrt((ia|ia)); full coefficient tails retained.
// Selection happens without an OVOV integral tensor or MP2 amplitudes.
Lmp2PreintegralPlan plan_lmp2_preintegral_domains(
    const Matrix& pair_bounds, const std::vector<int>& occupied_labels,
    const std::vector<int>& virtual_labels, double minimum_denominator,
    double budget_hartree);

// Libint2 stays in the implementation. Only shell-pair self quartets are
// needed. The provider must remain alive (owns Libint runtime/basis contract).
Matrix lmp2_ao_schwarz_pair_bounds(
    const modernqc::molecule::Molecule& molecule,
    const modernqc::integrals::Libint2IntegralProvider& provider,
    const modernqc::linalg::Matrix& occupied,
    const modernqc::linalg::Matrix& virtuals);

struct Lmp2OvovTile {
  std::array<int,4> labels{};
  std::array<std::vector<std::size_t>,4> indices;
  std::size_t offset{};
};
struct Lmp2SelectedOvov {
  Lmp2PreintegralPlan plan;
  std::vector<Lmp2OvovTile> tiles;
  std::map<std::array<int,4>,std::size_t> lookup;
  std::vector<std::size_t> occupied_local,virtual_local;
  std::vector<double> values;
  double operator()(std::size_t i,std::size_t j,std::size_t a,std::size_t b) const;
};
Lmp2SelectedOvov allocate_lmp2_selected_ovov(
    const Lmp2PreintegralPlan& plan,std::size_t maximum_bytes);
// Testable completion of the SAME lambda-blocked first-index interface used
// by ModernQC's LMP2. Three BLAS transformations complete retained tiles only.
void complete_lmp2_selected_ao_block(
    const modernqc::integrals::BatchedFirstIndexAoBlock& block,
    const modernqc::linalg::Matrix& occupied,
    const modernqc::linalg::Matrix& virtuals,
    const std::vector<std::size_t>& owned_occupied,
    Lmp2SelectedOvov& output,std::size_t maximum_workspace_bytes);
Lmp2SelectedOvov build_lmp2_selected_ovov(
    const modernqc::integrals::Libint2IntegralProvider& provider,
    const modernqc::linalg::Matrix& occupied,
    const modernqc::linalg::Matrix& virtuals,
    const Lmp2PreintegralPlan& plan,std::size_t maximum_bytes,
    std::size_t threads,const std::function<void(const std::string&)>& progress);
void write_lmp2_preintegral_plan(const std::filesystem::path& directory,
                               const Lmp2PreintegralPlan& plan);
// Collective domain-selection pilot; does not construct a full Hamiltonian.
void run_lmp2_preintegral_pilot(
    const modernqc::molecule::Molecule& molecule,
    const modernqc::integrals::Libint2IntegralProvider& provider,
    const modernqc::linalg::Matrix& occupied,
    const modernqc::linalg::Matrix& virtuals,
    const modernqc::linalg::Matrix& ao_fock,
    const std::vector<int>& occupied_labels,const std::vector<int>& virtual_labels,
    const std::filesystem::path& checkpoint,const std::string& fingerprint,
    double budget,std::size_t workspace_bytes,std::size_t threads,
    const std::function<void(const std::string&)>& progress);
} // namespace localized_cc
