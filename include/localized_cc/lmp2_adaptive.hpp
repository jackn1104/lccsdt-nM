#pragma once
#include "localized_cc/lmp2_preintegral.hpp"
#include <memory>

namespace localized_cc {
using Mp2TileKey = std::array<std::size_t,4>;
// Exchange-related tiles share an owner, so energy contractions stay local.
std::size_t mp2_tile_owner(Mp2TileKey key, std::size_t ranks);
struct Mp2AdaptiveTile {
  Mp2TileKey key{};
  std::array<std::vector<std::size_t>,4> indices;
  std::size_t span{};
  double source_bound{};
  std::vector<double> integrals, amplitudes;
};
struct Mp2AdaptiveOptions {
  double accuracy{1e-7};
  std::size_t occupied_tile{4}, virtual_tile{12};
  std::size_t maximum_bytes{1024ULL*1024*1024}, max_iterations{500};
  bool full_reference{false};
  bool efficient_planning{false}; // preflight domains + inexact preliminary solves
};
struct Mp2AdaptiveResult {
  std::vector<Mp2AdaptiveTile> tiles; // selected blocks only, never global T2
  std::array<double,4> energy{};
  double gap{}, full_source_bound{}, integral_error{}, residual_bound{}, energy_error{};
  std::size_t full_elements{}, active_elements{}, rounds{}, iterations{}, workspace_bytes{};
  std::size_t local_elements{}, halo_peak_elements{}, halo_transfers{};
  std::size_t preliminary_iterations{}, initial_elements{};
  bool full_domain_fallback{};
  double final_active_residual{}, strict_residual_target{};
};
// Replicated metadata, owner-only numerical payloads. Fill only this rank's
// integrals; other payloads must stay empty. Return a global Frobenius error
// bound. Collective, including ranks owning no tiles.
using Mp2TileBuilder = std::function<double(std::vector<Mp2AdaptiveTile>&,double)>;
Mp2AdaptiveResult solve_adaptive_mp2(const Matrix& fo,const Matrix& fv,
    const Matrix& pair_bounds,const std::vector<int>& ol,const std::vector<int>& vl,
    const Mp2AdaptiveOptions& options,const Mp2TileBuilder& build,
    const std::function<void(const std::string&)>& progress = {});

struct Mp2AoStatistics {
  double quartets_considered{}, quartets_evaluated{}, quartets_skipped{}, tile_quartets_skipped{};
  double tree_nodes_visited{}, quartet_leaves_visited{}, pruned_subtrees{};
  double requested_tiles{}, contracted_tiles{}, pair_exchange_reused_tiles{};
  double engine_evaluations{}, quartet_cache_hits{}, quartet_cache_peak_bytes{};
};
// Four-center shell engine remains private to the implementation.
class Mp2DomainAoProvider {
 public:
  Mp2DomainAoProvider(const modernqc::molecule::Molecule&,
      const modernqc::integrals::Libint2IntegralProvider&,
      const modernqc::linalg::Matrix&,const modernqc::linalg::Matrix&,
      std::size_t maximum_bytes,std::size_t threads,bool pair_exchange_reuse=false,
      std::size_t quartet_cache_bytes=0);
  ~Mp2DomainAoProvider();
  Matrix pair_bounds() const;
  double build(std::vector<Mp2AdaptiveTile>&,double norm_budget,
      const std::function<void(const std::string&)>& progress = {});
  Mp2AoStatistics statistics() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
void run_lmp2_adaptive_pilot(const modernqc::molecule::Molecule&,
    const modernqc::integrals::Libint2IntegralProvider&,
    const modernqc::linalg::Matrix&,const modernqc::linalg::Matrix&,
    const modernqc::linalg::Matrix&,const std::vector<int>&,const std::vector<int>&,
    const std::filesystem::path&,const std::string&,std::size_t,std::size_t,bool,
    const std::function<void(const std::string&)>&);
} // namespace localized_cc
