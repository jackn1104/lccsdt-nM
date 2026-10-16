#include "localized_cc/lmp2_adaptive.hpp"
#include "localized_cc/parallel.hpp"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <stdexcept>

namespace localized_cc {
void run_lmp2_adaptive_pilot(const modernqc::molecule::Molecule& molecule,
    const modernqc::integrals::Libint2IntegralProvider& integrals,
    const modernqc::linalg::Matrix& co,const modernqc::linalg::Matrix& cv,
    const modernqc::linalg::Matrix& fock,const std::vector<int>& ol,const std::vector<int>& vl,
    const std::filesystem::path& checkpoint,const std::string& fingerprint,
    std::size_t cap,std::size_t threads,bool full,
    const std::function<void(const std::string&)>& progress) {
  const auto world=parallel::world();
  const auto start=std::chrono::steady_clock::now();
  const std::filesystem::path directory=checkpoint.string()+".adaptive-mp2";
  std::vector<int> status{0};
  if(world.rank==0) {if(!std::filesystem::create_directory(directory)) status[0]=1;}
  parallel::broadcast_in_place(status);if(status[0]) throw std::runtime_error("adaptive MP2 output exists; use a fresh checkpoint copy");
  Matrix fo({co.columns(),co.columns()}),fv({cv.columns(),cv.columns()});
  if(world.rank==0) {
    auto a=modernqc::linalg::multiply(modernqc::linalg::transpose(co),modernqc::linalg::multiply(fock,co));
    auto b=modernqc::linalg::multiply(modernqc::linalg::transpose(cv),modernqc::linalg::multiply(fock,cv));
    for(std::size_t i=0;i<co.columns();++i) for(std::size_t j=0;j<co.columns();++j) fo(i,j)=a(i,j);
    for(std::size_t i=0;i<cv.columns();++i) for(std::size_t j=0;j<cv.columns();++j) fv(i,j)=b(i,j);
  }
  parallel::broadcast_in_place(fo.data());parallel::broadcast_in_place(fv.data());
  if(world.rank==0) progress("adaptive MP2 AO support/Schwarz plan started; no four-index AO or MO tensor; no density fitting");
  const char* reuse_env=std::getenv("CCSD_LMP2_AO_PAIR_REUSE");
  const std::string reuse_mode=reuse_env?reuse_env:"0";
  if(reuse_mode!="0"&&reuse_mode!="1") throw std::invalid_argument("CCSD_LMP2_AO_PAIR_REUSE must be 0 or 1");
  const bool pair_reuse=reuse_mode=="1";
  const char* efficient_env=std::getenv("CCSD_LMP2_EFFICIENT_PLANNING");
  const std::string efficient_mode=efficient_env?efficient_env:"0";
  if(efficient_mode!="0"&&efficient_mode!="1") throw std::invalid_argument("CCSD_LMP2_EFFICIENT_PLANNING must be 0 or 1");
  const bool efficient=efficient_mode=="1";
  // Small, bounded AO cache, never a full AO tensor. Charged to AO workspace.
  const std::size_t cache_bytes=efficient?std::min(std::size_t{32}*1024*1024,cap/16):0;
  Mp2DomainAoProvider provider(molecule,integrals,co,cv,cap/2,threads,pair_reuse,cache_bytes);
  if(world.rank==0) progress("adaptive MP2 exact Coulomb pair-exchange reuse="+reuse_mode+"; no individual occupied/virtual exchange approximation");
  auto q=provider.pair_bounds();parallel::broadcast_in_place(q.data());
  Mp2AdaptiveOptions options;options.maximum_bytes=cap/2;options.full_reference=full;options.efficient_planning=efficient;
  std::size_t round=0;
  if(world.rank==0) progress("adaptive MP2 selected-tile solve started; full PM classes; 1e-7 Eh target; owner-distributed selected vectors; bounded Fock halo");
  const auto result=solve_adaptive_mp2(fo,fv,q,ol,vl,options,
      [&](auto& tiles,double budget) {
        ++round;
        if(world.rank==0) {
          std::ofstream plan(directory/("round_"+std::to_string(round)+"_plan.csv"));
          plan.exceptions(std::ios::failbit|std::ios::badbit);
          plan<<std::setprecision(17)<<"i_tile,j_tile,a_tile,b_tile,nM,source_norm_bound,AO_batch_norm_budget,owner\n";
          for(const auto& tile:tiles) plan<<tile.key[0]<<','<<tile.key[1]<<','<<tile.key[2]<<','<<tile.key[3]<<','<<tile.span<<','<<tile.source_bound<<','<<budget<<','<<mp2_tile_owner(tile.key,world.size)<<'\n';
          plan.close();
        }
        if(world.rank==0) progress("domain AO construction started; new_tiles="+std::to_string(tiles.size()));
        const double bound=provider.build(tiles,budget,progress);
        if(world.rank==0) progress("domain AO construction complete; cumulative logical_evaluated="+std::to_string(provider.statistics().quartets_evaluated)+" skipped="+std::to_string(provider.statistics().quartets_skipped)+" engine_calls="+std::to_string(provider.statistics().engine_evaluations)+" cache_hits="+std::to_string(provider.statistics().quartet_cache_hits));
        return bound;
      },[&](const auto& message) {if(world.rank==0) progress(message);});
  // Scalar rank ledger, no selected integral/amplitude gather.
  std::vector<double> ownership(2*world.size,0.);
  ownership[2*world.rank]=static_cast<double>(result.local_elements);
  ownership[2*world.rank+1]=static_cast<double>(result.workspace_bytes);
  parallel::sum_in_place(ownership);
  if(world.rank==0) {
    auto output=[&](const char* name) {std::ofstream f(directory/name);f.exceptions(std::ios::failbit|std::ios::badbit);f<<std::setprecision(17);return f;};
    auto energy=output("energy.csv");energy<<"nM,mp2_hartree\n";
    for(std::size_t n=0;n<4;++n) {energy<<n+1<<','<<result.energy[n]<<'\n';progress("adaptive MP2 M"+std::to_string(n+1)+"="+std::to_string(result.energy[n])+" Eh");}energy.close();
    auto labels=output("orbital_labels.csv");labels<<"space,index,monomer\n";
    for(std::size_t i=0;i<ol.size();++i) labels<<"occupied,"<<i<<','<<ol[i]<<'\n';
    for(std::size_t a=0;a<vl.size();++a) labels<<"virtual,"<<a<<','<<vl[a]<<'\n';labels.close();
    auto ranks=output("ownership.csv");ranks<<"rank,elements,solver_planned_bytes\n";
    for(std::size_t rank=0;rank<world.size;++rank) ranks<<rank<<','<<ownership[2*rank]<<','<<ownership[2*rank+1]<<'\n';
    ranks.close();
    auto blocks=output("blocks.csv");blocks<<"i_tile,j_tile,a_tile,b_tile,nM,elements,owner\n";
    for(const auto& b:result.tiles) {
      std::size_t count=1;for(const auto& ids:b.indices) count*=ids.size();
      blocks<<b.key[0]<<','<<b.key[1]<<','<<b.key[2]<<','<<b.key[3]<<','<<b.span<<','<<count<<','<<mp2_tile_owner(b.key,world.size)<<'\n';
    }
    blocks.close();
    std::map<std::pair<int,std::size_t>,std::vector<std::size_t>> tile_indices;
    for(const auto& b:result.tiles) for(std::size_t axis=0;axis<4;++axis)
      tile_indices.emplace(std::make_pair(axis<2?0:1,b.key[axis]),b.indices[axis]);
    auto mapping=output("orbital_tiles.csv");mapping<<"space,tile,local_index,orbital_index,monomer\n";
    for(const auto& [key,ids]:tile_indices) for(std::size_t k=0;k<ids.size();++k)
      mapping<<(key.first==0?"occupied":"virtual")<<','<<key.second<<','<<k<<','<<ids[k]<<','<<(key.first==0?ol.at(ids[k]):vl.at(ids[k]))<<'\n';
    mapping.close();
    const auto stat=provider.statistics();auto meta=output("metadata.txt");
    meta<<"provenance="<<fingerprint<<"\naccuracy_hartree=1e-7\nno_density_fitting=1\nfull_nM=1\nccsd_t_accuracy_validated=0\n"
        <<"full_reference="<<int(full)<<"\nfull_elements="<<result.full_elements<<"\nactive_elements="<<result.active_elements
        <<"\nenergy_error_bound_hartree="<<result.energy_error<<"\nfull_residual_bound="<<result.residual_bound
        <<"\nintegral_error_norm="<<result.integral_error<<"\nminimum_denominator="<<result.gap
        <<"\nfull_source_norm_bound="<<result.full_source_bound<<"\nrounds="<<result.rounds<<"\niterations="<<result.iterations
        <<"\nsolver_planned_bytes_per_rank="<<result.workspace_bytes<<"\nAO_quartets_considered="<<stat.quartets_considered
        <<"\nAO_quartets_evaluated="<<stat.quartets_evaluated<<"\nAO_quartets_skipped="<<stat.quartets_skipped
        <<"\nAO_tile_quartets_skipped="<<stat.tile_quartets_skipped<<"\nglobal_dense_amplitude_tensor=0\nfull_AO_ERI_tensor=0\nfull_MO_ERI_tensor=0\n"
        <<"selected_vectors_replicated=0\nplanner_enumerates_tile_combinations=1\nlinear_scaling_demonstrated=0\n"
        <<"AO_flat_quartet_traversal=0\nAO_tree_nodes_visited="<<stat.tree_nodes_visited
        <<"\nAO_quartet_leaves_visited="<<stat.quartet_leaves_visited<<"\nAO_pruned_subtrees="<<stat.pruned_subtrees
        <<"\nAO_pair_exchange_reuse="<<int(pair_reuse)<<"\nAO_requested_tiles="<<stat.requested_tiles
        <<"\nAO_contracted_tiles="<<stat.contracted_tiles<<"\nAO_pair_exchange_reused_tiles="<<stat.pair_exchange_reused_tiles
        <<"\nefficient_planning="<<int(efficient)<<"\ninitial_elements="<<result.initial_elements
        <<"\nfull_domain_scheduling_fallback="<<int(result.full_domain_fallback)
        <<"\npreliminary_iterations="<<result.preliminary_iterations
        <<"\nfinal_active_residual="<<result.final_active_residual<<"\nstrict_residual_target="<<result.strict_residual_target
        <<"\nAO_engine_evaluations="<<stat.engine_evaluations<<"\nAO_quartet_cache_hits="<<stat.quartet_cache_hits
        <<"\nAO_quartet_cache_cap_bytes="<<cache_bytes<<"\nAO_quartet_cache_peak_bytes="<<stat.quartet_cache_peak_bytes
        <<"\nmpi_ranks="<<world.size<<"\nhalo_peak_elements="<<result.halo_peak_elements
        <<"\nhalo_transfers="<<result.halo_transfers<<"\nseconds="
        <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<'\n';meta.close();
    auto complete=output("COMPLETE");complete<<"Adaptive MP2 only; CCSD(T) accuracy and linear scaling not validated\n";complete.close();
    progress("adaptive MP2 complete; full Hamiltonian and CCSD/(T) skipped; output="+directory.string());
  }
  parallel::barrier();
}
} // namespace localized_cc
