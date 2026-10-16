#include "localized_cc/ccsd.hpp"
#include "localized_cc/benchmark_output.hpp"
#include "localized_cc/canonical_triples.hpp"
#include "localized_cc/checkpoint.hpp"
#include "localized_cc/cuda_triples.hpp"
#include "localized_cc/diis.hpp"
#include "localized_cc/diis_solver.hpp"
#include "localized_cc/distributed_block_tensor.hpp"
#include "localized_cc/distributed_pair_tensor.hpp"
#include "localized_cc/frozen_core.hpp"
#include "localized_cc/partition.hpp"
#include "localized_cc/parallel.hpp"
#include "localized_cc/selected_ccsd_manifold.hpp"
#include "localized_cc/selected_ccsd_storage.hpp"
#include "localized_cc/spatial_ccsd.hpp"
#include "localized_cc/sparse_ccsd.hpp"
#include "localized_cc/tce_output.hpp"
#include "localized_cc/triples.hpp"
#include "localized_cc/water_zmatrix.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
using namespace localized_cc;
namespace {
void require(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
void close(double a,double b,double t,const char* m){if(std::abs(a-b)>t){std::cerr<<m<<": "<<a<<" vs "<<b<<'\n';throw std::runtime_error(m);}}
void test_scale_invariant_diis(){
  const std::vector<std::vector<double>> gram{{4.0,1.0},{1.0,9.0}};
  const auto reference=solve_diis_coefficients(gram);
  require(reference.has_value()&&reference->first_history==0&&
              reference->coefficients.size()==2,
          "scale-invariant DIIS reference solve");
  for(const double scale:{1.0e-200,1.0e200}){
    auto scaled=gram;
    for(auto& row:scaled)for(auto& value:row)value*=scale;
    const auto solution=solve_diis_coefficients(scaled);
    require(solution.has_value()&&solution->first_history==0,
            "scale-invariant DIIS scaled solve");
    close(solution->coefficients[0],reference->coefficients[0],1e-14,
          "scale-invariant DIIS first coefficient");
    close(solution->coefficients[1],reference->coefficients[1],1e-14,
          "scale-invariant DIIS second coefficient");
  }
  const std::vector<std::vector<double>> duplicate{{1.0,1.0},{1.0,1.0}};
  const auto regularized=solve_diis_coefficients(duplicate);
  require(regularized.has_value()&&regularized->coefficients.size()==2,
          "dependent DIIS relative fallback");
  close(regularized->coefficients[0]+regularized->coefficients[1],1.0,1e-14,
        "dependent DIIS coefficient closure");
  Diis diis(4);
  const auto first=diis.extrapolate({1.0,3.0},{1.0e-12,0.0});
  require(first==std::vector<double>({1.0,3.0}),"first tiny-error DIIS step");
  const auto second=diis.extrapolate({5.0,7.0},{0.0,2.0e-12});
  close(second[0],1.8,1e-12,"tiny-error DIIS first vector component");
  close(second[1],3.8,1e-12,"tiny-error DIIS second vector component");

  Diis recomputed(4,DiisGramBackend::recomputed);
  Diis incremental(4,DiisGramBackend::incremental);
  std::size_t recomputed_products=0;
  std::size_t incremental_products=0;
  for(std::size_t step=0;step<12;++step){
    std::vector<double> vector(17),error(17);
    for(std::size_t index=0;index<vector.size();++index){
      const auto x=static_cast<double>((step+1)*(index+2));
      vector[index]=std::sin(0.17*x)+0.03*static_cast<double>(step);
      error[index]=std::cos(0.11*x)+0.02*static_cast<double>(index);
    }
    const auto reference_vector=recomputed.extrapolate(vector,error);
    const auto incremental_vector=incremental.extrapolate(vector,error);
    require(reference_vector.size()==incremental_vector.size(),
            "incremental DIIS output size");
    for(std::size_t index=0;index<reference_vector.size();++index)
      close(incremental_vector[index],reference_vector[index],1e-13,
            "incremental/recomputed DIIS output");
    require(incremental.history_size()==recomputed.history_size()&&
                incremental.history_size()<=4,
            "incremental DIIS history eviction");
    recomputed_products+=recomputed.last_work().error_dot_products;
    incremental_products+=incremental.last_work().error_dot_products;
  }
  require(incremental_products<recomputed_products,
          "incremental DIIS Gram-product reduction");

  Diis pruned_reference(4,DiisGramBackend::recomputed);
  Diis pruned_incremental(4,DiisGramBackend::incremental);
  const std::vector<std::pair<std::vector<double>,std::vector<double>>> cases{
      {{1.0,2.0,3.0},{1.0,0.0,0.0}},
      {{2.0,3.0,4.0},{1.0,0.0,0.0}},
      {{3.0,4.0,5.0},{0.0,1.0,0.0}},
      {{4.0,5.0,6.0},{0.0,0.0,1.0}}};
  for(const auto& [vector,error]:cases){
    const auto expected=pruned_reference.extrapolate(vector,error);
    const auto actual=pruned_incremental.extrapolate(vector,error);
    for(std::size_t index=0;index<expected.size();++index)
      close(actual[index],expected[index],1e-13,
            "rank-pruned incremental DIIS output");
    require(pruned_incremental.history_size()==pruned_reference.history_size(),
            "rank-pruned incremental DIIS history");
  }
  require(pruned_incremental.history_size()<cases.size(),
          "incremental DIIS rank-pruning exercised");
}
void test_parallel_group_reduction(){
  const auto world=parallel::world();
  const std::size_t color=world.size==1?0:world.rank%2;
  std::vector<double> values{static_cast<double>(world.rank+1)};
  parallel::sum_in_place_group(values,color);
  std::uint64_t expected=0;
  for(std::size_t rank=0;rank<world.size;++rank)
    if((world.size==1?0:rank%2)==color)expected+=rank+1;
  close(values.front(),static_cast<double>(expected),0.0,
        "MPI subgroup vector reduction");
  require(parallel::sum_group(static_cast<std::uint64_t>(world.rank+1),color)==
              expected,
          "MPI subgroup counter reduction");
}
void test_occupied_pair_layout(){
  const OccupiedPairLayout layout(3,5);
  require(layout.composite_extent()==15&&layout.pair_count()==9,
          "occupied-pair layout extents");
  require(layout.element_count()==225&&
              layout.global_bytes()==225*sizeof(double),
          "occupied-pair layout memory");
  require(layout.composite_index(2,4)==14,
          "occupied-pair composite index");
  require(layout.pair_from_flat(7)==std::array<std::size_t,2>{2,1},
          "occupied-pair flattened task");
  std::size_t owned=0;
  for(std::size_t rank=0;rank<4;++rank)
    owned+=layout.owned_pair_count(rank,4);
  require(owned==layout.pair_count(),"occupied-pair ownership coverage");
  bool zero_rejected=false;
  try{(void)OccupiedPairLayout(0,5);}
  catch(const std::invalid_argument&){zero_rejected=true;}
  require(zero_rejected,"zero occupied-pair dimension rejection");
  bool overflow_rejected=false;
  try{(void)OccupiedPairLayout(std::numeric_limits<std::size_t>::max(),2);}
  catch(const std::overflow_error&){overflow_rejected=true;}
  require(overflow_rejected,"occupied-pair overflow rejection");
}
void test_global_arrays_selected_block_tensor(){
  const SelectedBlockLayout layout({3,5,2});
  require(layout.block_count()==3&&layout.maximum_block_size()==5&&
              layout.scientific_element_count()==10&&
              layout.allocated_element_count()==15,
          "selected block layout extents");
  require(layout.block_owned(2,0,2)&&!layout.block_owned(1,0,2),
          "selected block cyclic ownership");
  validate_global_arrays_block_name("tb_valid");
  bool long_name_rejected=false;
  try{
    validate_global_arrays_block_name(
        std::string(global_arrays_block_name_maximum+1,'x'));
  }catch(const std::invalid_argument&){long_name_rejected=true;}
  require(long_name_rejected,
          "selected block Global Arrays name-length rejection");
  if(!global_arrays_backend_available())return;
  const std::vector<std::vector<double>> source{
      {1.0,2.0,3.0},{4.0,5.0,6.0,7.0,8.0},{9.0,10.0}};
  GlobalArraysBlockTensor distributed(layout,"tb_a");
  distributed.load_replicated(source);
  require(distributed.replicate()==source,
          "Global Arrays selected block round trip");
  close(distributed.dot(distributed),385.0,0.0,
        "Global Arrays selected block dot product");
  const auto memory=distributed.memory();
  require(memory.scientific_global_bytes==10*sizeof(double)&&
              memory.allocated_global_bytes==15*sizeof(double)&&
              memory.replicated_scientific_bytes==10*sizeof(double)&&
              std::abs(memory.padding_factor-1.5)<1e-15,
          "Global Arrays selected block memory report");
  std::vector<double> aggregate{
      static_cast<double>(memory.local_allocated_bytes)};
  parallel::sum_in_place(aggregate);
  close(aggregate.front(),
        static_cast<double>(memory.allocated_global_bytes),0.0,
        "Global Arrays selected block distributed byte coverage");
  std::size_t owned=0;
  for(std::size_t block=0;block<layout.block_count();++block)
    if(distributed.computation_block_owned(block))++owned;
  std::vector<double> ownership{static_cast<double>(owned)};
  parallel::sum_in_place(ownership);
  close(ownership.front(),static_cast<double>(layout.block_count()),0.0,
        "Global Arrays selected block ownership coverage");
  GlobalArraysBlockTensor copied(layout,"tb_b");
  copied.copy_from(distributed);
  std::vector<double> increment(5,0.0);increment[3]=0.25;
  if(parallel::world().rank==0)copied.accumulate_block(1,increment);
  copied.synchronize();
  const auto changed=copied.get_block(1);
  close(changed[3],7.25,0.0,
        "Global Arrays selected block accumulate");
  GlobalArraysBlockTensor vector_two(layout,"tb_v2");
  auto source_two=source;
  for(auto& block:source_two)for(auto& value:block)value+=2.0;
  vector_two.load_replicated(source_two);
  GlobalArraysBlockTensor error_one(layout,"tb_e1");
  GlobalArraysBlockTensor error_two(layout,"tb_e2");
  auto error_blocks_one=source;
  auto error_blocks_two=source;
  for(auto& block:error_blocks_one)std::fill(block.begin(),block.end(),0.0);
  for(auto& block:error_blocks_two)std::fill(block.begin(),block.end(),0.0);
  error_blocks_one[0][0]=1.0;
  error_blocks_two[2][1]=1.0;
  error_one.load_replicated(error_blocks_one);
  error_two.load_replicated(error_blocks_two);
  GlobalArraysBlockTensor output(layout,"tb_out");
  Matrix singles_one({2,2});
  Matrix singles_two({2,2});
  Matrix singles_error_one({2,2});
  Matrix singles_error_two({2,2});
  for(std::size_t index=0;index<singles_one.size();++index){
    singles_one.data()[index]=static_cast<double>(index);
    singles_two.data()[index]=static_cast<double>(index)+2.0;
  }
  GlobalArraysBlockDiis diis(2,layout,"tb_d");
  const auto first=diis.extrapolate_distributed(
      singles_one,distributed,singles_error_one,error_one,output);
  require(first.data()==singles_one.data()&&output.replicate()==source,
          "first selected block DIIS iterate");
  const auto second=diis.extrapolate_distributed(
      singles_two,vector_two,singles_error_two,error_two,output);
  for(std::size_t index=0;index<second.size();++index)
    close(second.data()[index],
          0.5*(singles_one.data()[index]+singles_two.data()[index]),1e-12,
          "selected block DIIS singles extrapolation");
  const auto diis_blocks=output.replicate();
  for(std::size_t block=0;block<source.size();++block)
    for(std::size_t index=0;index<source[block].size();++index)
      close(diis_blocks[block][index],
            0.5*(source[block][index]+source_two[block][index]),1e-12,
            "selected block DIIS doubles extrapolation");
  const auto diis_memory=diis.memory();
  require(diis_memory.history_entries==2&&
              diis_memory.doubles_history_scientific_global_bytes==
                  4*10*sizeof(double)&&
              diis_memory.doubles_history_allocated_global_bytes==
                  4*15*sizeof(double),
          "selected block DIIS memory report");
  SelectedCcsdManifoldOptions selected_options;
  selected_options.maximum_monomers=1;
  const auto plan=plan_selected_ccsd_manifold(
      std::vector<int>{0,1},std::vector<int>{0,1},selected_options);
  const SelectedDoublesLayout doubles_layout(plan);
  const SelectedIntermediateLayout wmnij_layout(
      plan,plan.required_wmnij_blocks);
  require(wmnij_layout.block_count()==2&&
              wmnij_layout.selected({0,0,0,0})&&
              !wmnij_layout.selected({0,1,0,1})&&
              wmnij_layout.element_index({1,1,1,1})==0&&
              wmnij_layout.axis_orbitals(
                  wmnij_layout.block_index({1,1,1,1}),0)==
                  std::vector<std::size_t>({1}),
          "selected intermediate addressing");
  Tensor4 dense({2,2,2,2});dense(0,0,0,0)=1.25;dense(1,1,1,1)=2.5;
  const auto selected_blocks=doubles_layout.pack(dense);
  const SelectedDoublesView selected_view(doubles_layout,selected_blocks);
  require(selected_view.dimensions()==
              std::array<std::size_t,4>{2,2,2,2}&&
              selected_view.size()==16&&selected_view.all_finite()&&
              selected_view(0,0,0,0)==1.25&&
              selected_view(1,1,1,1)==2.5&&
              selected_view(0,1,0,1)==0.0,
          "selected doubles read-only view");
  GlobalArraysBlockTensor amplitudes(
      doubles_layout.distributed_layout(),"tb_amp");
  amplitudes.load_replicated(doubles_layout.pack(dense));
  SelectedDoublesPhaseCache cache(
      doubles_layout,amplitudes,sizeof(double));
  cache.prepare({doubles_layout.block_index(0,0,0,0)});
  close(cache.value(0,0,0,0),1.25,0.0,
        "selected doubles phase-cache value");
  close(cache.value(0,1,0,1),0.0,0.0,
        "selected doubles phase-cache structural zero");
  bool cache_miss=false;
  try{static_cast<void>(cache.value(1,1,1,1));}
  catch(const std::logic_error&){cache_miss=true;}
  require(cache_miss,"selected doubles phase-cache fail-closed miss");
  const auto cache_statistics=cache.statistics();
  require(cache_statistics.prepared_phases==1&&
              cache_statistics.fetched_blocks==1&&
              cache_statistics.fetched_bytes==sizeof(double)&&
              cache_statistics.peak_cached_bytes==sizeof(double),
          "selected doubles phase-cache diagnostics");
  GlobalArraysBlockTensor intermediate_values(
      wmnij_layout.distributed_layout(),"tb_wmn");
  std::vector<std::vector<double>> intermediate_blocks(
      wmnij_layout.block_count());
  for(std::size_t block=0;block<wmnij_layout.block_count();++block)
    intermediate_blocks[block].assign(
        wmnij_layout.block(block).element_count,
        static_cast<double>(block+3));
  intermediate_values.load_replicated(intermediate_blocks);
  SelectedIntermediatePhaseCache intermediate_cache(
      wmnij_layout,intermediate_values,sizeof(double));
  const auto first_intermediate=
      wmnij_layout.block_index({0,0,0,0});
  intermediate_cache.prepare({first_intermediate});
  close(intermediate_cache({0,0,0,0}),
        intermediate_blocks[first_intermediate][0],0.0,
        "selected intermediate phase-cache value");
  close(intermediate_cache({0,1,0,1}),0.0,0.0,
        "selected intermediate phase-cache structural zero");
  bool intermediate_cache_miss=false;
  try{static_cast<void>(intermediate_cache({1,1,1,1}));}
  catch(const std::logic_error&){intermediate_cache_miss=true;}
  require(intermediate_cache_miss,
          "selected intermediate phase-cache fail-closed miss");
  const auto intermediate_statistics=intermediate_cache.statistics();
  require(intermediate_statistics.prepared_phases==1&&
              intermediate_statistics.fetched_blocks==1&&
              intermediate_statistics.fetched_bytes==sizeof(double)&&
              intermediate_statistics.peak_cached_bytes==sizeof(double),
          "selected intermediate phase-cache diagnostics");
}
void test_global_arrays_pair_tensor(){
  if(!global_arrays_backend_available())return;
  const OccupiedPairLayout layout(2,3);
  Tensor4 source({2,2,3,3});
  for(std::size_t i=0;i<2;++i)
    for(std::size_t j=0;j<2;++j)
      for(std::size_t a=0;a<3;++a)
        for(std::size_t b=0;b<3;++b)
          source(i,j,a,b)=static_cast<double>(1000*i+100*j+10*a+b);
  GlobalArraysPairTensor distributed(layout,"ccsd_test_t2");
  distributed.load_replicated(source);
  const auto pair=distributed.get_pair(1,0);
  for(std::size_t a=0;a<3;++a)
    for(std::size_t b=0;b<3;++b)
      close(pair[a*3+b],source(1,0,a,b),0.0,
            "Global Arrays occupied-pair get");
  const auto restored=distributed.replicate();
  require(restored.data()==source.data(),
          "Global Arrays replicated boundary round trip");
  const auto memory=distributed.memory();
  require(memory.global_bytes==layout.global_bytes()&&
              memory.replicated_bytes==layout.global_bytes()&&
              memory.local_bytes<=memory.global_bytes,
          "Global Arrays memory report");
  std::size_t computation_pairs=0;
  for(std::size_t i=0;i<layout.occupied();++i)
    for(std::size_t j=0;j<layout.occupied();++j)
      if(distributed.computation_pair_owned(i,j))++computation_pairs;
  require(computation_pairs==layout.owned_pair_count(
              parallel::world().rank,parallel::world().size),
          "Global Arrays computation ownership API");
  std::vector<double> aggregate{
      static_cast<double>(memory.local_bytes),
      static_cast<double>(layout.owned_pair_count(
          parallel::world().rank,parallel::world().size))};
  parallel::sum_in_place(aggregate);
  close(aggregate[0],static_cast<double>(memory.global_bytes),0.0,
        "Global Arrays distributed byte coverage");
  close(aggregate[1],static_cast<double>(layout.pair_count()),0.0,
        "Global Arrays computation ownership coverage");
  if(parallel::world().size>1)
    require(memory.local_bytes<memory.replicated_bytes,
            "Global Arrays per-rank memory reduction");
  if(parallel::world().rank==0)
    std::cout<<"Global Arrays T2 bytes rank0 "<<memory.local_bytes
             <<" global "<<memory.global_bytes
             <<" ranks "<<parallel::world().size<<'\n';

  GlobalArraysPairTensor copied(layout,"ccsd_test_t2_copy");
  copied.copy_from(distributed);
  require(copied.replicate().data()==source.data(),
          "Global Arrays pair copy round trip");
  std::vector<double> increment(9,0.0);increment[5]=0.25;
  if(parallel::world().rank==0)copied.accumulate_pair(1,0,increment);
  copied.synchronize();
  const auto accumulated=copied.get_pair(1,0);
  close(accumulated[5],source(1,0,1,2)+0.25,0.0,
        "Global Arrays pair accumulate");

  GlobalArraysPairDiis diis(2,layout,"ccsd_test_diis");
  Matrix singles_one({2,3});
  Matrix singles_two({2,3});
  Matrix singles_error_one({2,3});
  Matrix singles_error_two({2,3});
  Tensor4 doubles_one=source;
  Tensor4 doubles_two=source;
  Tensor4 doubles_error_one({2,2,3,3});
  Tensor4 doubles_error_two({2,2,3,3});
  for(std::size_t index=0;index<singles_one.size();++index){
    singles_one.data()[index]=static_cast<double>(index);
    singles_two.data()[index]=static_cast<double>(index)+2.0;
  }
  for(auto& value:doubles_two.data())value+=2.0;
  doubles_error_one(0,0,0,0)=1.0;
  doubles_error_two(1,1,2,2)=1.0;
  const auto first=diis.extrapolate(singles_one,doubles_one,
                                    singles_error_one,doubles_error_one);
  require(first.first.data()==singles_one.data()&&
              first.second.data()==doubles_one.data(),
          "first distributed DIIS iterate");
  const auto second=diis.extrapolate(singles_two,doubles_two,
                                     singles_error_two,doubles_error_two);
  for(std::size_t index=0;index<second.first.size();++index)
    close(second.first.data()[index],
          0.5*(singles_one.data()[index]+singles_two.data()[index]),1e-12,
          "distributed DIIS singles extrapolation");
  for(std::size_t index=0;index<second.second.size();++index)
    close(second.second.data()[index],
          0.5*(doubles_one.data()[index]+doubles_two.data()[index]),1e-12,
          "distributed DIIS doubles extrapolation");
  const auto diis_memory=diis.memory();
  require(diis_memory.history_entries==2&&
              diis_memory.doubles_history_global_bytes==
                  4*layout.global_bytes()&&
              diis_memory.doubles_history_replicated_bytes_per_rank==
                  4*layout.global_bytes(),
          "distributed DIIS history memory report");
  std::vector<double> diis_aggregate{
      static_cast<double>(diis_memory.doubles_history_local_bytes)};
  parallel::sum_in_place(diis_aggregate);
  close(diis_aggregate[0],
        static_cast<double>(diis_memory.doubles_history_global_bytes),0.0,
        "distributed DIIS history byte coverage");
  if(parallel::world().size>1)
    require(diis_memory.doubles_history_local_bytes<
                diis_memory.doubles_history_replicated_bytes_per_rank,
            "distributed DIIS known-case per-rank memory reduction");
}
void test_frozen_core(){Matrix id({3,3});for(std::size_t i=0;i<3;++i)id(i,i)=1;Matrix cc({3,1}),co({3,1}),cv({3,1}),h({3,3});cc(0,0)=1;co(1,0)=1;cv(2,0)=1;h(0,0)=-2;h(1,1)=-1;h(2,2)=.5;Tensor4 eri({3,3,3,3});eri(2,2,0,0)=.2;eri(2,0,2,0)=.05;FrozenCoreInput in{cc,co,cv,id,h,eri,.7,-5.3,true};const auto spatial=build_frozen_core_spatial_hamiltonian(in);require(spatial.nocc==1&&spatial.nvirt==1&&spatial.frozen_core==1,"spatial frozen-core dimensions");close(spatial.reference_energy,-5.3,1e-12,"spatial reference energy");close(spatial.fock(1,1),.85,1e-12,"spatial core potential");require(spatial.active_occ_indices==std::vector<std::size_t>({1})&&spatial.virtual_indices==std::vector<std::size_t>({2}),"spatial explicit indices");const auto expanded=expand_spatial_hamiltonian_to_spin_orbital(spatial);auto x=build_frozen_core_hamiltonian(in);require(x.nocc==2&&x.nvirt==2,"frozen-core dimensions");close(x.reference_energy,-5.3,1e-12,"reference energy");close(x.fock(2,2),.85,1e-12,"core potential");require(x.active_occ_indices[0]==2&&x.virtual_indices[0]==4,"explicit indices");require(expanded.fock.data()==x.fock.data()&&expanded.hcore.data()==x.hcore.data()&&expanded.eri.data()==x.eri.data(),"explicit spatial-to-spin adapter equivalence");require(x.eri.size()==16*spatial.eri_chemist.logical_size(),"spatial-to-spin ERI memory ratio");auto invalid=spatial;invalid.fock=Matrix({1,1});bool rejected=false;try{(void)expand_spatial_hamiltonian_to_spin_orbital(invalid);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"invalid spatial RHF adapter dimensions");if(parallel::world().rank==0)std::cout<<"Spatial RHF working ERI elements "<<spatial.eri_chemist.size()<<" stored / "<<spatial.eri_chemist.logical_size()<<" logical; spin reference "<<x.eri.size()<<'\n';}
void test_spatial_eri_view_oracle(){
  constexpr std::size_t spatial_orbitals=4;
  SpatialRhfHamiltonian spatial;
  spatial.nocc=2;spatial.nvirt=2;
  spatial.fock=Matrix({spatial_orbitals,spatial_orbitals});
  spatial.hcore=Matrix({spatial_orbitals,spatial_orbitals});
  spatial.eri_chemist=SymmetricEri4(spatial_orbitals);
  spatial.fock(0,0)=-1.2;spatial.fock(1,1)=-0.8;
  spatial.fock(2,2)=0.5;spatial.fock(3,3)=0.9;
  for(std::size_t p=0;p<spatial_orbitals;++p)
    for(std::size_t q=0;q<spatial_orbitals;++q)
      for(std::size_t r=0;r<spatial_orbitals;++r)
        for(std::size_t s=0;s<spatial_orbitals;++s)
          spatial.eri_chemist(p,q,r,s)=
              1.0e-3/(1.0+static_cast<double>(p+q+r+s));
  spatial.active_occ_indices={0,1};spatial.virtual_indices={2,3};
  const SpinOrbitalEriView view(spatial);
  const auto expanded=expand_spatial_hamiltonian_to_spin_orbital(spatial);
  require(view.spatial_elements()==spatial.eri_chemist.logical_size()&&
              view.spatial_stored_elements()==spatial.eri_chemist.size()&&
              view.spin_reference_elements()==expanded.eri.size()&&
              expanded.eri.size()==16*view.spatial_elements(),
          "zero-copy spin ERI view memory accounting");
  for(std::size_t p=0;p<8;++p)for(std::size_t q=0;q<8;++q)
    for(std::size_t r=0;r<8;++r)for(std::size_t s=0;s<8;++s)
      close(view(p,q,r,s),expanded.eri(p,q,r,s),0.0,
            "zero-copy spin ERI element");
  CcsdOptions options;options.max_iterations=150;
  options.energy_tolerance=1.0e-12;options.residual_tolerance=1.0e-10;
  const auto dense=solve_ccsd(expanded.fock,expanded.eri,expanded.nocc,options);
  const auto zero_copy=solve_ccsd(spatial,options);
  require(dense.converged&&zero_copy.converged,
          "zero-copy spatial ERI CCSD convergence");
  close(zero_copy.correlation_energy,dense.correlation_energy,2.0e-12,
        "zero-copy spatial ERI CCSD energy");
  for(std::size_t index=0;index<dense.t1.size();++index)
    close(zero_copy.t1.data()[index],dense.t1.data()[index],2.0e-11,
          "zero-copy spatial ERI T1");
  for(std::size_t index=0;index<dense.t2.size();++index)
    close(zero_copy.t2.data()[index],dense.t2.data()[index],2.0e-11,
          "zero-copy spatial ERI T2");
  const auto spin_mask=make_closed_shell_spin_projection_mask(
      spatial.nocc,spatial.nvirt);
  require(spin_mask.singles.size()==16&&spin_mask.selected_singles==8&&
              spin_mask.doubles.size()==256&&
              spin_mask.selected_doubles==96,
          "closed-shell spin projection counts");
  CcsdOptions spin_selected_options=options;
  spin_selected_options.amplitude_backend=CcsdAmplitudeBackend::replicated;
  spin_selected_options.projection_mask=
      std::make_shared<CcsdProjectionMask>(spin_mask);
  const auto spin_selected=solve_ccsd(spatial,spin_selected_options);
  require(spin_selected.converged,
          "closed-shell spin projection convergence");
  close(spin_selected.correlation_energy,zero_copy.correlation_energy,2.0e-12,
        "closed-shell spin projection energy");
  close(spin_selected.r1_norm,zero_copy.r1_norm,2.0e-11,
        "closed-shell spin projection R1");
  close(spin_selected.r2_norm,zero_copy.r2_norm,2.0e-11,
        "closed-shell spin projection R2");
  for(std::size_t index=0;index<zero_copy.t1.size();++index)
    close(spin_selected.t1.data()[index],zero_copy.t1.data()[index],2.0e-11,
          "closed-shell spin projection T1");
  for(std::size_t index=0;index<zero_copy.t2.size();++index)
    close(spin_selected.t2.data()[index],zero_copy.t2.data()[index],2.0e-11,
          "closed-shell spin projection T2");
  const std::vector<int> occupied_labels{0,0,1,1};
  const std::vector<int> virtual_labels{0,0,1,1};
  const auto full_selected_mask=make_selected_ccsd_projection_mask(
      occupied_labels,virtual_labels);
  const auto intersection=intersect_ccsd_projection_masks(
      full_selected_mask,spin_mask);
  require(intersection.singles==spin_mask.singles&&
              intersection.doubles==spin_mask.doubles&&
              intersection.selected_singles==spin_mask.selected_singles&&
              intersection.selected_doubles==spin_mask.selected_doubles,
          "selected and closed-shell spin manifold intersection");
  const auto sparse_plan=plan_selected_ccsd_manifold(
      occupied_labels,virtual_labels);
  const SelectedIntegralStore selected_integrals(sparse_plan,spatial);
  CcsdOptions selected_integral_options=options;
  selected_integral_options.amplitude_backend=
      CcsdAmplitudeBackend::replicated;
  selected_integral_options.projection_mask=
      std::make_shared<CcsdProjectionMask>(full_selected_mask);
  const auto selected_integral_ccsd=solve_ccsd(
      spatial.fock,spatial.nocc,spatial.nvirt,selected_integrals,
      selected_integral_options);
  require(selected_integral_ccsd.converged,
          "selected-integral CCSD convergence");
  close(selected_integral_ccsd.correlation_energy,
        zero_copy.correlation_energy,2.0e-12,
        "selected-integral CCSD energy");
  for(std::size_t index=0;index<zero_copy.t1.size();++index)
    close(selected_integral_ccsd.t1.data()[index],
          zero_copy.t1.data()[index],2.0e-11,
          "selected-integral CCSD T1");
  for(std::size_t index=0;index<zero_copy.t2.size();++index)
    close(selected_integral_ccsd.t2.data()[index],
          zero_copy.t2.data()[index],2.0e-11,
          "selected-integral CCSD T2");
  CcsdOptions selected_panel_options=selected_integral_options;
  selected_panel_options.use_tiled_intermediates=true;
  selected_panel_options.use_tiled_doubles=true;
  selected_panel_options.selected_eri_panel_cache_max_bytes=
      16U*1024U*1024U;
  const auto selected_panel_ccsd=solve_ccsd(
      spatial.fock,spatial.nocc,spatial.nvirt,selected_integrals,
      selected_panel_options);
  require(selected_panel_ccsd.converged&&
              selected_panel_ccsd.selected_eri_panel_cache_enabled&&
              selected_panel_ccsd.selected_eri_panel_cache_bytes>0&&
              selected_panel_ccsd.selected_eri_panel_cache_bytes==
                  selected_panel_ccsd.selected_eri_panel_cache_required_bytes&&
              selected_panel_ccsd.selected_eri_panel_cache_populated_elements>0&&
              selected_panel_ccsd.selected_eri_panel_cache_build_seconds>=0.0,
          "selected-integral panel-cache CCSD diagnostics");
  close(selected_panel_ccsd.correlation_energy,
        selected_integral_ccsd.correlation_energy,2.0e-12,
        "selected-integral panel-cache CCSD energy");
  close(selected_panel_ccsd.r1_norm,selected_integral_ccsd.r1_norm,2.0e-11,
        "selected-integral panel-cache CCSD R1");
  close(selected_panel_ccsd.r2_norm,selected_integral_ccsd.r2_norm,2.0e-11,
        "selected-integral panel-cache CCSD R2");
  if(global_arrays_backend_available()){
    SparseCcsdOptions sparse_options;
    sparse_options.equations=options;
    std::vector<CcsdIterationReport> sparse_reports;
    sparse_options.equations.iteration_observer=
        [&](const CcsdIterationReport& report){sparse_reports.push_back(report);};
    sparse_options.maximum_monomers=
        std::numeric_limits<std::size_t>::max();
    sparse_options.maximum_phase_cache_bytes=16U*1024U*1024U;
    const auto sparse=solve_sparse_ccsd(spatial,sparse_plan,sparse_options);
    require(sparse.converged,"block-sparse CCSD convergence");
    require(!sparse_reports.empty()&&sparse_reports.back().converged,
            "block-sparse CCSD timing reports");
    const auto valid_timing=[](const CcsdIterationTiming& timing){
      const std::array values{
          timing.intermediates_seconds,timing.singles_seconds,
          timing.doubles_seconds,timing.wabef_seconds,
          timing.residual_seconds,timing.diis_seconds,
          timing.energy_seconds,timing.iteration_seconds};
      return std::ranges::all_of(values,[](const double value){
        return std::isfinite(value)&&value>=0.0;
      });
    };
    for(const auto& report:sparse_reports)
      require(valid_timing(report.timing)&&
                  report.timing.iteration_seconds>0.0,
              "block-sparse CCSD per-iteration timing fields");
    require(valid_timing(sparse.cumulative_timing)&&
                sparse.cumulative_timing.iteration_seconds>0.0,
            "block-sparse CCSD cumulative timing fields");
    require(!sparse.phase_statistics.empty(),
            "block-sparse CCSD phase traffic records");
    bool saw_current_to_tau=false,saw_wmbej_to_r2=false;
    for(const auto& phase:sparse.phase_statistics){
      const bool declared_cache=
          phase.phase=="doubles/wmnij_to_r2"||
          phase.phase=="doubles/wmbej_to_r2";
      require(!phase.phase.empty()&&phase.ranks==parallel::world().size&&
                  phase.rank==parallel::world().rank&&
                  phase.declared_source_blocks>0&&
                  phase.unique_blocks_fetched==phase.declared_source_blocks&&
                  phase.locally_owned_source_blocks<=
                      phase.declared_source_blocks&&
                  phase.remote_unique_blocks_fetched==
                      phase.declared_source_blocks-
                          phase.locally_owned_source_blocks&&
                  phase.fetched_bytes>=phase.remote_fetched_bytes&&
                  phase.replicate_calls==(declared_cache?0U:1U)&&
                  phase.cache_hits==0&&
                  phase.dgemm_contractions==0&&
                  phase.ga_gets_inside_kernel==0&&
                  std::isfinite(phase.fetch_seconds)&&
                  phase.fetch_seconds>=0.0,
              "block-sparse CCSD baseline phase traffic fields");
      saw_current_to_tau|=phase.phase=="iteration/current_to_tau";
      saw_wmbej_to_r2|=phase.phase=="doubles/wmbej_to_r2";
    }
    require(saw_current_to_tau&&saw_wmbej_to_r2,
            "block-sparse CCSD named phase traffic coverage");
    close(sparse.correlation_energy,dense.correlation_energy,2.0e-12,
          "block-sparse CCSD energy");
    const SelectedDoublesLayout sparse_layout(sparse_plan);
    const auto sparse_t2=sparse_layout.unpack(sparse.t2_blocks);
    for(std::size_t index=0;index<dense.t1.size();++index)
      close(sparse.t1.data()[index],dense.t1.data()[index],2.0e-11,
            "block-sparse CCSD T1");
    for(std::size_t index=0;index<dense.t2.size();++index)
      close(sparse_t2.data()[index],dense.t2.data()[index],2.0e-11,
            "block-sparse CCSD T2");
  }
  const auto dense_triples=solve_canonical_triples_streamed(
      expanded.fock,expanded.eri,dense.t1,dense.t2);
  const auto zero_copy_triples=solve_canonical_triples_streamed(
      spatial,zero_copy.t1,zero_copy.t2);
  close(zero_copy_triples.bracket_energy,dense_triples.bracket_energy,
        2.0e-13,"zero-copy spatial ERI CCSD[T]");
  close(zero_copy_triples.parenthetical_energy,
        dense_triples.parenthetical_energy,2.0e-13,
        "zero-copy spatial ERI CCSD(T)");
  const auto dense_partition=partition_ccsd(
      expanded.fock,expanded.eri,zero_copy.t1,zero_copy.t2,expanded.nocc,
      occupied_labels,virtual_labels);
  const auto zero_copy_partition=partition_ccsd(
      spatial,zero_copy.t1,zero_copy.t2,occupied_labels,virtual_labels);
  close(zero_copy_partition.sum,dense_partition.sum,1.0e-14,
        "zero-copy spatial ERI CCSD partition");
  close(zero_copy_partition.parent_energy,dense_partition.parent_energy,
        1.0e-14,"zero-copy spatial ERI CCSD partition parent");
  if(parallel::world().rank==0)
    std::cout<<"Zero-copy spatial ERI elements "<<view.spatial_elements()
             <<" eliminated spin elements "<<view.spin_reference_elements()
             <<" ratio 16\n";
}
void test_spatial_diis_weighting(){
  const std::vector<double> d1{-2.0}, d2{-4.0,-4.0,-4.0,-4.0};
  const std::vector<double> raw{2.0,4.0,4.0,4.0,4.0};
  auto error=raw;
  weight_spatial_diis_error(error,d1,d2,SpatialDiisMetric::raw);
  require(error==raw,"raw spatial DIIS is unchanged");
  weight_spatial_diis_error(error,d1,d2,SpatialDiisMetric::rms);
  for(const auto x:error) close(x,2.0,1.0e-15,"RMS block weighting");
  error=raw;
  weight_spatial_diis_error(error,d1,d2,SpatialDiisMetric::preconditioned_rms);
  close(error[0],-1.0,1.0e-15,"signed singles DIIS preconditioning");
  for(std::size_t k=1;k<error.size();++k)
    close(error[k],-0.5,1.0e-15,"signed doubles DIIS preconditioning");
  bool rejected=false;
  try { weight_spatial_diis_error(error,d1,std::vector<double>{0.0},SpatialDiisMetric::rms); }
  catch(const std::invalid_argument&) { rejected=true; }
  require(rejected,"DIIS rejects inconsistent dimensions");
  rejected=false;
  try { weight_spatial_diis_error(error,d1,std::vector<double>(4,0.0),SpatialDiisMetric::preconditioned_rms); }
  catch(const std::invalid_argument&) { rejected=true; }
  require(rejected,"DIIS rejects zero preconditioners");
}

void test_native_spatial_pairs(){
  test_spatial_diis_weighting();
  // Nonseparable deterministic ERIs, unequal o/v dimensions, noncanonical
  // Fock blocks, and independent pair-symmetric (not individually symmetric)
  // amplitudes expose permutations hidden by the older sum-of-indices case.
  for(const auto dims : {std::pair<std::size_t,std::size_t>{1,2},{2,3},{3,2}}) {
    const auto [o,v]=dims;
    const auto n=o+v;
    SpatialRhfHamiltonian h;
    h.nocc=o;h.nvirt=v;h.fock=Matrix({n,n});h.hcore=Matrix({n,n});
    h.eri_chemist=SymmetricEri4(n);
    for(std::size_t p=0;p<n;++p) for(std::size_t q=0;q<n;++q)
      h.fock(p,q)=p==q ? (p<o?-1.2-0.13*static_cast<double>(p):0.4+0.17*static_cast<double>(p))
          : 0.003*std::cos(static_cast<double>((p+1)*(q+1)));
    for(std::size_t k=0;k<h.eri_chemist.data().size();++k)
      h.eri_chemist.data()[k]=0.002*std::sin(0.73*static_cast<double>(k+1));
    CcsdOptions reference;
    reference.amplitude_backend=CcsdAmplitudeBackend::replicated;
    auto native=reference;
    native.wabef_backend=CcsdWabefBackend::tiled_cpu;
    native.use_direct_spatial_singles=true;
    native.use_direct_spatial_doubles=true;
    native.use_tiled_spatial_pairs=true;
    native.use_root_replicated_diis=true;
    for(const double scale:{1e-4,1e-2,1e-1}) {
      SpatialRccsdAmplitudes a{Matrix({o,v}),Tensor4({o,o,v,v})};
      for(std::size_t k=0;k<a.t1.size();++k)
        a.t1.data()[k]=scale*std::sin(0.37*static_cast<double>(k+1));
      for(std::size_t i=0;i<o;++i) for(std::size_t j=0;j<o;++j)
        for(std::size_t x=0;x<v;++x) for(std::size_t y=0;y<v;++y) {
          const auto k=((i*o+j)*v+x)*v+y, l=((j*o+i)*v+y)*v+x;
          a.t2(i,j,x,y)=0.5*scale*(std::sin(0.61*static_cast<double>(k+1))+
              std::sin(0.61*static_cast<double>(l+1)));
        }
      const auto oracle=evaluate_spatial_rccsd_residual(h,a,reference);
      for(const bool compact : {false,true}) {
      native.use_compact_spatial_doubles=compact;
      for(const std::size_t cap:{std::size_t{128*1024},std::size_t{4*1024*1024}}) {
        native.wabef_tile_max_bytes=cap;
        const auto tiled=evaluate_spatial_rccsd_residual(h,a,native);
        for(const auto tensors:{std::pair{&oracle.r1.data(),&tiled.r1.data()},
                                std::pair{&oracle.r2.data(),&tiled.r2.data()}}) {
          double inf=0,l2=0,ref=0;
          for(std::size_t k=0;k<tensors.first->size();++k) {
            const double x=(*tensors.first)[k],y=(*tensors.second)[k];
            require(std::isfinite(x)&&std::isfinite(y),"native spatial finite residual");
            inf=std::max(inf,std::abs(x-y)); l2+=(x-y)*(x-y);
            ref=std::max({ref,std::abs(x),std::abs(y)});
          }
          if(parallel::world().rank==0)
            std::cout<<"Native spatial residual scale "<<scale<<" cap "<<cap<<" compact "<<compact
                     <<" occupied "<<o<<" virtual "<<v
                     <<" residual "<<(tensors.first==&oracle.r1.data()?"R1":"R2")
                     <<" infinity "<<inf<<" two "<<std::sqrt(l2)
                     <<" limit "<<(1e-12+1e-12*ref)<<std::endl;
          require(inf<=1e-12+1e-12*ref,"native spatial residual oracle");
        }
        require(tiled.timing.wabef_blas_contractions>0,"native spatial BLAS used");
        close(spatial_rccsd_pair_symmetry_error(tiled.r2),0.0,1e-12,"native spatial R2 pair symmetry");
      }
      }
      auto too_small=native;too_small.wabef_invariant_cache_max_bytes=1;
      bool rejected=false;
      try { (void)evaluate_spatial_rccsd_residual(h,a,too_small); }
      catch(const std::runtime_error&) { rejected=true; }
      require(rejected,"native spatial workspace memory rejection");
    }
    // Isolate P(ij) t_i^e <ab||ej> with one occupied/three-virtual ERIs.
    // With T2=0 and no other ERI families, there are no mixed-pair,
    // Woooo or Wvvvv contributions. For i!=j the surviving exchange is
    // t_j^e (a i|b e), NOT t_j^e (a j|b e).
    if(o>=2) {
      auto single_h=h;
      std::fill(single_h.eri_chemist.data().begin(),single_h.eri_chemist.data().end(),0.0);
      single_h.eri_chemist(o,0,o+1,o+1)=0.007;
      SpatialRccsdAmplitudes single_a{Matrix({o,v}),Tensor4({o,o,v,v})};
      single_a.t1(1,1)=0.03;
      native.use_compact_spatial_doubles=true;
      const auto single_ref=evaluate_spatial_rccsd_residual(single_h,single_a,reference);
      const auto single_new=evaluate_spatial_rccsd_residual(single_h,single_a,native);
      for(std::size_t i=0;i<o;++i) for(std::size_t j=0;j<o;++j)
        for(std::size_t a=0;a<v;++a) for(std::size_t b=0;b<v;++b) {
          double expected=0.0;
          for(std::size_t e=0;e<v;++e)
            expected+=single_a.t1(i,e)*single_h.eri_chemist(o+a,o+e,o+b,j)
                     +single_a.t1(j,e)*single_h.eri_chemist(o+a,i,o+b,o+e);
          close(single_ref.r2(i,j,a,b),expected,1e-12,"singles O-VVV analytic reference");
          close(single_new.r2(i,j,a,b),expected,1e-12,"compact singles O-VVV occupied index");
        }
      close(single_new.r2(0,1,0,1),0.00021,1e-12,"compact nonzero singles exchange");
    }
    // Analytic zero-amplitude limit: R2_ijab=(ia|jb), including nonzero Fov.
    SpatialRccsdAmplitudes zero{Matrix({o,v}),Tensor4({o,o,v,v})};
    native.use_compact_spatial_doubles=true;
    const auto bare=evaluate_spatial_rccsd_residual(h,zero,native);
    for(std::size_t i=0;i<o;++i) for(std::size_t j=0;j<o;++j)
      for(std::size_t x=0;x<v;++x) for(std::size_t y=0;y<v;++y)
        close(bare.r2(i,j,x,y),h.eri_chemist(i,o+x,j,o+y),1e-12,"compact spatial bare R2");
    auto invalid=native; invalid.use_tiled_spatial_pairs=false;
    bool rejected=false;
    try { (void)evaluate_spatial_rccsd_residual(h,zero,invalid); }
    catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"compact spatial requires native pairs");
    invalid=native; invalid.use_pm_residual_orbits=true;
    rejected=false;
    try { (void)evaluate_spatial_rccsd_residual(h,zero,invalid); }
    catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"compact spatial rejects unvalidated PM orbit combination");
    reference.max_iterations=native.max_iterations=150;
    reference.residual_tolerance=native.residual_tolerance=1e-11;
    reference.energy_tolerance=native.energy_tolerance=1e-12;
    native.audit_compact_spatial_doubles=true;
    const auto a=solve_spatial_rccsd_reference(h,reference);
    const auto b=solve_spatial_rccsd_reference(h,native);
    require(a.converged&&b.converged,"native spatial nonlinear convergence");
    close(a.correlation_energy,b.correlation_energy,1e-11,"native spatial converged energy");
    require(b.residual_workspace_bytes==0,"native spatial avoids expanded solve workspace");
    native.initial_t1=expand_spatial_rccsd_t1_to_spin_orbital(b.amplitudes.t1);
    native.initial_t2=expand_spatial_rccsd_t2_to_spin_orbital(b.amplitudes.t2);
    const auto restart=solve_spatial_rccsd_reference(h,native);
    require(restart.converged,"native spatial restart convergence");
    close(restart.correlation_energy,b.correlation_energy,1e-11,"native spatial restart energy");
  }
}
void test_spatial_rccsd_convention(){
  test_native_spatial_pairs();
  SpatialRhfHamiltonian spatial;
  spatial.nocc=2;spatial.nvirt=2;
  spatial.fock=Matrix({4,4});spatial.hcore=Matrix({4,4});
  spatial.eri_chemist=SymmetricEri4(4);
  spatial.fock(0,0)=-1.2;spatial.fock(1,1)=-0.9;
  spatial.fock(2,2)=0.4;spatial.fock(3,3)=0.7;
  spatial.fock(0,2)=0.02;spatial.fock(1,3)=-0.015;
  for(std::size_t p=0;p<4;++p)for(std::size_t q=0;q<4;++q)
    for(std::size_t r=0;r<4;++r)for(std::size_t s=0;s<4;++s)
      spatial.eri_chemist(p,q,r,s)=
          2.0e-3/(1.0+static_cast<double>(p+q+r+s));
  SpatialRccsdAmplitudes amplitudes{Matrix({2,2}),Tensor4({2,2,2,2})};
  amplitudes.t1(0,0)=0.03;amplitudes.t1(0,1)=-0.01;
  amplitudes.t1(1,0)=0.02;amplitudes.t1(1,1)=0.04;
  for(std::size_t i=0;i<2;++i)for(std::size_t j=0;j<2;++j)
    for(std::size_t a=0;a<2;++a)for(std::size_t b=0;b<2;++b)
      amplitudes.t2(i,j,a,b)=
          1.0e-2*(1.0+static_cast<double>(i+j+a+b));
  close(spatial_rccsd_pair_symmetry_error(amplitudes.t2),0.0,0.0,
        "spatial RCCSD pair symmetry");
  const auto spin_fock=expand_spatial_one_body_to_spin_orbital(spatial.fock);
  const SpinOrbitalEriView spin_eri_view(spatial);
  Tensor4 spin_eri(spin_eri_view.dimensions());
  for(std::size_t p=0;p<8;++p)for(std::size_t q=0;q<8;++q)
    for(std::size_t r=0;r<8;++r)for(std::size_t s=0;s<8;++s)
      spin_eri(p,q,r,s)=spin_eri_view(p,q,r,s);
  const auto spin_t1=expand_spatial_rccsd_t1_to_spin_orbital(amplitudes.t1);
  const auto spin_t2=expand_spatial_rccsd_t2_to_spin_orbital(amplitudes.t2);
  require(t2_antisymmetry_error(spin_t2)<1.0e-15,
          "expanded spatial RCCSD T2 antisymmetry");
  close(spatial_rccsd_energy(spatial,amplitudes),
        ccsd_energy(spin_fock,spin_eri,spin_t1,spin_t2,4),1.0e-14,
        "spatial RCCSD correlation-energy convention");

  CcsdOptions residual_options;
  residual_options.wabef_backend=CcsdWabefBackend::cpu;
  const auto spin_residual=evaluate_ccsd_residual(
      spin_fock,spin_eri,spin_t1,spin_t2,4,residual_options);
  const auto spatial_residual=evaluate_spatial_rccsd_residual(
      spatial,amplitudes,residual_options);
  CcsdOptions cached_spatial_residual_options=residual_options;
  cached_spatial_residual_options.wabef_backend=CcsdWabefBackend::tiled_cpu;
  cached_spatial_residual_options.wabef_packing_backend=
      CcsdWabefPackingBackend::cached_invariants;
  cached_spatial_residual_options.wabef_invariant_cache_max_bytes=
      1024U*1024U;
  const auto cached_spatial_residual=evaluate_spatial_rccsd_residual(
      spatial,amplitudes,cached_spatial_residual_options);
  auto symmetric_options=cached_spatial_residual_options;
  symmetric_options.use_wabef_pair_symmetry=true;
  const auto symmetric_residual=evaluate_spatial_rccsd_residual(
      spatial,amplitudes,symmetric_options);
  for(std::size_t i=0;i<2;++i)for(std::size_t a=0;a<2;++a)
  {
    close(spatial_residual.r1(i,a),spin_residual.r1(2*i,2*a),1.0e-13,
          "spatial RCCSD singles residual projection");
    close(cached_spatial_residual.r1(i,a),spatial_residual.r1(i,a),1.0e-13,
          "cached spatial RCCSD singles residual");
    close(symmetric_residual.r1(i,a),spatial_residual.r1(i,a),1.0e-13,
          "pair-symmetric spatial RCCSD singles residual");
  }
  for(std::size_t i=0;i<2;++i)for(std::size_t j=0;j<2;++j)
    for(std::size_t a=0;a<2;++a)for(std::size_t b=0;b<2;++b)
    {
      close(spatial_residual.r2(i,j,a,b),
            spin_residual.r2(2*i,2*j+1,2*a,2*b+1),1.0e-13,
            "spatial RCCSD doubles residual projection");
      close(cached_spatial_residual.r2(i,j,a,b),
            spatial_residual.r2(i,j,a,b),1.0e-13,
            "cached spatial RCCSD doubles residual");
      close(symmetric_residual.r2(i,j,a,b),
            spatial_residual.r2(i,j,a,b),1.0e-13,
            "pair-symmetric spatial RCCSD doubles residual");
    }
  close(spatial_rccsd_pair_symmetry_error(spatial_residual.r2),0.0,1.0e-13,
        "spatial RCCSD residual pair symmetry");

  for(const double scale:{1.0e-4,1.0e-2,1.0e-1}){
    SpatialRccsdAmplitudes scaled{Matrix({2,2}),Tensor4({2,2,2,2})};
    for(std::size_t i=0;i<2;++i)for(std::size_t a=0;a<2;++a)
      scaled.t1(i,a)=scale*(1.0+static_cast<double>(2*i+a));
    for(std::size_t i=0;i<2;++i)for(std::size_t j=0;j<2;++j)
      for(std::size_t a=0;a<2;++a)for(std::size_t b=0;b<2;++b)
        scaled.t2(i,j,a,b)=scale*(1.0+static_cast<double>(i+j+a+b));
    const auto scaled_spin_t1=
        expand_spatial_rccsd_t1_to_spin_orbital(scaled.t1);
    const auto scaled_spin_t2=
        expand_spatial_rccsd_t2_to_spin_orbital(scaled.t2);
    const auto scaled_spin_residual=evaluate_ccsd_residual(
        spin_fock,spin_eri,scaled_spin_t1,scaled_spin_t2,4,residual_options);
    const auto direct_singles=evaluate_spatial_rccsd_singles_direct(
        spatial,scaled,residual_options);
    const auto direct_doubles=evaluate_spatial_rccsd_doubles_direct(
        spatial,scaled,residual_options);
    double maximum_singles_difference=0.0;
    for(std::size_t i=0;i<2;++i)for(std::size_t a=0;a<2;++a){
      const auto difference=std::abs(
          direct_singles.r1(i,a)-scaled_spin_residual.r1(2*i,2*a));
      maximum_singles_difference=std::max(
          maximum_singles_difference,difference);
      close(difference,0.0,1.0e-12,
            "direct spatial RCCSD singles residual oracle");
    }
    double maximum_doubles_difference=0.0;
    for(std::size_t i=0;i<2;++i)for(std::size_t j=0;j<2;++j)
      for(std::size_t a=0;a<2;++a)for(std::size_t b=0;b<2;++b){
        const auto difference=std::abs(
            direct_doubles.r2(i,j,a,b)-
            scaled_spin_residual.r2(2*i,2*j+1,2*a,2*b+1));
        maximum_doubles_difference=std::max(
            maximum_doubles_difference,difference);
        close(difference,0.0,1.0e-12,
              "direct spatial RCCSD doubles residual oracle");
      }
    const auto global_intermediate_tasks=parallel::sum_group(
        static_cast<std::uint64_t>(direct_singles.local_intermediate_tasks),0);
    const auto global_output_tasks=parallel::sum_group(
        static_cast<std::uint64_t>(direct_singles.local_output_tasks),0);
    require(global_intermediate_tasks==
                direct_singles.global_intermediate_tasks,
            "direct spatial RCCSD intermediate owner-computes accounting");
    require(global_output_tasks==direct_singles.global_output_tasks,
            "direct spatial RCCSD singles owner-computes accounting");
    const auto global_doubles_tasks=parallel::sum_group(
        static_cast<std::uint64_t>(direct_doubles.local_output_tasks),0);
    require(global_doubles_tasks==direct_doubles.global_output_tasks,
            "direct spatial RCCSD doubles owner-computes accounting");
    require(direct_singles.avoided_spin_amplitude_bytes_per_rank>
                direct_singles.replicated_intermediate_bytes_per_rank,
            "direct spatial RCCSD singles memory reduction");
    require(direct_doubles.avoided_spin_amplitude_bytes_per_rank>
                direct_doubles.replicated_small_intermediate_bytes_per_rank,
            "direct spatial RCCSD doubles amplitude-memory reduction");
    require(direct_doubles.avoided_spin_four_index_intermediate_bytes_per_rank>
                direct_doubles.replicated_small_intermediate_bytes_per_rank,
            "direct spatial RCCSD doubles intermediate-memory reduction");
    close(spatial_rccsd_pair_symmetry_error(direct_doubles.r2),0.0,1.0e-12,
          "direct spatial RCCSD doubles pair symmetry");
    if(parallel::world().rank==0)
      std::cout<<"Spatial RCCSD direct singles scale "<<scale
               <<" maximum difference "<<maximum_singles_difference
               <<" intermediate bytes "
               <<direct_singles.replicated_intermediate_bytes_per_rank
               <<" avoided spin amplitude bytes "
               <<direct_singles.avoided_spin_amplitude_bytes_per_rank<<'\n'
               <<"Spatial RCCSD direct doubles scale "<<scale
               <<" maximum difference "<<maximum_doubles_difference
               <<" small intermediate bytes "
               <<direct_doubles.replicated_small_intermediate_bytes_per_rank
               <<" avoided spin amplitude bytes "
               <<direct_doubles.avoided_spin_amplitude_bytes_per_rank
               <<" avoided spin four-index intermediate bytes "
               <<direct_doubles.avoided_spin_four_index_intermediate_bytes_per_rank
               <<'\n';
  }

  CcsdOptions invalid_direct_doubles_options;
  invalid_direct_doubles_options.use_direct_spatial_doubles=true;
  bool incomplete_direct_spatial_executor_rejected=false;
  try{
    (void)evaluate_spatial_rccsd_residual(
        spatial,amplitudes,invalid_direct_doubles_options);
  }catch(const std::invalid_argument&){
    incomplete_direct_spatial_executor_rejected=true;
  }
  require(incomplete_direct_spatial_executor_rejected,
          "direct spatial RCCSD doubles requires direct singles");

  CcsdOptions solve_options;
  solve_options.max_iterations=120;
  solve_options.diis_start=1;
  solve_options.energy_tolerance=1.0e-12;
  solve_options.residual_tolerance=1.0e-10;
  solve_options.residual_metric=CcsdResidualMetric::rms;
  solve_options.wabef_backend=CcsdWabefBackend::cpu;
  solve_options.use_direct_spatial_singles=true;
  solve_options.use_direct_spatial_doubles=true;
  std::size_t spatial_checkpoint_calls=0;
  bool observed_spatial_convergence=false;
  solve_options.spatial_observer=
      [&](const SpatialRccsdProgress& progress){
        ++spatial_checkpoint_calls;
        observed_spatial_convergence=
            observed_spatial_convergence||progress.converged;
        require(progress.amplitudes.t1.dimensions()==
                    std::array<std::size_t,2>{2,2},
                "spatial RCCSD checkpoint T1 dimensions");
        require(progress.amplitudes.t2.dimensions()==
                    std::array<std::size_t,4>{2,2,2,2},
                "spatial RCCSD checkpoint T2 dimensions");
      };
  const auto spatial_solution=solve_spatial_rccsd_reference(
      spatial,solve_options);
  CcsdOptions spin_solve_options=solve_options;
  spin_solve_options.spatial_observer={};
  const auto spin_solution=solve_ccsd(spatial,spin_solve_options);
  CcsdOptions cached_spatial_solve_options=solve_options;
  cached_spatial_solve_options.use_direct_spatial_singles=false;
  cached_spatial_solve_options.use_direct_spatial_doubles=false;
  cached_spatial_solve_options.wabef_backend=CcsdWabefBackend::tiled_cpu;
  cached_spatial_solve_options.wabef_packing_backend=
      CcsdWabefPackingBackend::cached_invariants;
  cached_spatial_solve_options.wabef_invariant_cache_max_bytes=
      1024U*1024U;
  const auto cached_spatial_solution=solve_spatial_rccsd_reference(
      spatial,cached_spatial_solve_options);
  for(const auto metric:{SpatialDiisMetric::rms,SpatialDiisMetric::preconditioned_rms}) {
    auto weighted=cached_spatial_solve_options;
    weighted.spatial_diis_metric=metric;
    weighted.use_root_replicated_diis=true;
    weighted.diis_size=12;
    const auto solution=solve_spatial_rccsd_reference(spatial,weighted);
    require(solution.converged,"weighted spatial DIIS convergence");
    close(solution.correlation_energy,cached_spatial_solution.correlation_energy,
          1.0e-11,"weighted DIIS preserves converged CCSD energy");
    for(std::size_t k=0;k<solution.amplitudes.t1.size();++k)
      close(solution.amplitudes.t1.data()[k],cached_spatial_solution.amplitudes.t1.data()[k],
            1.0e-8,"weighted DIIS converged T1");
    for(std::size_t k=0;k<solution.amplitudes.t2.size();++k)
      close(solution.amplitudes.t2.data()[k],cached_spatial_solution.amplitudes.t2.data()[k],
            1.0e-8,"weighted DIIS converged T2");
    const auto checked=evaluate_spatial_rccsd_residual(spatial,solution.amplitudes,weighted);
    require(ccsd_residual_measure(checked.r1_norm,checked.r1.size(),weighted.residual_metric)<weighted.residual_tolerance &&
            ccsd_residual_measure(checked.r2_norm,checked.r2.size(),weighted.residual_metric)<weighted.residual_tolerance,
            "weighted DIIS retains unweighted residual gates");
    weighted.initial_t1=expand_spatial_rccsd_t1_to_spin_orbital(solution.amplitudes.t1);
    weighted.initial_t2=expand_spatial_rccsd_t2_to_spin_orbital(solution.amplitudes.t2);
    const auto restarted=solve_spatial_rccsd_reference(spatial,weighted);
    require(restarted.converged,"weighted DIIS amplitude-only restart");
    close(restarted.correlation_energy,solution.correlation_energy,1.0e-11,
          "weighted DIIS restart energy");
  }
  auto symmetric_solve_options=cached_spatial_solve_options;
  symmetric_solve_options.use_wabef_pair_symmetry=true;
  const auto symmetric_solution=solve_spatial_rccsd_reference(
      spatial,symmetric_solve_options);
  require(symmetric_solution.converged,
          "pair-symmetric spatial RCCSD convergence");
  close(symmetric_solution.correlation_energy,
        cached_spatial_solution.correlation_energy,1.0e-12,
        "pair-symmetric spatial RCCSD converged energy");
  require(symmetric_solution.wabef_invariant_cache_bytes <
              cached_spatial_solution.wabef_invariant_cache_bytes,
          "pair-symmetric spatial RCCSD cache reduction");
  for(std::size_t k=0;k<symmetric_solution.amplitudes.t2.size();++k)
    close(symmetric_solution.amplitudes.t2.data()[k],
          cached_spatial_solution.amplitudes.t2.data()[k],1.0e-11,
          "pair-symmetric spatial RCCSD converged T2");
  auto symmetric_restart=symmetric_solve_options;
  symmetric_restart.initial_t1=expand_spatial_rccsd_t1_to_spin_orbital(
      cached_spatial_solution.amplitudes.t1);
  symmetric_restart.initial_t2=expand_spatial_rccsd_t2_to_spin_orbital(
      cached_spatial_solution.amplitudes.t2);
  const auto restarted_symmetric=solve_spatial_rccsd_reference(
      spatial,symmetric_restart);
  require(restarted_symmetric.converged,
          "pair-symmetric spatial RCCSD amplitude restart convergence");
  close(restarted_symmetric.correlation_energy,
        symmetric_solution.correlation_energy,1.0e-11,
        "pair-symmetric spatial RCCSD amplitude restart energy");
  require(spatial_solution.converged&&spin_solution.converged,
          "spatial/spin RCCSD reference convergence");
  require(cached_spatial_solution.converged,
          "cached spatial RCCSD reference convergence");
  require(cached_spatial_solution.residual_workspace_bytes>0&&
              cached_spatial_solution.wabef_invariant_cache_bytes>0&&
              cached_spatial_solution.cumulative_timing
                      .wabef_factorized_cached_evaluations>0&&
              cached_spatial_solution.residual_workspace_build_seconds>=0.0&&
              cached_spatial_solution.wabef_invariant_cache_build_seconds>=0.0,
          "cached spatial RCCSD solve-lifetime workspace accounting");
  require(spatial_checkpoint_calls>0&&observed_spatial_convergence,
          "spatial RCCSD checkpoint observer convergence");
  close(spatial_solution.correlation_energy,spin_solution.correlation_energy,
        1.0e-11,"spatial/spin RCCSD converged energy");
  close(cached_spatial_solution.correlation_energy,
        spatial_solution.correlation_energy,1.0e-11,
        "cached/direct spatial RCCSD converged energy");
  close(spatial_rccsd_pair_symmetry_error(spatial_solution.amplitudes.t2),
        0.0,1.0e-12,"converged spatial RCCSD pair symmetry");
  CcsdOptions restart_options=solve_options;
  restart_options.initial_t1=
      expand_spatial_rccsd_t1_to_spin_orbital(
          spatial_solution.amplitudes.t1);
  restart_options.initial_t2=
      expand_spatial_rccsd_t2_to_spin_orbital(
          spatial_solution.amplitudes.t2);
  restart_options.completed_iterations=
      spatial_solution.history.back().iteration;
  restart_options.initial_history=spatial_solution.history;
  restart_options.max_iterations=restart_options.completed_iterations+2;
  const auto spatial_restart=solve_spatial_rccsd_reference(
      spatial,restart_options);
  require(spatial_restart.converged,
          "spatial RCCSD reference restart convergence");
  close(spatial_restart.correlation_energy,
        spatial_solution.correlation_energy,1.0e-12,
        "spatial RCCSD reference restart energy");
  CcsdOptions insufficient_spatial_cache_options=
      cached_spatial_residual_options;
  insufficient_spatial_cache_options.wabef_invariant_cache_max_bytes=1;
  bool insufficient_spatial_cache_rejected=false;
  try{
    (void)evaluate_spatial_rccsd_residual(
        spatial,amplitudes,insufficient_spatial_cache_options);
  }catch(const std::runtime_error&){
    insufficient_spatial_cache_rejected=true;
  }
  require(insufficient_spatial_cache_rejected,
          "cached spatial RCCSD invariant-cache memory gate");
  if(global_arrays_runtime_active()){
    const OccupiedPairLayout spatial_layout(2,2);
    const OccupiedPairLayout spin_layout(4,4);
    GlobalArraysPairTensor distributed_spatial_t2(
        spatial_layout,"sp_rccsd_t2");
    distributed_spatial_t2.load_replicated(
        spatial_solution.amplitudes.t2);
    const auto restored_spatial_t2=distributed_spatial_t2.replicate();
    double distributed_error=0.0;
    for(std::size_t index=0;index<restored_spatial_t2.size();++index)
      distributed_error=std::max(
          distributed_error,
          std::abs(restored_spatial_t2.data()[index]-
                   spatial_solution.amplitudes.t2.data()[index]));
    close(distributed_error,0.0,1.0e-14,
          "distributed spatial RCCSD T2 round trip");
    require(16*spatial_layout.global_bytes()==spin_layout.global_bytes(),
            "spatial RCCSD distributed T2 logical reduction");
    const auto memory=distributed_spatial_t2.memory();
    require(memory.global_bytes==spatial_layout.global_bytes(),
            "distributed spatial RCCSD T2 memory accounting");
    if(parallel::world().size>1)
      require(memory.local_bytes<memory.replicated_bytes,
              "distributed spatial RCCSD T2 local memory reduction");
  }

  const auto occupied_labels=expand_spatial_labels({0,1});
  const auto virtual_labels=expand_spatial_labels({0,1});
  const auto audit=audit_ccsd_energy_terms(
      spatial,spin_t1,spin_t2,occupied_labels,virtual_labels,1.0e-6);
  require(audit.total_doubles_terms==spin_t2.size(),
          "CCSD energy-term audit extent");
  require(audit.exact_zero_terms>0,
          "CCSD energy-term audit spin-forbidden zeros");
  close(audit.singles_energy_hartree+audit.doubles_energy_hartree,
        ccsd_energy(spin_fock,spin_eri,spin_t1,spin_t2,4),1.0e-14,
        "CCSD energy-term audit reconstruction");
  require(audit.rows.size()==2,
          "CCSD energy-term audit criteria");
  require(audit.rows[0].criterion=="certified_absolute"&&
              audit.rows[0].omitted_absolute_bound_hartree<=1.0e-6&&
              std::abs(audit.rows[0].omitted_signed_energy_hartree)<=
                  audit.rows[0].omitted_absolute_bound_hartree,
          "CCSD energy-term certified audit bound");
  require(audit.rows[1].criterion=="observed_signed"&&
              std::abs(audit.rows[1].omitted_signed_energy_hartree)<=1.0e-6,
          "CCSD energy-term observed audit error");
  for(const auto& row:audit.rows){
    const auto classified=std::accumulate(
        row.omitted_many_body_terms.begin(),
        row.omitted_many_body_terms.end(),std::size_t{0});
    require(classified==row.omitted_terms,
            "CCSD energy-term audit M-class count");
  }

  const auto audit_path=std::filesystem::temp_directory_path()/
                        ("ccsd_energy_term_audit_"+
                         std::to_string(parallel::world().rank)+".csv");
  std::error_code ignored;
  std::filesystem::remove(audit_path,ignored);
  write_ccsd_energy_term_audit_csv(audit_path,audit);
  std::ifstream audit_input(audit_path);
  std::ostringstream audit_text;audit_text<<audit_input.rdbuf();
  require(audit_text.str().find("certified_absolute")!=std::string::npos&&
              audit_text.str().find("reconstructed_ccsd_energy_hartree")!=
                  std::string::npos,
          "CCSD energy-term audit CSV content");
  bool audit_overwrite_rejected=false;
  try{write_ccsd_energy_term_audit_csv(audit_path,audit);}
  catch(const std::runtime_error&){audit_overwrite_rejected=true;}
  require(audit_overwrite_rejected,
          "CCSD energy-term audit CSV overwrite rejection");
  std::filesystem::remove(audit_path,ignored);
  if(parallel::world().rank==0)
    std::cout<<"Spatial RCCSD admission tests passed\n";
}
void test_all_electron(){Matrix id({2,2});id(0,0)=1;id(1,1)=1;Matrix core,occ({2,1}),virt({2,1}),h({2,2});occ(0,0)=1;virt(1,0)=1;h(0,0)=-1;h(1,1)=.5;Tensor4 eri({2,2,2,2});FrozenCoreInput in{core,occ,virt,id,h,eri,0,-2,true};auto x=build_frozen_core_hamiltonian(in);require(x.core_indices.empty(),"all-electron core indices");require(x.nocc==2&&x.nvirt==2,"all-electron dimensions");close(x.reference_energy,-2,1e-12,"all-electron reference energy");}
void test_water_zmatrix(){const auto z=read_water_zmatrix(std::string(CCSD_SOURCE_DIR)+"/examples/water/nwchem_qa_h2o_ccpvdz.zmat");require(z.has_c2v_symmetry(),"water C2v detection");require(z.point_group()=="C2v","water point group");const auto atoms=z.cartesian_atoms();require(atoms.size()==3,"water atom count");close(atoms[1].position_bohr[0]+atoms[2].position_bohr[0],0,1e-14,"water C2v x coordinates");close(atoms[1].position_bohr[2]-atoms[2].position_bohr[2],0,1e-14,"water C2v z coordinates");const auto bond=std::hypot(atoms[1].position_bohr[0],atoms[1].position_bohr[2])/angstrom_to_bohr;close(bond,z.oh1_angstrom,1e-14,"water Z-matrix bond");const auto dot=atoms[1].position_bohr[0]*atoms[2].position_bohr[0]+atoms[1].position_bohr[2]*atoms[2].position_bohr[2];const auto angle=std::acos(dot/(bond*bond*angstrom_to_bohr*angstrom_to_bohr))*180.0/std::acos(-1.0);close(angle,z.hoh_degrees,1e-12,"water Z-matrix angle");}
void test_many_body_limits(){Matrix f({4,4}),s({2,2});Tensor4 g({4,4,4,4}),t({2,2,2,2});g(0,1,2,3)=4;t(0,1,0,1)=1;const auto c=partition_ccsd(f,g,s,t,2,{0,1},{2,3});close(c.many_body_classes.at("M4"),1,1e-14,"CCSD M4 class");close(c.sum,c.parent_energy,1e-14,"CCSD M4 sum");Tensor6 e({3,3,3,3,3,3});e(0,1,2,0,1,2)=2;const auto triples=partition_triples(e,{0,1,2},{3,4,5});close(triples.many_body_classes.at("M6"),2,1e-14,"triples M6 class");close(triples.sum,triples.parent_energy,1e-14,"triples M6 sum");const auto monomer=partition_triples_single_monomer(-0.005,{0,0,0},{0,0,0});close(monomer.many_body_classes.at("M1"),-0.005,0.0,"streamed triples monomer M1");close(monomer.sum_error,0.0,0.0,"streamed triples monomer closure");bool screened_rejected=false;try{(void)partition_triples_single_monomer(-0.005,{0,0,0},{0,0,0},{1e-8,1e-6});}catch(const std::invalid_argument&){screened_rejected=true;}require(screened_rejected,"streamed triples monomer term-screen rejection");}
void test_numeric_ccsd_partition() {
  constexpr std::size_t o=3, v=4;
  Matrix f({o+v,o+v}), s({o,v});
  Tensor4 g({o+v,o+v,o+v,o+v}), t({o,o,v,v});
  const std::vector<int> vl{0,99,105,203};
  for (bool unassigned : {false,true}) {
    const std::vector<int> ol{0,unassigned ? -1 : 0,99};
    for (double threshold : {0.0,0.02}) {
      std::array<double,4> energy{};
      std::array<std::size_t,4> counts{};
      double parent=0, omitted=0, bound=0, unknown=0, detail=0;
      std::size_t omitted_count=0, unknown_count=0;
      // Singles are zero but still counted, or screened, exactly as before.
      for (auto hole : ol) for (auto particle : vl) {
        if (threshold>0) ++omitted_count;
        else if (hole<0) ++unknown_count;
        else ++counts[std::set<int>{hole,particle}.size()-1];
      }
      for (std::size_t i=0;i<o;++i) for (std::size_t j=0;j<o;++j)
        for (std::size_t a=0;a<v;++a) for (std::size_t b=0;b<v;++b) {
          g(i,j,o+a,o+b)=1;
          t(i,j,a,b)=(static_cast<int>(i*48+j*16+a*4+b)%17-8)/64.0;
          const double x=0.25*t(i,j,a,b);
          parent+=x;
          if (threshold>0 && std::abs(x)<=threshold) {
            omitted+=x; bound+=std::abs(x); ++omitted_count;
          } else if (ol[i]<0 || ol[j]<0) {
            unknown+=x; ++unknown_count;
          } else {
            const auto order=std::set<int>{ol[i],ol[j],vl[a],vl[b]}.size();
            energy[order-1]+=x; ++counts[order-1];
            if (((ol[i]==0 && ol[j]==99)||(ol[i]==99 && ol[j]==0)) &&
                ((a==2 && b==3)||(a==3 && b==2))) detail+=x;
          }
        }
      const auto p=partition_ccsd(f,g,s,t,o,ol,vl,{threshold,10.0});
      close(p.parent_energy,parent,1e-14,"fused independent CCSD total");
      close(p.omitted_energy,omitted,1e-14,"numeric partition screened energy");
      close(p.omitted_absolute_bound,bound,1e-14,"numeric partition screen bound");
      close(p.unassigned_energy,unknown,1e-14,"numeric partition unassigned");
      require(p.omitted_terms==omitted_count && p.unassigned_terms==unknown_count,
              "numeric partition omitted/unassigned counts");
      for (std::size_t m=0;m<4;++m) {
        const auto key="M"+std::to_string(m+1);
        close(p.many_body_classes.contains(key)?p.many_body_classes.at(key):0,
              energy[m],1e-14,"numeric partition scalar nM oracle");
        require((p.many_body_term_counts.contains(key)?p.many_body_term_counts.at(key):0)==counts[m],
                "numeric partition scalar class counts");
      }
      const std::string key="M4|h:0,99,|p:105,203,|q:0:1,99:1,105:-1,203:-1,|singly_ionic";
      close(p.classes.contains(key)?p.classes.at(key):0,detail,1e-14,
            "numeric partition preserves detailed CSV key");
      close(p.sum_error,0,1e-14,"numeric partition closure");
    }
  }
}

void test_singles_partition(){test_numeric_ccsd_partition();Matrix f({2,2}),s({1,1});Tensor4 g({2,2,2,2}),t({1,1,1,1});f(0,1)=0.5;s(0,0)=0.2;const auto p=partition_ccsd(f,g,s,t,1,{0},{1});close(p.singles_fia_t1,0.1,1e-15,"CCSD singles value");close(p.many_body_classes.at("M2"),0.1,1e-15,"CCSD singles M2 class");close(p.sum,p.parent_energy,1e-15,"CCSD singles partition sum");}
void test_cpp_energy_known_case(){
  constexpr std::size_t occupied=2,orbitals=4;
  Matrix fock({orbitals,orbitals}),t1({occupied,orbitals-occupied});
  Tensor4 eri({orbitals,orbitals,orbitals,orbitals});
  Tensor4 t2({occupied,occupied,orbitals-occupied,orbitals-occupied});
  fock(0,2)=0.3;fock(1,3)=-0.2;t1(0,0)=0.04;t1(1,1)=-0.03;
  eri(0,1,2,3)=0.5;t2(0,1,0,1)=0.08;
  close(ccsd_energy(fock,eri,t1,t2,occupied),0.0277,1.0e-14,
        "C++ CCSD energy known case");
}
void test_screened_partition_and_csv(){
  Tensor6 contributions({1,1,1,2,2,2});
  contributions(0,0,0,0,0,0)=2.0;
  contributions(0,0,0,1,1,1)=0.25;
  contributions(0,0,0,0,0,1)=1.0e-8;
  const PartitionScreeningOptions screening{1.0e-7,1.0e-6};
  const auto partition=partition_triples(contributions,{0},{0,-1},screening);
  close(partition.sum,2.0,1.0e-15,"screened retained energy");
  close(partition.unassigned_energy,0.25,1.0e-15,
        "unassigned retained energy");
  close(partition.omitted_energy,1.0e-8,1.0e-18,"screened omitted energy");
  close(partition.omitted_absolute_bound,1.0e-8,1.0e-18,
        "screened omitted bound");
  close(partition.sum+partition.unassigned_energy+partition.omitted_energy,
        partition.parent_energy,1.0e-15,
        "screened reconstruction");
  require(partition.assigned_terms==1&&partition.unassigned_terms==1&&
              partition.retained_terms==2&&partition.omitted_terms==6,
          "screened retained/omitted counts");
  bool budget_rejected=false;
  try{(void)partition_triples(contributions,{0},{0,-1},{1.0e-7,1.0e-9});}
  catch(const std::runtime_error&){budget_rejected=true;}
  require(budget_rejected,"screened omitted-budget rejection");

  const auto stem="localized_cc_phase12_"+
                  std::to_string(parallel::world().rank);
  const auto energy_path=std::filesystem::temp_directory_path()/
                         (stem+".energy.csv");
  const auto orbital_path=std::filesystem::temp_directory_path()/
                          (stem+".orbitals.csv");
  std::error_code ignored;
  std::filesystem::remove(energy_path,ignored);
  std::filesystem::remove(orbital_path,ignored);
  write_partition_csv(energy_path,partition,partition);
  write_orbital_assignments_csv(orbital_path,"fm4",0.8,{0},{0.95},
                                {0,-1},{0.81,0.79});
  std::ifstream energy_input(energy_path);
  std::ostringstream energy_text;energy_text<<energy_input.rdbuf();
  require(energy_text.str().find("CCSD,many_body,M1")!=std::string::npos,
          "Phase 12 energy CSV content");
  std::ifstream orbital_input(orbital_path);
  std::ostringstream orbital_text;orbital_text<<orbital_input.rdbuf();
  require(orbital_text.str().find("fm4,virtual,1,unassigned,")!=
              std::string::npos&&
              orbital_text.str().find(",false")!=std::string::npos,
          "Phase 12 orbital CSV content");
  require(expand_spatial_labels({-1})==std::vector<int>({-1,-1}),
          "unassigned spatial-label expansion");
  bool invalid_label_rejected=false;
  try{(void)partition_triples(contributions,{0},{0,-2},screening);}
  catch(const std::invalid_argument&){invalid_label_rejected=true;}
  require(invalid_label_rejected,"invalid monomer-label rejection");
  bool inconsistent_assignment_rejected=false;
  const auto inconsistent_path=std::filesystem::temp_directory_path()/
                               (stem+".inconsistent.csv");
  std::filesystem::remove(inconsistent_path,ignored);
  try{write_orbital_assignments_csv(inconsistent_path,"fm4",0.8,{0},{0.95},
                                    {-1},{0.95});}
  catch(const std::invalid_argument&){inconsistent_assignment_rejected=true;}
  require(inconsistent_assignment_rejected,
          "assignment confidence/label consistency rejection");
  bool overwrite_rejected=false;
  try{write_partition_csv(energy_path,partition,partition);}
  catch(const std::runtime_error&){overwrite_rejected=true;}
  require(overwrite_rejected,"Phase 12 CSV overwrite rejection");
  std::filesystem::remove(energy_path,ignored);
  std::filesystem::remove(orbital_path,ignored);
}
void test_checkpoint_roundtrip(){
  CalculationCheckpoint checkpoint;
  checkpoint.fingerprint="water|avdz|atomic|pm";
  checkpoint.rhf_restart_density=Matrix({2,2});
  (*checkpoint.rhf_restart_density)(0,0)=1.5;
  RhfCheckpointState rhf;
  rhf.total_energy=-76.0;
  rhf.overlap=Matrix({2,2});rhf.overlap(0,0)=1;rhf.overlap(1,1)=1;
  rhf.core_hamiltonian=Matrix({2,2});rhf.density=Matrix({2,2});
  rhf.fock=Matrix({2,2});rhf.coefficients=Matrix({2,2});
  rhf.orbital_energies={-1.0,0.5};rhf.occupations={2.0,0.0};
  checkpoint.rhf=std::move(rhf);
  CcsdCheckpointState ccsd;
  ccsd.completed_iterations=3;ccsd.correlation_energy=-0.2;
  ccsd.t1=Matrix({1,1});ccsd.t1(0,0)=0.01;
  ccsd.t2=Tensor4({1,1,1,1});ccsd.t2(0,0,0,0)=0.02;
  ccsd.history.push_back({3,-0.2,-1e-8,1e-9,2e-9});
  checkpoint.canonical_ccsd=std::move(ccsd);
  checkpoint.basis_projection=BasisProjectionCheckpointState{
      "small.chk","source-fingerprint","avdz","avtz",2,3,3,
      0.9,1e-13,2.0,true,true,false};
  checkpoint.localized_streamed_triples=CanonicalTriplesCheckpointState{
      {-0.01},{-0.009},{1U}};
  checkpoint.localized_laplace_triples=LaplaceTriplesCheckpointState{
      2,{-0.01,-0.002,0.0,0.0,0.0,0.0,
         -0.003,-0.0004,0.0,0.0,0.0,0.0},{1U,1U},
      "scaled-gauss-laguerre"};
  checkpoint.localized_sparse_ccsd=SparseCcsdCheckpointState{
      true,3,2,-0.25,1.0e-11,2.0e-10,Matrix({2,3}),
      {0,1},{0,0,1},{2,3},{1.0,2.0,3.0,4.0,5.0},
      {{3,-0.25,1.0e-12,1.0e-11,2.0e-10}}};
  checkpoint.occupied_localization_optimization=
      LocalizationOptimizationCheckpointState{
          "pm","occupied",false,Matrix({2,2}),0.125,7,4,3,
          {1.25,1.5}};
  checkpoint.occupied_localization_optimization->rotation(0,0)=1.0;
  checkpoint.occupied_localization_optimization->rotation(1,1)=1.0;
  const auto path=std::filesystem::temp_directory_path()/
                  ("localized_cc_checkpoint_"+std::to_string(parallel::world().rank)+".bin");
  std::error_code ignored;std::filesystem::remove(path,ignored);
  write_checkpoint_transactional(path,checkpoint);
  const auto restored=read_checkpoint(path,checkpoint.fingerprint);
  require(restored.rhf.has_value()&&restored.canonical_ccsd.has_value(),
          "checkpoint state presence");
  require(restored.rhf_restart_density.has_value(),
          "checkpoint RHF restart density presence");
  close((*restored.rhf_restart_density)(0,0),1.5,0.0,
        "checkpoint RHF restart density");
  close(restored.rhf->total_energy,-76.0,0.0,"checkpoint RHF energy");
  close(restored.canonical_ccsd->t2(0,0,0,0),0.02,0.0,
        "checkpoint T2");
  require(restored.basis_projection.has_value()&&
              restored.basis_projection->source_basis=="avdz"&&
              restored.basis_projection->target_basis=="avtz"&&
              restored.basis_projection->canonical_amplitudes_projected,
          "checkpoint basis-projection provenance");
  require(restored.localized_streamed_triples.has_value()&&
              restored.localized_streamed_triples->completed_tasks==
                  std::vector<std::uint8_t>{1U},
          "checkpoint localized streamed triples");
  require(restored.localized_laplace_triples.has_value()&&
              restored.localized_laplace_triples->quadrature_points==2&&
              restored.localized_laplace_triples->completed_points==
                  std::vector<std::uint8_t>{1U,1U}&&
              restored.localized_laplace_triples->quadrature_scheme==
                  "scaled-gauss-laguerre",
          "checkpoint localized Laplace triples");
  require(restored.localized_sparse_ccsd.has_value()&&
              restored.localized_sparse_ccsd->maximum_monomers==2&&
              restored.localized_sparse_ccsd->t2_block_sizes==
                  std::vector<std::uint64_t>({2,3})&&
              restored.localized_sparse_ccsd->t2_block_values==
                  std::vector<double>({1.0,2.0,3.0,4.0,5.0}),
          "checkpoint sparse CCSD state");
  require(restored.occupied_localization_optimization.has_value()&&
              restored.occupied_localization_optimization->space=="occupied"&&
              restored.occupied_localization_optimization->
                      completed_iterations==7&&
              restored.occupied_localization_optimization->
                      objective_history==std::vector<double>{1.25,1.5},
          "checkpoint occupied-localization optimizer state");
  auto version_seven_checkpoint=checkpoint;
  version_seven_checkpoint.localized_laplace_triples.reset();
  const auto version_seven_path=path.string()+".v7";
  std::filesystem::remove(version_seven_path,ignored);
  write_checkpoint_transactional(version_seven_path,
                                 version_seven_checkpoint);
  {
    std::fstream version_file(version_seven_path,
                              std::ios::binary|std::ios::in|std::ios::out);
    require(static_cast<bool>(version_file),"open checkpoint v7 fixture");
    constexpr std::uint32_t version_seven=7;
    version_file.seekp(8);
    version_file.write(reinterpret_cast<const char*>(&version_seven),
                       sizeof(version_seven));
    require(static_cast<bool>(version_file),"write checkpoint v7 fixture");
  }
  const auto restored_v7=read_checkpoint(version_seven_path,
                                         checkpoint.fingerprint);
  require(restored_v7.localized_sparse_ccsd.has_value()&&
              !restored_v7.localized_laplace_triples.has_value(),
          "checkpoint v7 pre-triples backward compatibility");
  auto version_two_checkpoint=checkpoint;
  version_two_checkpoint.basis_projection.reset();
  version_two_checkpoint.localized_streamed_triples.reset();
  version_two_checkpoint.occupied_localization_optimization.reset();
  version_two_checkpoint.virtual_localization_optimization.reset();
  version_two_checkpoint.localized_laplace_triples.reset();
  version_two_checkpoint.localized_sparse_ccsd.reset();
  const auto version_two_path=path.string()+".v2";
  std::filesystem::remove(version_two_path,ignored);
  write_checkpoint_transactional(version_two_path,version_two_checkpoint);
  {
    std::fstream version_file(version_two_path,
                              std::ios::binary|std::ios::in|std::ios::out);
    require(static_cast<bool>(version_file),"open checkpoint v2 fixture");
    constexpr std::uint32_t version_two=2;
    version_file.seekp(8);
    version_file.write(reinterpret_cast<const char*>(&version_two),
                       sizeof(version_two));
    require(static_cast<bool>(version_file),"write checkpoint v2 fixture");
  }
  const auto version_two_size=std::filesystem::file_size(version_two_path);
  require(version_two_size>5,"checkpoint v2 fixture size");
  std::filesystem::resize_file(version_two_path,version_two_size-6);
  const auto restored_v2=read_checkpoint(version_two_path,
                                         checkpoint.fingerprint);
  require(restored_v2.rhf.has_value()&&
              restored_v2.canonical_ccsd.has_value()&&
              !restored_v2.basis_projection.has_value()&&
              !restored_v2.localized_streamed_triples.has_value()&&
              !restored_v2.localized_laplace_triples.has_value()&&
              !restored_v2.localized_sparse_ccsd.has_value(),
          "checkpoint v2 backward compatibility");
  bool rejected=false;try{(void)read_checkpoint(path,"wrong");}
  catch(const std::runtime_error&){rejected=true;}
  require(rejected,"checkpoint fingerprint rejection");
  std::filesystem::remove(path,ignored);
  std::filesystem::remove(version_seven_path,ignored);
  std::filesystem::remove(version_two_path,ignored);
}
void test_converged_ccsd_checkpoint_restore(){
  CcsdCheckpointState saved;
  saved.localized=true;
  saved.converged=true;
  saved.completed_iterations=7;
  saved.correlation_energy=-0.2;
  saved.r1_norm=1.0e-10;
  saved.r2_norm=2.0e-10;
  saved.t1=Matrix({1,1});
  saved.t1(0,0)=0.01;
  saved.t2=Tensor4({1,1,1,1});
  saved.history.push_back(
      {7,-0.2,5.0e-12,saved.r1_norm,saved.r2_norm});
  const auto restored=restore_converged_ccsd(
      saved,true,1,1,1.0e-11,1.0e-9,
      CcsdResidualMetric::l2,
      CcsdAmplitudeBackend::replicated,CcsdWabefBackend::cuda);
  require(restored.converged&&restored.history.back().iteration==7,
          "converged CCSD checkpoint direct restore");
  close(restored.correlation_energy,-0.2,0.0,
        "converged CCSD checkpoint restored energy");
  require(restored.amplitude_backend==CcsdAmplitudeBackend::replicated&&
              restored.wabef_backend==CcsdWabefBackend::cuda,
          "converged CCSD checkpoint restored backend metadata");

  auto failed=saved;
  failed.r2_norm=2.0e-9;
  bool tolerance_rejected=false;
  try{(void)restore_converged_ccsd(
      failed,true,1,1,1.0e-11,1.0e-9,
      CcsdResidualMetric::l2,
      CcsdAmplitudeBackend::replicated,CcsdWabefBackend::cpu);}
  catch(const std::runtime_error&){tolerance_rejected=true;}
  require(tolerance_rejected,
          "converged CCSD checkpoint current-tolerance rejection");
  bool refinement_required=false;
  try { validate_converged_ccsd_checkpoint(failed,true,1,1,1e-11,1e-9,
                                          CcsdResidualMetric::l2); }
  catch (const CcsdCheckpointConvergenceError&) { refinement_required=true; }
  require(refinement_required,"valid loose checkpoint requests refinement");
  failed.t2(0,0,0,0)=0.01;
  bool corrupt_rejected=false;
  try { validate_converged_ccsd_checkpoint(failed,true,1,1,1e-11,1e-9,
                                          CcsdResidualMetric::l2); }
  catch (const CcsdCheckpointConvergenceError&) {
    throw std::runtime_error("corrupt checkpoint must not be treated as refinable");
  } catch (const std::runtime_error&) { corrupt_rejected=true; }
  require(corrupt_rejected,"invalid amplitudes rejected before tolerance recovery");

  CcsdCheckpointState rms_saved;
  rms_saved.localized=true;
  rms_saved.converged=true;
  rms_saved.completed_iterations=5;
  rms_saved.correlation_energy=-0.1;
  rms_saved.r1_norm=1.1e-7;
  rms_saved.r2_norm=1.5e-7;
  rms_saved.t1=Matrix({1,2});
  rms_saved.t2=Tensor4({1,1,2,2});
  rms_saved.history.push_back(
      {5,-0.1,5.0e-8,rms_saved.r1_norm,rms_saved.r2_norm});
  const auto rms_restored=restore_converged_ccsd(
      rms_saved,true,1,2,1.0e-7,1.0e-7,CcsdResidualMetric::rms,
      CcsdAmplitudeBackend::replicated,CcsdWabefBackend::cpu);
  require(rms_restored.converged,
          "RMS-converged CCSD checkpoint direct restore");
  require(ccsd_residual_measure(rms_saved.r1_norm,rms_saved.t1.size(),
                                CcsdResidualMetric::rms)<1.0e-7&&
              ccsd_residual_measure(rms_saved.r2_norm,rms_saved.t2.size(),
                                    CcsdResidualMetric::rms)<1.0e-7,
          "CCSD RMS residual normalization");
}
void test_tce_output_parser(){std::istringstream input(" Total SCF energy = -76.0\n CCSD correlation energy / hartree = -0.2\n CCSD[T]  correction energy / hartree = -0.02\n CCSD(T)  correction energy / hartree = -0.01\n");const auto records=parse_tce_energy_records(input);require(records.size()==1,"TCE record count");close(records[0].rhf_hartree,-76.0,1e-15,"TCE RHF parse");close(records[0].ccsd_correlation_hartree,-0.2,1e-15,"TCE CCSD parse");close(records[0].triples_parenthetical_hartree,-0.01,1e-15,"TCE parenthetical triples parse");const double total_tolerance=2.0*std::numeric_limits<double>::epsilon()*76.21;close(records[0].ccsdt_total_hartree(),-76.21,total_tolerance,"TCE total parse");}
void test_benchmark_output_parser(){
  std::istringstream input(
      "RHF                       -76.0 Eh\n"
      "canonical CCSD corr       -0.2 Eh\n"
      "canonical (T)             -0.01 Eh\n"
      "canonical CCSD(T)         -76.21 Eh\n"
      "wall time                   12.5 s\n"
      "Maximum resident set size (kbytes): 4321\n");
  const auto record=parse_ccsdt_benchmark_output(input);
  require(record.rhf_hartree&&record.ccsd_correlation_hartree&&
              record.triples_parenthetical_hartree&&
              record.total_ccsdt_hartree&&record.wall_seconds&&
              record.maximum_rss_kib,"benchmark output fields");
  close(*record.total_ccsdt_hartree,-76.21,1e-15,
        "benchmark total parse");
  close(*record.wall_seconds,12.5,0.0,"benchmark wall parse");
  close(*record.maximum_rss_kib,4321.0,0.0,"benchmark RSS parse");
  std::istringstream localized_input(
      "RHF                       -76.0 Eh\n"
      "localized CCSD corr       -0.2 Eh\n"
      "localized (T)             -0.01 Eh\n"
      "localized CCSD(T)         -76.21 Eh\n");
  const auto localized_record=
      parse_ccsdt_benchmark_output(localized_input);
  require(localized_record.ccsd_correlation_hartree&&
              localized_record.triples_parenthetical_hartree&&
              localized_record.total_ccsdt_hartree,
          "localized benchmark output fields");
  close(*localized_record.total_ccsdt_hartree,-76.21,1e-15,
        "localized benchmark total parse");
  std::istringstream elapsed_only(
      "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02.50\n");
  const auto elapsed_record=parse_ccsdt_benchmark_output(elapsed_only);
  require(elapsed_record.wall_seconds.has_value(),
          "benchmark elapsed wall presence");
  close(*elapsed_record.wall_seconds,62.5,0.0,"benchmark elapsed wall parse");
}
void test_operator(){Matrix oo({3,3}),vv({3,3});oo(0,0)=-1;oo(1,1)=-.8;oo(2,2)=-.6;vv(0,0)=.4;vv(1,1)=.7;vv(2,2)=1;Tensor6 x({3,3,3,3,3,3});x(0,1,2,0,1,2)=2;auto y=triples_fock_action(x,oo,vv);close(y(0,1,2,0,1,2),9.0,1e-14,"canonical A3");}
Tensor4 make_eri(std::size_t n){Tensor4 r({n,n,n,n}),g({n,n,n,n});for(std::size_t p=0;p<n;++p)for(std::size_t q=0;q<n;++q)for(std::size_t s=0;s<n;++s)for(std::size_t t=0;t<n;++t)r(p,q,s,t)=1e-3*std::sin(static_cast<double>(1+13*p+7*q+5*s+3*t));for(std::size_t p=0;p<n;++p)for(std::size_t q=0;q<n;++q)for(std::size_t s=0;s<n;++s)for(std::size_t t=0;t<n;++t){auto a=.25*(r(p,q,s,t)-r(q,p,s,t)-r(p,q,t,s)+r(q,p,t,s));auto b=.25*(r(s,t,p,q)-r(t,s,p,q)-r(s,t,q,p)+r(t,s,q,p));g(p,q,s,t)=.5*(a+b);}return g;}
struct DifferenceNorms{double infinity{},two{},reference_infinity{},reference_two{},candidate_infinity{},candidate_two{};};
DifferenceNorms difference_norms(const std::vector<double>& reference,const std::vector<double>& candidate){
  require(reference.size()==candidate.size(),"residual-oracle norm dimensions");
  DifferenceNorms result;
  double difference_square=0.0,reference_square=0.0,candidate_square=0.0;
  for(std::size_t index=0;index<reference.size();++index){
    const auto difference=reference[index]-candidate[index];
    result.infinity=std::max(result.infinity,std::abs(difference));
    result.reference_infinity=std::max(result.reference_infinity,std::abs(reference[index]));
    result.candidate_infinity=std::max(result.candidate_infinity,std::abs(candidate[index]));
    difference_square+=difference*difference;
    reference_square+=reference[index]*reference[index];
    candidate_square+=candidate[index]*candidate[index];
  }
  result.two=std::sqrt(difference_square);
  result.reference_two=std::sqrt(reference_square);
  result.candidate_two=std::sqrt(candidate_square);
  return result;
}
void require_residual_oracle_gate(const DifferenceNorms& norms,const char* message){
  constexpr double absolute_tolerance=1.0e-12;
  constexpr double relative_tolerance=1.0e-12;
  const auto infinity_limit=absolute_tolerance+relative_tolerance*
      std::max(norms.reference_infinity,norms.candidate_infinity);
  const auto two_limit=absolute_tolerance+relative_tolerance*
      std::max(norms.reference_two,norms.candidate_two);
  if(norms.infinity>infinity_limit||norms.two>two_limit){
    std::cerr<<message<<": infinity="<<norms.infinity
             <<" limit="<<infinity_limit<<" two="<<norms.two
             <<" limit="<<two_limit<<'\n';
    throw std::runtime_error(message);
  }
}
void test_random_amplitude_residual_oracle(){
  constexpr std::size_t occupied=3,orbitals=6,virtuals=3;
  Matrix fock({orbitals,orbitals});
  fock(0,0)=-1.3;fock(1,1)=-1.0;fock(2,2)=-0.8;
  fock(3,3)=0.3;fock(4,4)=0.6;fock(5,5)=1.1;
  fock(0,1)=fock(1,0)=2.0e-2;
  fock(1,2)=fock(2,1)=-1.5e-2;
  fock(3,4)=fock(4,3)=1.0e-2;
  fock(4,5)=fock(5,4)=-1.2e-2;
  const auto eri=make_eri(orbitals);
  std::mt19937_64 generator(0x5a17c0deULL);
  std::uniform_real_distribution<double> distribution(-1.0,1.0);
  for(const double scale:{1.0e-4,1.0e-2,1.0e-1}){
    Matrix t1({occupied,virtuals});
    for(auto& value:t1.data())value=scale*distribution(generator);
    Tensor4 unconstrained({occupied,occupied,virtuals,virtuals});
    for(auto& value:unconstrained.data())value=scale*distribution(generator);
    Tensor4 t2({occupied,occupied,virtuals,virtuals});
    for(std::size_t i=0;i<occupied;++i)
      for(std::size_t j=0;j<occupied;++j)
        for(std::size_t a=0;a<virtuals;++a)
          for(std::size_t b=0;b<virtuals;++b)
            t2(i,j,a,b)=0.25*(unconstrained(i,j,a,b)-
                unconstrained(j,i,a,b)-unconstrained(i,j,b,a)+
                unconstrained(j,i,b,a));
    const auto antisymmetry_error=t2_antisymmetry_error(t2);
    require(antisymmetry_error<=64.0*std::numeric_limits<double>::epsilon()*scale,
            "random T2 explicit antisymmetrization");
    CcsdOptions dense_options;
    dense_options.amplitude_backend=CcsdAmplitudeBackend::replicated;
    dense_options.wabef_backend=CcsdWabefBackend::cpu;
    dense_options.use_tiled_doubles=false;
    const auto dense=evaluate_ccsd_residual(
        fock,eri,t1,t2,occupied,dense_options);
    const auto compare=[&](const CcsdOptions& candidate_options,
                           const char* label){
      const auto candidate=evaluate_ccsd_residual(
          fock,eri,t1,t2,occupied,candidate_options);
      const auto r1=difference_norms(
          dense.r1.data(),candidate.r1.data());
      const auto r2=difference_norms(
          dense.r2.data(),candidate.r2.data());
      require_residual_oracle_gate(r1,
          "random-amplitude optimized CCSD R1 oracle");
      require_residual_oracle_gate(r2,
          "random-amplitude optimized CCSD R2 oracle");
      if(parallel::world().rank==0){
        std::cout.setf(std::ios::scientific);
        std::cout<<"CCSD residual oracle "<<label<<" scale "<<scale
                 <<" T2 antisymmetry "<<antisymmetry_error
                 <<" R1 infinity "<<r1.infinity<<" R1 two "<<r1.two
                 <<" R2 infinity "<<r2.infinity<<" R2 two "<<r2.two<<'\n';
      }
    };
    auto tiled_wabef=dense_options;
    tiled_wabef.wabef_backend=CcsdWabefBackend::tiled_cpu;
    tiled_wabef.wabef_tile_max_bytes=128;
    compare(tiled_wabef,"tiled-wabef");
    auto cached_wabef=tiled_wabef;
    cached_wabef.wabef_packing_backend=
        CcsdWabefPackingBackend::cached_invariants;
    cached_wabef.wabef_invariant_cache_max_bytes=1024U*1024U;
    compare(cached_wabef,"cached-invariant-wabef");
    auto symmetric_wabef=cached_wabef;
    symmetric_wabef.use_wabef_pair_symmetry=true;
    symmetric_wabef.wabef_tile_max_bytes=1;
    compare(symmetric_wabef,"pair-symmetric-panel-wabef");
    symmetric_wabef.wabef_tile_max_bytes=1024U*1024U;
    compare(symmetric_wabef,"pair-symmetric-factorized-wabef");
    const auto counted=evaluate_ccsd_residual(
        fock,eri,t1,t2,occupied,symmetric_wabef);
    std::vector<double> tasks{
        static_cast<double>(counted.timing.wabef_output_tasks)};
    parallel::sum_in_place(tasks);
    close(tasks[0],static_cast<double>(virtuals*(virtuals-1)/2),0.0,
          "unique W_abef task ownership including idle ranks");
    for(const bool incompatible_packing:{false,true}){
      auto invalid=symmetric_wabef;
      if(incompatible_packing)
        invalid.wabef_packing_backend=CcsdWabefPackingBackend::direct;
      else
        invalid.projection_mask=std::make_shared<CcsdProjectionMask>();
      bool rejected=false;
      try{(void)evaluate_ccsd_residual(fock,eri,t1,t2,occupied,invalid);}
      catch(const std::invalid_argument&){rejected=true;}
      require(rejected,"pair symmetry incompatible options fail closed");
    }
    auto tiled_intermediates=dense_options;
    tiled_intermediates.use_tiled_intermediates=true;
    compare(tiled_intermediates,"tiled-intermediates");
    auto tiled_doubles=dense_options;
    tiled_doubles.use_tiled_doubles=true;
    compare(tiled_doubles,"tiled-doubles");
    auto tiled_both=tiled_wabef;
    tiled_both.use_tiled_intermediates=true;
    tiled_both.use_tiled_doubles=true;
    compare(tiled_both,"tiled-all");
  }
}
void test_incremental_diis_ccsd(){
  constexpr std::size_t occupied=3,orbitals=6;
  Matrix fock({orbitals,orbitals});
  fock(0,0)=-1.2;fock(1,1)=-1.0;fock(2,2)=-0.8;
  fock(3,3)=0.4;fock(4,4)=0.7;fock(5,5)=1.0;
  auto eri=make_eri(orbitals);
  for(auto& value:eri.data())value*=20.0;
  CcsdOptions recomputed_options;
  recomputed_options.amplitude_backend=CcsdAmplitudeBackend::replicated;
  recomputed_options.use_incremental_diis_gram=false;
  recomputed_options.max_iterations=300;
  recomputed_options.energy_tolerance=1.0e-14;
  recomputed_options.residual_tolerance=1.0e-12;
  recomputed_options.diis_start=1;
  const auto recomputed=solve_ccsd(
      fock,eri,occupied,recomputed_options);
  require(recomputed.converged,"recomputed-Gram CCSD convergence");
  auto incremental_options=recomputed_options;
  incremental_options.use_incremental_diis_gram=true;
  const auto incremental=solve_ccsd(
      fock,eri,occupied,incremental_options);
  require(incremental.converged,"incremental-Gram CCSD convergence");
  close(incremental.correlation_energy,recomputed.correlation_energy,1.0e-12,
        "incremental/recomputed DIIS CCSD energy");
  require(incremental.history.size()==recomputed.history.size(),
          "incremental/recomputed DIIS iteration count");
  for(std::size_t index=0;index<recomputed.t1.size();++index)
    close(incremental.t1.data()[index],recomputed.t1.data()[index],1.0e-12,
          "incremental/recomputed DIIS T1");
  for(std::size_t index=0;index<recomputed.t2.size();++index)
    close(incremental.t2.data()[index],recomputed.t2.data()[index],1.0e-12,
          "incremental/recomputed DIIS T2");
  require(incremental.cumulative_timing.diis_error_dot_products>0&&
              incremental.cumulative_timing.diis_error_dot_products<
                  recomputed.cumulative_timing.diis_error_dot_products,
          "incremental CCSD DIIS Gram-product reduction");
  require(incremental.cumulative_timing
                  .diis_history_vector_accumulations==
              recomputed.cumulative_timing
                  .diis_history_vector_accumulations,
          "incremental/recomputed DIIS history accumulation count");
  auto root_diis_options=incremental_options;
  root_diis_options.use_root_replicated_diis=true;
  const auto root_diis=solve_ccsd(fock,eri,occupied,root_diis_options);
  require(root_diis.converged,"rank-zero replicated DIIS convergence");
  close(root_diis.correlation_energy,incremental.correlation_energy,1.0e-12,
        "rank-zero/all-rank replicated DIIS CCSD energy");
  require(root_diis.history.size()==incremental.history.size(),
          "rank-zero/all-rank replicated DIIS iteration count");
  for(std::size_t index=0;index<incremental.t1.size();++index)
    close(root_diis.t1.data()[index],incremental.t1.data()[index],1.0e-12,
          "rank-zero/all-rank replicated DIIS T1");
  for(std::size_t index=0;index<incremental.t2.size();++index)
    close(root_diis.t2.data()[index],incremental.t2.data()[index],1.0e-12,
          "rank-zero/all-rank replicated DIIS T2");
  if(parallel::world().rank==0)
    require(root_diis.cumulative_timing.diis_error_dot_products>0,
            "rank-zero replicated DIIS owner work");
  else
    require(root_diis.cumulative_timing.diis_error_dot_products==0,
            "nonowner replicated DIIS avoids redundant Gram work");
  auto distributed_energy_options=incremental_options;
  distributed_energy_options.use_distributed_replicated_energy=true;
  const auto distributed_energy=solve_ccsd(
      fock,eri,occupied,distributed_energy_options);
  require(distributed_energy.converged,
          "distributed replicated-energy CCSD convergence");
  close(distributed_energy.correlation_energy,
        incremental.correlation_energy,1.0e-12,
        "distributed/serial replicated CCSD energy");
  require(distributed_energy.history.size()==incremental.history.size(),
          "distributed/serial replicated-energy iteration count");
  for(std::size_t index=0;index<incremental.t1.size();++index)
    close(distributed_energy.t1.data()[index],incremental.t1.data()[index],
          1.0e-12,"distributed/serial replicated-energy T1");
  for(std::size_t index=0;index<incremental.t2.size();++index)
    close(distributed_energy.t2.data()[index],incremental.t2.data()[index],
          1.0e-12,"distributed/serial replicated-energy T2");
  auto cached_wabef_options=incremental_options;
  cached_wabef_options.wabef_backend=CcsdWabefBackend::tiled_cpu;
  cached_wabef_options.wabef_packing_backend=
      CcsdWabefPackingBackend::cached_invariants;
  cached_wabef_options.wabef_invariant_cache_max_bytes=1024U*1024U;
  const auto cached_wabef=solve_ccsd(
      fock,eri,occupied,cached_wabef_options);
  require(cached_wabef.converged,"cached-invariant W_abef CCSD convergence");
  close(cached_wabef.correlation_energy,incremental.correlation_energy,1.0e-12,
        "cached-invariant/direct W_abef CCSD energy");
  require(cached_wabef.history.size()==incremental.history.size(),
          "cached-invariant/direct W_abef CCSD iteration count");
  for(std::size_t index=0;index<incremental.t1.size();++index)
    close(cached_wabef.t1.data()[index],incremental.t1.data()[index],1.0e-12,
          "cached-invariant/direct W_abef CCSD T1");
  for(std::size_t index=0;index<incremental.t2.size();++index)
    close(cached_wabef.t2.data()[index],incremental.t2.data()[index],1.0e-12,
          "cached-invariant/direct W_abef CCSD T2");
  require(cached_wabef.wabef_invariant_cache_bytes>0&&
              cached_wabef.wabef_invariant_cache_build_seconds>=0.0,
          "cached-invariant W_abef instrumentation");
  if(parallel::world().rank==0)
    std::cout<<"CCSD DIIS Gram products recomputed "
             <<recomputed.cumulative_timing.diis_error_dot_products
             <<" incremental "
             <<incremental.cumulative_timing.diis_error_dot_products<<'\n';
}
void test_wabef_invariant_cache_memory_gate(){
  constexpr std::size_t occupied=2,orbitals=4,virtuals=2;
  Matrix fock({orbitals,orbitals});
  fock(0,0)=-1.1;fock(1,1)=-0.9;fock(2,2)=0.4;fock(3,3)=0.8;
  const auto eri=make_eri(orbitals);
  Matrix t1({occupied,virtuals});
  Tensor4 t2({occupied,occupied,virtuals,virtuals});
  CcsdOptions options;
  options.amplitude_backend=CcsdAmplitudeBackend::replicated;
  options.wabef_backend=CcsdWabefBackend::tiled_cpu;
  options.wabef_packing_backend=
      CcsdWabefPackingBackend::cached_invariants;
  options.wabef_invariant_cache_max_bytes=1;
  bool rejected=false;
  try{
    static_cast<void>(evaluate_ccsd_residual(
        fock,eri,t1,t2,occupied,options));
  }catch(const std::runtime_error& error){
    rejected=std::string(error.what()).find(
        "invariant cache requires")!=std::string::npos;
  }
  require(rejected,"W_abef invariant cache fail-closed memory gate");
}
void rotate(const Matrix& f,const Tensor4& g,std::size_t o,Matrix& fr,Tensor4& gr){auto n=f.dimensions()[0];Matrix u({n,n});for(std::size_t p=0;p<n;++p)u(p,p)=1;double c=std::cos(.37),s=std::sin(.37),d=std::cos(-.29),z=std::sin(-.29);u(0,0)=c;u(0,1)=-s;u(1,0)=s;u(1,1)=c;u(o,o)=d;u(o,o+1)=-z;u(o+1,o)=z;u(o+1,o+1)=d;fr=Matrix({n,n});gr=Tensor4({n,n,n,n});for(std::size_t P=0;P<n;++P)for(std::size_t Q=0;Q<n;++Q)for(std::size_t p=0;p<n;++p)for(std::size_t q=0;q<n;++q)fr(P,Q)+=u(p,P)*f(p,q)*u(q,Q);for(std::size_t P=0;P<n;++P)for(std::size_t Q=0;Q<n;++Q)for(std::size_t R=0;R<n;++R)for(std::size_t S=0;S<n;++S)for(std::size_t p=0;p<n;++p)for(std::size_t q=0;q<n;++q)for(std::size_t r=0;r<n;++r)for(std::size_t s2=0;s2<n;++s2)gr(P,Q,R,S)+=u(p,P)*u(q,Q)*u(r,R)*u(s2,S)*g(p,q,r,s2);}
void validate_distributed_diis(const Matrix& f,const Tensor4& g,std::size_t o,const CcsdOptions& options,const CcsdResult& distributed){
  if(!global_arrays_runtime_active()){require(!distributed.diis_memory.global_arrays,"serial solver reported GA DIIS");return;}
  auto invalid_tiled_doubles=options;invalid_tiled_doubles.amplitude_backend=CcsdAmplitudeBackend::global_arrays;invalid_tiled_doubles.wabef_backend=CcsdWabefBackend::cpu;invalid_tiled_doubles.use_tiled_doubles=true;bool tiled_doubles_rejected=false;try{(void)solve_ccsd(f,g,o,invalid_tiled_doubles);}catch(const std::runtime_error&){tiled_doubles_rejected=true;}require(tiled_doubles_rejected,"GA tiled doubles fail-closed gate");
  auto invalid_tiled_intermediates=options;invalid_tiled_intermediates.amplitude_backend=CcsdAmplitudeBackend::global_arrays;invalid_tiled_intermediates.wabef_backend=CcsdWabefBackend::cpu;invalid_tiled_intermediates.use_tiled_intermediates=true;bool tiled_intermediates_rejected=false;try{(void)solve_ccsd(f,g,o,invalid_tiled_intermediates);}catch(const std::runtime_error&){tiled_intermediates_rejected=true;}require(tiled_intermediates_rejected,"GA tiled intermediates fail-closed gate");
  require(distributed.diis_memory.global_arrays&&distributed.diis_memory.history_entries>0,"GA DIIS activation");
  require(distributed.distributed_memory.global_arrays&&distributed.distributed_memory.solver_pair_arrays==4,"GA current/residual activation");
  const auto solver_expected=4*distributed.t2.size()*sizeof(double);
  require(distributed.distributed_memory.solver_pair_global_bytes==solver_expected&&distributed.distributed_memory.solver_pair_replicated_bytes_per_rank==solver_expected,"GA current/residual global memory accounting");
  const auto intermediate_expected=3*distributed.t2.size()*sizeof(double);
  require(distributed.distributed_memory.intermediate_pair_arrays==3&&distributed.distributed_memory.intermediate_pair_global_bytes==intermediate_expected&&distributed.distributed_memory.intermediate_pair_replicated_bytes_per_rank==intermediate_expected,"GA tau/tilde-tau/W_mbej memory accounting");
  std::vector<double> intermediate_aggregate{static_cast<double>(distributed.distributed_memory.intermediate_pair_local_bytes)};parallel::sum_in_place(intermediate_aggregate);close(intermediate_aggregate[0],static_cast<double>(intermediate_expected),0.0,"GA intermediate distributed byte coverage");
  if(parallel::world().size>1)require(distributed.distributed_memory.intermediate_pair_local_bytes<distributed.distributed_memory.intermediate_pair_replicated_bytes_per_rank,"GA intermediate per-rank memory reduction");
  const auto wmnij_expected=o*o*o*o*sizeof(double);
  require(distributed.distributed_memory.occupied_intermediate_pair_arrays==1&&distributed.distributed_memory.occupied_intermediate_pair_global_bytes==wmnij_expected&&distributed.distributed_memory.occupied_intermediate_pair_replicated_bytes_per_rank==wmnij_expected,"GA W_mnij memory accounting");
  std::vector<double> wmnij_aggregate{static_cast<double>(distributed.distributed_memory.occupied_intermediate_pair_local_bytes)};parallel::sum_in_place(wmnij_aggregate);close(wmnij_aggregate[0],static_cast<double>(wmnij_expected),0.0,"GA W_mnij distributed byte coverage");
  if(parallel::world().size>1)require(distributed.distributed_memory.occupied_intermediate_pair_local_bytes<distributed.distributed_memory.occupied_intermediate_pair_replicated_bytes_per_rank,"GA W_mnij per-rank memory reduction");
  const auto virtuals=distributed.t1.dimensions()[1];
  const auto cache_expected=((4*o+2)*virtuals*virtuals+o*o)*sizeof(double);
  require(distributed.distributed_memory.bounded_pair_cache_bytes_per_rank==cache_expected,"bounded pair cache memory accounting");
  const auto small_expected=(virtuals*virtuals+o*o+o*virtuals)*sizeof(double);
  require(distributed.distributed_memory.replicated_small_intermediate_bytes_per_rank==small_expected,"small replicated intermediate memory accounting");
  const auto rank=parallel::world().rank;const auto ranks=parallel::world().size;const auto virtual_pairs=virtuals*virtuals;const auto owned_virtual_pairs=rank<virtual_pairs?1+(virtual_pairs-1-rank)/ranks:0;
  require(distributed.distributed_memory.streamed_wabef_accumulator_bytes_per_rank==owned_virtual_pairs*o*o*sizeof(double),"streamed W_abef accumulator memory accounting");
  std::vector<double> solver_aggregate{static_cast<double>(distributed.distributed_memory.solver_pair_local_bytes)};parallel::sum_in_place(solver_aggregate);close(solver_aggregate[0],static_cast<double>(solver_expected),0.0,"GA current/residual distributed byte coverage");
  if(parallel::world().size>1)require(distributed.distributed_memory.solver_pair_local_bytes<distributed.distributed_memory.solver_pair_replicated_bytes_per_rank,"GA current/residual per-rank memory reduction");
  const auto expected=2*distributed.diis_memory.history_entries*distributed.t2.size()*sizeof(double);
  require(distributed.diis_memory.doubles_history_global_bytes==expected&&distributed.diis_memory.doubles_history_replicated_bytes_per_rank==expected,"GA DIIS global memory accounting");
  require(distributed.diis_memory.singles_history_bytes_per_rank==2*distributed.diis_memory.history_entries*distributed.t1.size()*sizeof(double),"GA DIIS singles memory accounting");
  std::vector<double> aggregate{static_cast<double>(distributed.diis_memory.doubles_history_local_bytes)};parallel::sum_in_place(aggregate);close(aggregate[0],static_cast<double>(expected),0.0,"GA DIIS distributed byte coverage");
  if(parallel::world().size>1)require(distributed.diis_memory.doubles_history_local_bytes<distributed.diis_memory.doubles_history_replicated_bytes_per_rank,"GA DIIS per-rank memory reduction");
  auto reference_options=options;reference_options.use_global_arrays_diis=false;reference_options.amplitude_backend=CcsdAmplitudeBackend::replicated;const auto replicated=solve_ccsd(f,g,o,reference_options);require(replicated.converged,"replicated DIIS reference convergence");close(distributed.correlation_energy,replicated.correlation_energy,1e-12,"GA/replicated DIIS energy");
  double amplitude_error=0.0;for(std::size_t i=0;i<distributed.t1.size();++i)amplitude_error=std::max(amplitude_error,std::abs(distributed.t1.data()[i]-replicated.t1.data()[i]));for(std::size_t i=0;i<distributed.t2.size();++i)amplitude_error=std::max(amplitude_error,std::abs(distributed.t2.data()[i]-replicated.t2.data()[i]));require(amplitude_error<1e-10,"GA/replicated DIIS amplitudes");
  if(parallel::world().rank==0)std::cout<<"Global Arrays DIIS T2 history bytes rank0 "<<distributed.diis_memory.doubles_history_local_bytes<<" global "<<distributed.diis_memory.doubles_history_global_bytes<<" replicated/rank "<<distributed.diis_memory.doubles_history_replicated_bytes_per_rank<<" entries "<<distributed.diis_memory.history_entries<<'\n';
  if(parallel::world().rank==0)std::cout<<"Global Arrays solver pair bytes rank0 "<<distributed.distributed_memory.solver_pair_local_bytes<<" global "<<distributed.distributed_memory.solver_pair_global_bytes<<" replicated/rank "<<distributed.distributed_memory.solver_pair_replicated_bytes_per_rank<<" arrays "<<distributed.distributed_memory.solver_pair_arrays<<'\n';
  if(parallel::world().rank==0)std::cout<<"Global Arrays intermediate pair bytes rank0 "<<distributed.distributed_memory.intermediate_pair_local_bytes<<" global "<<distributed.distributed_memory.intermediate_pair_global_bytes<<" replicated/rank "<<distributed.distributed_memory.intermediate_pair_replicated_bytes_per_rank<<" arrays "<<distributed.distributed_memory.intermediate_pair_arrays<<" bounded-cache/rank "<<distributed.distributed_memory.bounded_pair_cache_bytes_per_rank<<'\n';
  if(parallel::world().rank==0)std::cout<<"Global Arrays W_mnij bytes rank0 "<<distributed.distributed_memory.occupied_intermediate_pair_local_bytes<<" global "<<distributed.distributed_memory.occupied_intermediate_pair_global_bytes<<" replicated/rank "<<distributed.distributed_memory.occupied_intermediate_pair_replicated_bytes_per_rank<<" arrays "<<distributed.distributed_memory.occupied_intermediate_pair_arrays<<'\n';
}
void test_rotation(){constexpr std::size_t o=3,n=6;Matrix f({n,n});f(0,0)=-1.2;f(1,1)=-1;f(2,2)=-.8;f(3,3)=.4;f(4,4)=.7;f(5,5)=1;auto g=make_eri(n);CcsdOptions co;co.max_iterations=150;co.residual_tolerance=1e-9;co.diis_start=1;if(global_arrays_runtime_active())co.amplitude_backend=CcsdAmplitudeBackend::global_arrays;auto c=solve_ccsd(f,g,o,co);require(c.converged,"canonical CCSD convergence");validate_distributed_diis(f,g,o,co,c);require(t2_antisymmetry_error(c.t2)<1e-12,"t2 antisymmetry");TriplesOptions to;to.residual_tolerance=1e-11;to.memory_limit_mib=64;auto t=solve_triples(f,g,c.t1,c.t2,to);require(t.converged,"canonical triples convergence");require(triples_antisymmetry_error(t.right_moment)<1e-12,"triples antisymmetry");Matrix fr;Tensor4 gr;rotate(f,g,o,fr,gr);auto cr=solve_ccsd(fr,gr,o,co);require(cr.converged,"rotated CCSD convergence");auto tr=solve_triples(fr,gr,cr.t1,cr.t2,to);require(tr.converged,"rotated triples convergence");close(cr.correlation_energy,c.correlation_energy,2e-9,"CCSD invariance");close(tr.energy,t.energy,2e-10,"triples invariance");auto cp=partition_ccsd(fr,gr,cr.t1,cr.t2,o,{0,0,1},{0,1,1});close(cp.sum,cr.correlation_energy,1e-12,"CCSD partition");require(cp.many_body_classes.contains("M1")&&cp.many_body_classes.contains("M2"),"CCSD M1/M2 classes");auto tp=partition_triples(tr.energy_contribution,{0,0,1},{0,1,1});close(tp.sum,tr.energy,1e-14,"triples partition");require(tp.many_body_classes.contains("M1")&&tp.many_body_classes.contains("M2"),"triples M1/M2 classes");if(global_arrays_runtime_active()){auto automatic=co;automatic.amplitude_backend=CcsdAmplitudeBackend::automatic;automatic.replicated_t2_max_bytes=std::numeric_limits<std::size_t>::max();const auto auto_result=solve_ccsd(f,g,o,automatic);require(auto_result.amplitude_backend==CcsdAmplitudeBackend::replicated,"automatic small-system CCSD backend");close(auto_result.correlation_energy,c.correlation_energy,1e-12,"automatic/GA CCSD energy");}if(parallel::world().rank==0){std::cout.setf(std::ios::scientific);std::cout<<"CCSD rotation error "<<std::abs(cr.correlation_energy-c.correlation_energy)<<"\n(T) rotation error "<<std::abs(tr.energy-t.energy)<<"\nR1 "<<cr.r1_norm<<" R2 "<<cr.r2_norm<<" R3 "<<tr.residual_norm<<"\nCCSD partition error "<<std::abs(cp.sum_error)<<" triples partition error "<<std::abs(tp.sum_error)<<'\n';}}

void test_streamed_canonical_triples(){
  constexpr std::size_t o=4,n=8;
  Matrix f({n,n});
  f(0,0)=-1.4;f(1,1)=-1.2;f(2,2)=-1.0;f(3,3)=-.8;
  f(4,4)=.3;f(5,5)=.5;f(6,6)=.7;f(7,7)=.9;
  auto g=make_eri(n);
  CcsdOptions co;co.max_iterations=150;co.residual_tolerance=1e-9;
  auto c=solve_ccsd(f,g,o,co);
  require(c.converged,"streamed reference CCSD convergence");
  TriplesOptions to;to.residual_tolerance=1e-11;to.memory_limit_mib=64;
  auto full=solve_triples(f,g,c.t1,c.t2,to);
  require(full.converged,"streamed full triples convergence");
  const double bracket=dot(full.right_moment.data(),full.response.data())/36.0;
  CanonicalTriplesOptions so;so.occupied_task_grain=1;
  auto streamed=solve_canonical_triples_streamed(f,g,c.t1,c.t2,so);
  close(streamed.bracket_energy,bracket,1e-14,"streamed CCSD[T]");
  close(streamed.parenthetical_energy,full.energy,1e-14,"streamed CCSD(T)");
  require(streamed.occupied_triples==4&&streamed.virtual_triples==4&&
          streamed.tasks==4,"streamed task counts");
  require(streamed.workspace_per_worker_mib>0.0,
          "streamed workspace estimate");
  CanonicalTriplesOptions tiled_options;
  tiled_options.backend=CanonicalTriplesBackend::tiled_cpu;
  tiled_options.compute_bracket=false;
  for(const std::size_t tile_size:{1U,2U,3U,4U}){
    tiled_options.virtual_tile_size=tile_size;
    auto tiled=solve_canonical_triples_streamed(
        f,g,c.t1,c.t2,tiled_options);
    close(tiled.parenthetical_energy,streamed.parenthetical_energy,2e-13,
          "bounded tiled CPU CCSD(T)");
    close(tiled.bracket_energy,0.0,0.0,"bounded tiled CCSD[T] disabled");
    require(tiled.tile_workspace_per_task_mib>0.0,
            "bounded tiled workspace estimate");
  }
  tiled_options.virtual_tile_size=2;
  if(cuda_triples_runtime_info().available){
    auto cuda_options=tiled_options;
    cuda_options.backend=CanonicalTriplesBackend::cuda;
    auto accelerated=solve_canonical_triples_streamed(
        f,g,c.t1,c.t2,cuda_options);
    close(accelerated.parenthetical_energy,streamed.parenthetical_energy,
          1e-11,"production CUDA moment CCSD(T)");
    require(accelerated.tile_workspace_per_task_mib>0.0,
            "production CUDA moment workspace estimate");
    auto tce_cuda_options=tiled_options;
    tce_cuda_options.backend=CanonicalTriplesBackend::tce_cuda;
    auto tce_accelerated=solve_canonical_triples_streamed(
        f,g,c.t1,c.t2,tce_cuda_options);
    close(tce_accelerated.parenthetical_energy,
          streamed.parenthetical_energy,1e-11,
          "TCE-style CUDA cuBLAS moment CCSD(T)");
    require(tce_accelerated.tile_workspace_per_task_mib>0.0,
            "TCE-style CUDA moment workspace estimate");
  }
  auto invalid_tiled=tiled_options;invalid_tiled.compute_bracket=true;
  bool invalid_tiled_rejected=false;
  try{(void)solve_canonical_triples_streamed(
      f,g,c.t1,c.t2,invalid_tiled);}
  catch(const std::invalid_argument&){invalid_tiled_rejected=true;}
  require(invalid_tiled_rejected,"tiled CCSD[T] request rejection");
  so.occupied_task_grain=3;
  auto regrouped=solve_canonical_triples_streamed(f,g,c.t1,c.t2,so);
  require(regrouped.tasks==2,"streamed regrouped task count");
  close(regrouped.bracket_energy,streamed.bracket_energy,1e-15,
        "streamed task-grain CCSD[T]");
  close(regrouped.parenthetical_energy,streamed.parenthetical_energy,1e-15,
        "streamed task-grain CCSD(T)");
  Matrix noncanonical=f;noncanonical(0,1)=1e-4;
  bool rejected=false;
  try{(void)solve_canonical_triples_streamed(noncanonical,g,c.t1,c.t2,so);}
  catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"streamed noncanonical rejection");

  struct StopCheckpoint {};
  std::vector<double> saved_bracket,saved_parenthetical;
  std::vector<std::uint8_t> saved_completed;
  CanonicalTriplesOptions interrupted_options;
  interrupted_options.checkpoint_interval_tasks=1;
  interrupted_options.observer=[&](const CanonicalTriplesProgress& progress){
    saved_bracket=progress.bracket_task_energies;
    saved_parenthetical=progress.parenthetical_task_energies;
    saved_completed=progress.completed_tasks;
    throw StopCheckpoint{};
  };
  bool stopped=false;try{(void)solve_canonical_triples_streamed(
      f,g,c.t1,c.t2,interrupted_options);}catch(const StopCheckpoint&){stopped=true;}
  require(stopped&&std::count(saved_completed.begin(),saved_completed.end(),1U)==1,
          "canonical triples checkpoint interruption");
  CanonicalTriplesOptions restart_options;
  restart_options.initial_bracket_task_energies=saved_bracket;
  restart_options.initial_parenthetical_task_energies=saved_parenthetical;
  restart_options.completed_tasks=saved_completed;
  auto restarted=solve_canonical_triples_streamed(
      f,g,c.t1,c.t2,restart_options);
  close(restarted.bracket_energy,streamed.bracket_energy,1e-15,
        "restarted streamed CCSD[T]");
  close(restarted.parenthetical_energy,streamed.parenthetical_energy,1e-15,
        "restarted streamed CCSD(T)");
}

void test_iterative_restarts(){
  constexpr std::size_t o=3,n=6;Matrix f({n,n});
  f(0,0)=-1.2;f(1,1)=-1;f(2,2)=-.8;f(3,3)=.4;f(4,4)=.7;f(5,5)=1;
  auto g=make_eri(n);CcsdOptions full_options;full_options.max_iterations=150;
  full_options.residual_tolerance=1e-9;
  std::vector<CcsdIterationReport> reports;
  full_options.iteration_observer=[&](const CcsdIterationReport& report){
    reports.push_back(report);
  };
  const auto full=solve_ccsd(f,g,o,full_options);require(full.converged,"restart full CCSD");
  require(full.amplitude_backend==CcsdAmplitudeBackend::replicated,
          "automatic small CCSD amplitude backend");
  require(!full.diis_memory.global_arrays,
          "automatic small CCSD local DIIS backend");
  require(!reports.empty()&&reports.back().converged,
          "lightweight CCSD convergence report");
  require(reports.size()>=full.history.size(),
          "lightweight CCSD iteration report count");
  require(reports.back().timing.iteration_seconds>0.0&&
          full.cumulative_timing.iteration_seconds>0.0,
          "CCSD iteration timing");
  for(const auto& report:reports)
    require(report.amplitude_backend==CcsdAmplitudeBackend::replicated&&
            report.timing.wabef_seconds>=0.0&&
            report.timing.residual_seconds>=0.0,
            "lightweight CCSD timing fields");
  CcsdOptions tiled_options=full_options;
  tiled_options.iteration_observer={};
  tiled_options.wabef_backend=CcsdWabefBackend::tiled_cpu;
  tiled_options.use_tiled_intermediates=true;
  tiled_options.use_tiled_doubles=true;
  tiled_options.wabef_tile_max_bytes=256;
  const auto tiled=solve_ccsd(f,g,o,tiled_options);
  require(tiled.converged,"tiled BLAS CCSD convergence");
  require(tiled.wabef_backend==CcsdWabefBackend::tiled_cpu,
          "tiled BLAS CCSD backend report");
  require(tiled.cumulative_timing.wabef_external_tiles>0&&
              tiled.cumulative_timing.wabef_internal_panels>
                  tiled.cumulative_timing.wabef_external_tiles&&
              tiled.cumulative_timing.wabef_output_tasks>0&&
              tiled.cumulative_timing.wabef_blas_contractions==
                  2*tiled.cumulative_timing.wabef_internal_panels,
          "tiled BLAS CCSD task counters");
  require(tiled.cumulative_timing.doubles_occupied_pair_tasks>0&&
              tiled.cumulative_timing.doubles_blas_contractions==
                  tiled.cumulative_timing.doubles_occupied_pair_tasks*
                      (2*o+7),
          "tiled BLAS CCSD doubles counters");
  require(tiled.cumulative_timing.intermediate_mixed_pair_tasks>0&&
              tiled.cumulative_timing.intermediate_blas_contractions==
                  2*tiled.cumulative_timing.intermediate_mixed_pair_tasks,
          "tiled BLAS CCSD intermediate counters");
  close(tiled.correlation_energy,full.correlation_energy,2e-11,
        "tiled BLAS CCSD energy");
  close(tiled.r1_norm,full.r1_norm,2e-10,
        "tiled BLAS CCSD R1 norm");
  close(tiled.r2_norm,full.r2_norm,2e-10,
        "tiled BLAS CCSD R2 norm");
  for(std::size_t element=0;element<full.t1.size();++element)
    close(tiled.t1.data()[element],full.t1.data()[element],2e-10,
          "tiled BLAS CCSD T1 amplitude");
  for(std::size_t element=0;element<full.t2.size();++element)
    close(tiled.t2.data()[element],full.t2.data()[element],2e-10,
          "tiled BLAS CCSD T2 amplitude");
  CcsdOptions partial_options=full_options;partial_options.max_iterations=1;
  partial_options.iteration_observer={};
  const auto partial=solve_ccsd(f,g,o,partial_options);
  require(partial.history.size()==1,"partial CCSD state");
  CcsdOptions restart_options=full_options;restart_options.initial_t1=partial.t1;
  restart_options.initial_t2=partial.t2;restart_options.completed_iterations=1;
  restart_options.initial_history=partial.history;
  const auto restarted=solve_ccsd(f,g,o,restart_options);
  require(restarted.converged,"restarted CCSD convergence");
  close(restarted.correlation_energy,full.correlation_energy,2e-11,
        "restarted CCSD energy");

  TriplesOptions triples_full;triples_full.memory_limit_mib=64;
  triples_full.residual_tolerance=1e-11;
  const auto full_triples=solve_triples(f,g,full.t1,full.t2,triples_full);
  TriplesOptions triples_partial=triples_full;triples_partial.max_iterations=1;
  const auto partial_triples=solve_triples(f,g,full.t1,full.t2,triples_partial);
  TriplesOptions triples_restart=triples_full;
  triples_restart.initial_response=partial_triples.response;
  triples_restart.completed_iterations=partial_triples.iterations;
  triples_restart.initial_solver_history=partial_triples.solver_history;
  const auto restarted_triples=solve_triples(
      f,g,full.t1,full.t2,triples_restart);
  require(restarted_triples.converged,"restarted triples convergence");
  close(restarted_triples.energy,full_triples.energy,1e-13,
        "restarted triples energy");
}

void test_three_monomer_selected_panel_cache(){
  constexpr std::size_t spatial_occupied=3;
  constexpr std::size_t spatial_virtuals=3;
  constexpr std::size_t spatial_orbitals=
      spatial_occupied+spatial_virtuals;
  SpatialRhfHamiltonian spatial;
  spatial.nocc=spatial_occupied;
  spatial.nvirt=spatial_virtuals;
  spatial.fock=Matrix({spatial_orbitals,spatial_orbitals});
  spatial.hcore=Matrix({spatial_orbitals,spatial_orbitals});
  spatial.eri_chemist=SymmetricEri4(spatial_orbitals);
  spatial.fock(0,0)=-1.3;spatial.fock(1,1)=-1.0;
  spatial.fock(2,2)=-0.8;spatial.fock(3,3)=0.4;
  spatial.fock(4,4)=0.7;spatial.fock(5,5)=1.0;
  for(std::size_t p=0;p<spatial_orbitals;++p)
    for(std::size_t q=0;q<spatial_orbitals;++q)
      for(std::size_t r=0;r<spatial_orbitals;++r)
        for(std::size_t s=0;s<spatial_orbitals;++s)
          spatial.eri_chemist(p,q,r,s)=
              2.0e-4/(1.0+static_cast<double>(p+2*q+3*r+5*s));
  const std::vector<int> occupied_labels{0,0,1,1,2,2};
  const std::vector<int> virtual_labels{0,0,1,1,2,2};
  SelectedCcsdManifoldOptions manifold_options;
  manifold_options.maximum_monomers=2;
  const auto plan=plan_selected_ccsd_manifold(
      occupied_labels,virtual_labels,manifold_options);
  const auto projection=make_selected_ccsd_projection_mask(
      occupied_labels,virtual_labels,manifold_options);
  CcsdOptions options;
  options.max_iterations=80;
  options.energy_tolerance=1.0e-12;
  options.residual_tolerance=1.0e-10;
  options.amplitude_backend=CcsdAmplitudeBackend::replicated;
  options.projection_mask=
      std::make_shared<CcsdProjectionMask>(projection);
  const auto dense=solve_ccsd(spatial,options);
  require(dense.converged,"three-monomer dense selected CCSD convergence");
  const SelectedIntegralStore selected(plan,spatial);
  options.wabef_backend=CcsdWabefBackend::tiled_cpu;
  options.use_tiled_intermediates=true;
  options.use_tiled_doubles=true;
  options.wabef_tile_max_bytes=4096;
  options.selected_eri_panel_cache_max_bytes=16U*1024U*1024U;
  options.use_distributed_replicated_energy=true;
  CcsdResult cached;
  try {
    cached=solve_ccsd(spatial.fock,spatial.nocc,spatial.nvirt,
                      selected,options);
  } catch(const std::exception& error) {
    throw std::runtime_error(
        std::string("three-monomer projected tiled solve: ")+error.what());
  }
  require(cached.converged&&cached.selected_eri_panel_cache_enabled,
          "three-monomer selected panel-cache CCSD convergence");
  require(cached.cumulative_timing.intermediate_mixed_pair_tasks>0&&
              cached.cumulative_timing.intermediate_blas_contractions>0,
          "three-monomer projected tiled-intermediate execution");
  close(cached.correlation_energy,dense.correlation_energy,2.0e-11,
        "three-monomer selected panel-cache CCSD energy");
  close(cached.r1_norm,dense.r1_norm,2.0e-10,
        "three-monomer selected panel-cache CCSD R1");
  close(cached.r2_norm,dense.r2_norm,2.0e-10,
        "three-monomer selected panel-cache CCSD R2");
  EnergyPartition selected_partition;
  try {
    selected_partition=partition_ccsd(
        spatial.fock,selected,cached.t1,cached.t2,spatial.nocc,
        occupied_labels,virtual_labels);
  } catch(const std::exception& error) {
    throw std::runtime_error(
        std::string("three-monomer selected partition: ")+error.what());
  }
  close(selected_partition.parent_energy,cached.correlation_energy,2.0e-11,
        "three-monomer selected CCSD partition parent energy");
  close(selected_partition.sum_error,0.0,2.0e-11,
        "three-monomer selected CCSD partition closure");
  const auto partition_value=[&](const std::string& label){
    const auto found=selected_partition.many_body_classes.find(label);
    return found==selected_partition.many_body_classes.end()?0.0:found->second;
  };
  const auto retained_energy=partition_value("M1")+partition_value("M2");
  const auto higher_span_disconnected=
      partition_value("M3")+partition_value("M4");
  close(retained_energy+higher_span_disconnected,
        selected_partition.parent_energy,2.0e-11,
        "strict selected M1+M2 energy plus disconnected higher span");
  options.selected_eri_panel_cache_max_bytes=0;
  const auto uncached=solve_ccsd(spatial.fock,spatial.nocc,spatial.nvirt,
                                 selected,options);
  require(uncached.converged&&!uncached.selected_eri_panel_cache_enabled,
          "three-monomer selected panel-cache fallback convergence");
  close(uncached.correlation_energy,dense.correlation_energy,2.0e-11,
        "three-monomer selected panel-cache fallback energy");
}
}
int main(int argc,char** argv) {
  parallel::Runtime runtime(argc,argv);
  // Flush phase markers even when stdout is piped through tee. This also
  // distinguishes GA initialization/finalization from equation-test work.
  std::cout << std::unitbuf;
  const auto report=[&](const std::string& message) {
    if(runtime.world().rank==0)std::cout<<"CCSD test progress: "<<message<<std::endl;
  };
  const auto run=[&](const char* name,auto test) {
    report(std::string(name)+" started");
    test();
    report(std::string(name)+" complete (rank zero)");
  };
  try {
    report("MPI initialized; ranks="+std::to_string(runtime.world().size));
    report(global_arrays_backend_available()
               ? "Global Arrays initialization started"
               : "Global Arrays disabled at build time; GA-specific checks not applicable");
    {
      GlobalArraysRuntime global_arrays;
      report(global_arrays.active() ? "Global Arrays initialized" : "MPI-only runtime ready");
      run("DIIS",test_scale_invariant_diis);
      run("MPI subgroup reductions",test_parallel_group_reduction);
      run("occupied pair layout",test_occupied_pair_layout);
      run("selected block storage",test_global_arrays_selected_block_tensor);
      run("pair storage",test_global_arrays_pair_tensor);
      run("frozen core",test_frozen_core);
      run("spatial ERI oracle",test_spatial_eri_view_oracle);
      run("spatial RCCSD convention",test_spatial_rccsd_convention);
      run("all electron",test_all_electron);
      run("water Z-matrix",test_water_zmatrix);
      run("many-body limits",test_many_body_limits);
      run("singles partition",test_singles_partition);
      run("C++ energy known case",test_cpp_energy_known_case);
      run("screened partition",test_screened_partition_and_csv);
      run("checkpoint roundtrip",test_checkpoint_roundtrip);
      run("converged checkpoint restore",test_converged_ccsd_checkpoint_restore);
      run("TCE output parser",test_tce_output_parser);
      run("benchmark output parser",test_benchmark_output_parser);
      run("operator",test_operator);
      run("random amplitude residual oracle",test_random_amplitude_residual_oracle);
      run("incremental DIIS CCSD",test_incremental_diis_ccsd);
      run("Wabef cache memory",test_wabef_invariant_cache_memory_gate);
      run("rotation invariance",test_rotation);
      run("streamed canonical triples",test_streamed_canonical_triples);
      run("iterative restarts",test_iterative_restarts);
      run("three-monomer selected panel cache",test_three_monomer_selected_panel_cache);
      report("runtime teardown started");
    }
    // Do not announce suite success before GA collective teardown completes.
    parallel::barrier();
    if(runtime.world().rank==0)std::cout<<"all CCSD(T) tests passed"<<std::endl;
    return 0;
  } catch(const std::exception& e) {
    if(runtime.world().rank==0)std::cerr<<"test failure: "<<e.what()<<std::endl;
    if(runtime.world().size>1)runtime.abort(1);
    return 1;
  }
}
