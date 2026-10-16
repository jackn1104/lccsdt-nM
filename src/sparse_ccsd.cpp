#include "localized_cc/sparse_ccsd.hpp"

#include "localized_cc/distributed_block_tensor.hpp"
#include "localized_cc/parallel.hpp"
#include "localized_cc/selected_ccsd_storage.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace localized_cc {
namespace {

using SteadyClock = std::chrono::steady_clock;

double elapsed_seconds(const SteadyClock::time_point begin,
                       const SteadyClock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

void add_timing(CcsdIterationTiming& total,
                const CcsdIterationTiming& iteration) {
  total.intermediates_seconds += iteration.intermediates_seconds;
  total.singles_seconds += iteration.singles_seconds;
  total.doubles_seconds += iteration.doubles_seconds;
  total.doubles_occupied_pair_tasks +=
      iteration.doubles_occupied_pair_tasks;
  total.doubles_blas_contractions += iteration.doubles_blas_contractions;
  total.wabef_seconds += iteration.wabef_seconds;
  total.wabef_build_seconds += iteration.wabef_build_seconds;
  total.wabef_contraction_seconds += iteration.wabef_contraction_seconds;
  total.wabef_accumulation_seconds +=
      iteration.wabef_accumulation_seconds;
  total.wabef_external_tiles += iteration.wabef_external_tiles;
  total.wabef_internal_panels += iteration.wabef_internal_panels;
  total.wabef_output_tasks += iteration.wabef_output_tasks;
  total.wabef_blas_contractions += iteration.wabef_blas_contractions;
  total.residual_seconds += iteration.residual_seconds;
  total.diis_seconds += iteration.diis_seconds;
  total.energy_seconds += iteration.energy_seconds;
  total.iteration_seconds += iteration.iteration_seconds;
}

std::size_t checked_sum(const std::size_t left, const std::size_t right,
                        const char* message) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error(message);
  return left + right;
}

std::size_t checked_product(const std::size_t left,
                            const std::size_t right,
                            const char* message) {
  if (left != 0 &&
      right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(message);
  return left * right;
}

std::size_t block_bytes(const SelectedBlockLayout& layout) {
  if (layout.scientific_element_count() >
      std::numeric_limits<std::size_t>::max() / sizeof(double))
    throw std::overflow_error("selected CCSD phase-cache bytes overflow");
  return layout.scientific_element_count() * sizeof(double);
}

std::size_t owned_block_count(const GlobalArraysBlockTensor& tensor) {
  std::size_t result = 0;
  for (std::size_t block = 0; block < tensor.layout().block_count(); ++block)
    if (tensor.computation_block_owned(block))
      ++result;
  return result;
}

std::size_t owned_block_bytes(const GlobalArraysBlockTensor& tensor) {
  std::size_t elements = 0;
  for (std::size_t block = 0; block < tensor.layout().block_count(); ++block)
    if (tensor.computation_block_owned(block))
      elements = checked_sum(elements, tensor.layout().block_size(block),
                             "selected CCSD owned block bytes overflow");
  return checked_product(elements, sizeof(double),
                         "selected CCSD owned block bytes overflow");
}

std::vector<std::vector<double>> fetch_all(
    const GlobalArraysBlockTensor& tensor, const std::size_t maximum_bytes,
    std::size_t& peak_bytes, const std::size_t iteration,
    std::string phase, const std::size_t owned_output_blocks,
    std::vector<SparseCcsdPhaseStatistics>& statistics,
    const SparseCcsdPhaseObserver& observer) {
  const auto bytes = block_bytes(tensor.layout());
  if (bytes > maximum_bytes)
    throw std::runtime_error(
        "selected CCSD phase requires " + std::to_string(bytes) +
        " bytes but the explicit cache limit is " +
        std::to_string(maximum_bytes));
  peak_bytes = std::max(peak_bytes, bytes);
  const auto begin = SteadyClock::now();
  auto blocks = tensor.replicate();
  const auto world = parallel::world();
  const auto local_blocks = owned_block_count(tensor);
  const auto local_bytes = owned_block_bytes(tensor);
  SparseCcsdPhaseStatistics record{
      iteration,
      std::move(phase),
      world.rank,
      world.size,
      owned_output_blocks,
      tensor.layout().block_count(),
      local_blocks,
      tensor.layout().block_count(),
      tensor.layout().block_count() - local_blocks,
      bytes,
      bytes - local_bytes,
      0,
      0,
      0,
      1,
      elapsed_seconds(begin, SteadyClock::now())};
  statistics.push_back(record);
  if (observer)
    observer(record);
  return blocks;
}

std::vector<std::size_t> all_block_ids(const std::size_t block_count) {
  std::vector<std::size_t> result(block_count);
  for (std::size_t block = 0; block < block_count; ++block)
    result[block] = block;
  return result;
}

void prepare_intermediate_phase_cache(
    SelectedIntermediatePhaseCache& cache,
    const GlobalArraysBlockTensor& tensor,
    std::vector<std::size_t> declared_blocks, const std::size_t iteration,
    std::string phase, const std::size_t owned_output_blocks,
    std::size_t& peak_bytes,
    std::vector<SparseCcsdPhaseStatistics>& phase_statistics,
    const SparseCcsdPhaseObserver& observer) {
  std::sort(declared_blocks.begin(), declared_blocks.end());
  declared_blocks.erase(
      std::unique(declared_blocks.begin(), declared_blocks.end()),
      declared_blocks.end());
  std::size_t local_blocks = 0;
  std::size_t remote_bytes = 0;
  for (const auto block : declared_blocks) {
    if (tensor.computation_block_owned(block)) {
      ++local_blocks;
      continue;
    }
    remote_bytes = checked_sum(
        remote_bytes,
        checked_product(tensor.layout().block_size(block), sizeof(double),
                        "selected CCSD remote cache bytes overflow"),
        "selected CCSD remote cache bytes overflow");
  }
  const auto before = cache.statistics();
  const auto begin = SteadyClock::now();
  cache.prepare(declared_blocks);
  const auto after = cache.statistics();
  const auto fetched_blocks = after.fetched_blocks - before.fetched_blocks;
  const auto fetched_bytes = after.fetched_bytes - before.fetched_bytes;
  if (fetched_blocks != declared_blocks.size())
    throw std::logic_error(
        "selected CCSD phase cache did not fetch every declared block");
  peak_bytes = std::max(peak_bytes, fetched_bytes);
  const auto world = parallel::world();
  SparseCcsdPhaseStatistics record{
      iteration,
      std::move(phase),
      world.rank,
      world.size,
      owned_output_blocks,
      declared_blocks.size(),
      local_blocks,
      fetched_blocks,
      fetched_blocks - local_blocks,
      fetched_bytes,
      remote_bytes,
      declared_blocks.size() - fetched_blocks,
      0,
      0,
      0,
      elapsed_seconds(begin, SteadyClock::now())};
  phase_statistics.push_back(record);
  if (observer)
    observer(record);
}

class DoublesValues {
public:
  DoublesValues(const SelectedDoublesLayout& layout,
                const std::vector<std::vector<double>>& blocks)
      : layout_(&layout), blocks_(&blocks) {
    if (blocks.size() != layout.block_count())
      throw std::invalid_argument("selected T2 cache block-count mismatch");
  }

  [[nodiscard]] double operator()(const std::size_t i, const std::size_t j,
                                  const std::size_t a,
                                  const std::size_t b) const {
    if (!layout_->selected(i, j, a, b))
      return 0.0;
    const auto block = layout_->block_index(i, j, a, b);
    return blocks_->at(block).at(layout_->element_index(i, j, a, b));
  }

  [[nodiscard]] bool selected(const std::size_t i, const std::size_t j,
                              const std::size_t a,
                              const std::size_t b) const {
    return layout_->selected(i, j, a, b);
  }

private:
  const SelectedDoublesLayout* layout_{};
  const std::vector<std::vector<double>>* blocks_{};
};

class IntermediateValues {
public:
  IntermediateValues(const SelectedIntermediateLayout& layout,
                     const std::vector<std::vector<double>>& blocks)
      : layout_(&layout), blocks_(&blocks) {
    if (blocks.size() != layout.block_count())
      throw std::invalid_argument(
          "selected intermediate cache block-count mismatch");
  }

  [[nodiscard]] double operator()(
      const std::array<std::size_t, 4>& orbitals) const {
    if (!layout_->selected(orbitals))
      return 0.0;
    const auto block = layout_->block_index(orbitals);
    return blocks_->at(block).at(layout_->element_index(orbitals));
  }

private:
  const SelectedIntermediateLayout* layout_{};
  const std::vector<std::vector<double>>* blocks_{};
};

std::array<std::size_t, 4> doubles_orbitals(
    const SelectedDoublesLayout& layout, const std::size_t block,
    std::size_t flat) {
  std::array<std::size_t, 4> result{};
  for (std::size_t reversed = 0; reversed < 4; ++reversed) {
    const auto axis = 3 - reversed;
    const auto& orbitals = layout.axis_orbitals(block, axis);
    result[axis] = orbitals.at(flat % orbitals.size());
    flat /= orbitals.size();
  }
  if (flat != 0)
    throw std::logic_error("selected T2 block offset is invalid");
  return result;
}

std::array<std::size_t, 4> intermediate_orbitals(
    const SelectedIntermediateLayout& layout, const std::size_t block,
    std::size_t flat) {
  std::array<std::size_t, 4> result{};
  for (std::size_t reversed = 0; reversed < 4; ++reversed) {
    const auto axis = 3 - reversed;
    const auto& orbitals = layout.axis_orbitals(block, axis);
    result[axis] = orbitals.at(flat % orbitals.size());
    flat /= orbitals.size();
  }
  if (flat != 0)
    throw std::logic_error("selected intermediate block offset is invalid");
  return result;
}

std::vector<unsigned char> singles_mask(
    const SelectedCcsdManifoldPlan& plan, const std::size_t occupied,
    const std::size_t virtuals) {
  std::vector<unsigned char> result(occupied * virtuals, 0U);
  for (const auto& block : plan.singles_blocks)
    for (const auto i : plan.occupied_orbitals_by_monomer.at(
             block.occupied_monomer))
      for (const auto a : plan.virtual_orbitals_by_monomer.at(
               block.virtual_monomer))
        result[i * virtuals + a] = 1U;
  return result;
}

std::vector<int> labels_from_groups(
    const std::vector<std::vector<std::size_t>>& groups) {
  std::size_t count = 0;
  for (const auto& group : groups)
    count += group.size();
  std::vector<int> labels(count, -1);
  for (std::size_t monomer = 0; monomer < groups.size(); ++monomer) {
    if (monomer > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("sparse CCSD monomer label exceeds int");
    for (const auto orbital : groups[monomer]) {
      if (orbital >= labels.size() || labels[orbital] != -1)
        throw std::invalid_argument("sparse CCSD orbital labels are invalid");
      labels[orbital] = static_cast<int>(monomer);
    }
  }
  if (std::ranges::find(labels, -1) != labels.end())
    throw std::invalid_argument("sparse CCSD orbital labels are incomplete");
  return labels;
}

double denominator_two(const Matrix& d1, const std::size_t i,
                       const std::size_t j, const std::size_t a,
                       const std::size_t b) {
  return d1(i, a) + d1(j, b);
}

template <typename Eri>
double selected_energy(const Matrix& fock, const Eri& eri,
                       const Matrix& t1, const DoublesValues& t2,
                       const std::size_t occupied,
                       const std::size_t virtuals) {
  double energy = 0.0;
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t a = 0; a < virtuals; ++a)
      energy += fock(i, occupied + a) * t1(i, a);
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t j = 0; j < occupied; ++j)
      for (std::size_t a = 0; a < virtuals; ++a)
        for (std::size_t b = 0; b < virtuals; ++b) {
          const auto t2_value = t2(i, j, a, b);
          const auto t1_product = t1(i, a) * t1(j, b);
          if (!t2.selected(i, j, a, b) && t1_product == 0.0)
            continue;
          const auto integral = eri(i, j, occupied + a, occupied + b);
          energy += 0.25 * integral * t2_value +
                    0.5 * integral * t1_product;
        }
  return energy;
}

void load_flattened_restart(const SparseCcsdCheckpointState& restart,
                            const SelectedDoublesLayout& layout,
                            GlobalArraysBlockTensor& t2) {
  if (restart.t2_block_sizes.size() != layout.block_count())
    throw std::runtime_error(
        "sparse CCSD restart block count differs from current plan");
  std::vector<std::vector<double>> blocks;
  blocks.reserve(layout.block_count());
  std::size_t offset = 0;
  for (std::size_t block = 0; block < layout.block_count(); ++block) {
    const auto size = layout.block(block).element_count;
    if (restart.t2_block_sizes[block] != size ||
        size > restart.t2_block_values.size() - offset)
      throw std::runtime_error(
          "sparse CCSD restart block layout differs from current plan");
    blocks.emplace_back(
        restart.t2_block_values.begin() +
            static_cast<std::ptrdiff_t>(offset),
        restart.t2_block_values.begin() +
            static_cast<std::ptrdiff_t>(offset + size));
    offset += size;
  }
  if (offset != restart.t2_block_values.size())
    throw std::runtime_error(
        "sparse CCSD restart contains trailing T2 amplitudes");
  t2.load_replicated(blocks);
}

} // namespace

SparseCcsdCheckpointState make_sparse_ccsd_checkpoint(
    const SparseCcsdResult& result,
    const std::vector<int>& occupied_monomers,
    const std::vector<int>& virtual_monomers,
    const std::size_t maximum_monomers) {
  if (maximum_monomers == 0 || result.history.empty())
    throw std::invalid_argument("cannot checkpoint an empty sparse CCSD state");
  SparseCcsdCheckpointState state;
  state.converged = result.converged;
  state.completed_iterations = result.history.back().iteration;
  state.maximum_monomers = maximum_monomers;
  state.correlation_energy = result.correlation_energy;
  state.r1_norm = result.r1_norm;
  state.r2_norm = result.r2_norm;
  state.t1 = result.t1;
  state.occupied_monomers = occupied_monomers;
  state.virtual_monomers = virtual_monomers;
  state.history = result.history;
  for (const auto& block : result.t2_blocks) {
    state.t2_block_sizes.push_back(
        static_cast<std::uint64_t>(block.size()));
    state.t2_block_values.insert(state.t2_block_values.end(), block.begin(),
                                 block.end());
  }
  return state;
}

SparseCcsdResult solve_sparse_ccsd(
    const SpatialRhfHamiltonian& hamiltonian,
    const SelectedCcsdManifoldPlan& plan,
    const SparseCcsdOptions& options) {
  const SelectedIntegralStore integrals(plan, hamiltonian);
  return solve_sparse_ccsd(hamiltonian.fock, hamiltonian.nocc,
                           hamiltonian.nvirt, integrals, plan, options);
}

SparseCcsdResult solve_sparse_ccsd(
    const Matrix& spatial_fock, const std::size_t spatial_occupied,
    const std::size_t spatial_virtuals,
    const SelectedIntegralStore& integrals,
    const SelectedCcsdManifoldPlan& plan,
    const SparseCcsdOptions& options) {
  if (!global_arrays_runtime_active())
    throw std::runtime_error(
        "sparse CCSD requires an active Global Arrays runtime");
  if (!plan.dependency_status.ready_for_sparse_ccsd())
    throw std::invalid_argument("sparse CCSD dependency plan is incomplete");
  if (options.maximum_phase_cache_bytes == 0 ||
      options.maximum_monomers == 0)
    throw std::invalid_argument("sparse CCSD resource limits are invalid");
  if (!std::isfinite(options.equations.damping) ||
      options.equations.damping <= 0.0 || options.equations.damping > 1.0)
    throw std::invalid_argument(
        "sparse CCSD damping must be in the interval (0,1]");
  if (options.equations.wabef_backend != CcsdWabefBackend::cpu)
    throw std::runtime_error(
        "sparse CCSD non-scalar W_abef requires a selected-tile kernel");
  const auto occupied = 2 * spatial_occupied;
  const auto virtuals = 2 * spatial_virtuals;
  if (occupied == 0 || virtuals == 0)
    throw std::invalid_argument("sparse CCSD orbital spaces are empty");
  const SelectedDoublesLayout t2_layout(plan);
  const SelectedIntermediateLayout tau_layout(plan,
                                               plan.required_tau_blocks);
  const SelectedIntermediateLayout wmnij_layout(
      plan, plan.required_wmnij_blocks);
  const SelectedIntermediateLayout wmbej_layout(
      plan, plan.required_wmbej_blocks);
  const SelectedIntermediateLayout wabef_layout(
      plan, plan.required_wabef_blocks);
  if (t2_layout.occupied() != occupied ||
      t2_layout.virtuals() != virtuals)
    throw std::invalid_argument(
        "sparse CCSD plan dimensions differ from Hamiltonian");

  if (spatial_fock.dimensions() !=
      std::array<std::size_t, 2>{spatial_occupied + spatial_virtuals,
                                 spatial_occupied + spatial_virtuals})
    throw std::invalid_argument(
        "sparse CCSD spatial Fock dimensions are invalid");
  if (integrals.dimensions() !=
      std::array<std::size_t, 4>{occupied + virtuals, occupied + virtuals,
                                 occupied + virtuals, occupied + virtuals})
    throw std::invalid_argument(
        "sparse CCSD selected integral dimensions are invalid");
  const auto fock = expand_spatial_one_body_to_spin_orbital(spatial_fock);
  const auto& eri = integrals;
  const auto selected_singles = singles_mask(plan, occupied, virtuals);
  const auto occupied_labels =
      labels_from_groups(plan.occupied_orbitals_by_monomer);
  const auto virtual_labels =
      labels_from_groups(plan.virtual_orbitals_by_monomer);
  Matrix d1({occupied, virtuals});
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t a = 0; a < virtuals; ++a) {
      d1(i, a) = fock(i, i) - fock(occupied + a, occupied + a);
      if (std::abs(d1(i, a)) < options.equations.denominator_floor)
        throw std::runtime_error("small sparse CCSD preconditioner");
    }

  GlobalArraysBlockTensor current(t2_layout.distributed_layout(),
                                  "sc_cur");
  GlobalArraysBlockTensor raw(t2_layout.distributed_layout(),
                              "sc_raw");
  GlobalArraysBlockTensor candidate(t2_layout.distributed_layout(),
                                    "sc_cand");
  GlobalArraysBlockTensor residual(t2_layout.distributed_layout(),
                                   "sc_res");
  GlobalArraysBlockTensor tau(tau_layout.distributed_layout(),
                              "sc_tau");
  GlobalArraysBlockTensor tilde_tau(tau_layout.distributed_layout(),
                                    "sc_ttau");
  GlobalArraysBlockTensor wmnij(wmnij_layout.distributed_layout(),
                                "sc_wmn");
  GlobalArraysBlockTensor wmbej(wmbej_layout.distributed_layout(),
                                "sc_wmb");
  SelectedIntermediatePhaseCache wmnij_cache(
      wmnij_layout, wmnij, options.maximum_phase_cache_bytes);
  SelectedIntermediatePhaseCache wmbej_cache(
      wmbej_layout, wmbej, options.maximum_phase_cache_bytes);
  Matrix t1({occupied, virtuals});
  SparseCcsdResult result;
  std::size_t completed_iterations = 0;
  if (options.restart) {
    const auto& restart = *options.restart;
    if (restart.maximum_monomers != options.maximum_monomers ||
        restart.t1.dimensions() !=
            std::array<std::size_t, 2>{occupied, virtuals} ||
        restart.occupied_monomers != occupied_labels ||
        restart.virtual_monomers != virtual_labels)
      throw std::runtime_error(
          "sparse CCSD restart contract differs from current run");
    t1 = restart.t1;
    completed_iterations = restart.completed_iterations;
    result.history = restart.history;
    result.converged = restart.converged;
    load_flattened_restart(restart, t2_layout, current);
  } else {
    for (std::size_t block = 0; block < t2_layout.block_count(); ++block) {
      if (!raw.computation_block_owned(block))
        continue;
      std::vector<double> values(t2_layout.block(block).element_count);
      for (std::size_t flat = 0; flat < values.size(); ++flat) {
        const auto [i, j, a, b] =
            doubles_orbitals(t2_layout, block, flat);
        values[flat] = eri(i, j, occupied + a, occupied + b) /
                       denominator_two(d1, i, j, a, b);
      }
      raw.put_block(block, values);
    }
    raw.synchronize();
    auto raw_blocks = fetch_all(
        raw, options.maximum_phase_cache_bytes,
        result.memory.largest_phase_cache_bytes, 0,
        "initialization/raw_to_current", owned_block_count(current),
        result.phase_statistics, options.phase_observer);
    const DoublesValues raw_values(t2_layout, raw_blocks);
    for (std::size_t block = 0; block < t2_layout.block_count(); ++block) {
      if (!current.computation_block_owned(block))
        continue;
      std::vector<double> values(t2_layout.block(block).element_count);
      for (std::size_t flat = 0; flat < values.size(); ++flat) {
        const auto [i, j, a, b] =
            doubles_orbitals(t2_layout, block, flat);
        values[flat] =
            0.25 * (raw_values(i, j, a, b) - raw_values(j, i, a, b) -
                    raw_values(i, j, b, a) + raw_values(j, i, b, a));
      }
      current.put_block(block, values);
    }
    current.synchronize();
  }

  GlobalArraysBlockDiis diis(options.equations.diis_size,
                             t2_layout.distributed_layout(),
                             "sc_d");
  auto current_blocks = fetch_all(
      current, options.maximum_phase_cache_bytes,
      result.memory.largest_phase_cache_bytes, completed_iterations,
      "energy/current", 0, result.phase_statistics,
      options.phase_observer);
  double old_energy = selected_energy(fock, eri, t1,
                                      DoublesValues(t2_layout, current_blocks),
                                      occupied, virtuals);
  if (options.restart) {
    const auto scale = std::max(1.0, std::abs(old_energy));
    const auto tolerance =
        64.0 * std::numeric_limits<double>::epsilon() * scale;
    if (std::abs(old_energy - options.restart->correlation_energy) > tolerance)
      throw std::runtime_error(
          "sparse CCSD restart energy differs from restored amplitudes");
    if (result.converged &&
        (result.history.empty() ||
         result.history.back().iteration != completed_iterations ||
         std::abs(result.history.back().delta_energy) >=
             options.equations.energy_tolerance ||
         std::max(
             ccsd_residual_measure(options.restart->r1_norm, t1.size(),
                                   options.equations.residual_metric),
             ccsd_residual_measure(
                 options.restart->r2_norm,
                 t2_layout.selected_element_count(),
                 options.equations.residual_metric)) >=
             options.equations.residual_tolerance))
      throw std::runtime_error(
          "saved sparse CCSD state does not satisfy current tolerances");
    result.r1_norm = options.restart->r1_norm;
    result.r2_norm = options.restart->r2_norm;
  }

  for (std::size_t iteration = completed_iterations + 1;
       !result.converged && iteration <= options.equations.max_iterations;
       ++iteration) {
    CcsdIterationTiming iteration_timing;
    const auto iteration_begin = SteadyClock::now();
    const auto intermediates_begin = SteadyClock::now();
    current_blocks = fetch_all(
        current, options.maximum_phase_cache_bytes,
        result.memory.largest_phase_cache_bytes, iteration,
        "iteration/current_to_tau",
        checked_sum(owned_block_count(tau), owned_block_count(tilde_tau),
                    "selected CCSD owned output block count overflows"),
        result.phase_statistics, options.phase_observer);
    const DoublesValues t2(t2_layout, current_blocks);

    tau.zero();
    tilde_tau.zero();
    for (std::size_t block = 0; block < tau_layout.block_count(); ++block) {
      if (!tau.computation_block_owned(block))
        continue;
      std::vector<double> tau_values(tau_layout.block(block).element_count);
      std::vector<double> tilde_values(tau_values.size());
      for (std::size_t flat = 0; flat < tau_values.size(); ++flat) {
        const auto [i, j, a, b] =
            intermediate_orbitals(tau_layout, block, flat);
        const auto cross = t1(i, a) * t1(j, b) -
                           t1(i, b) * t1(j, a);
        tau_values[flat] = t2(i, j, a, b) + cross;
        tilde_values[flat] = t2(i, j, a, b) + 0.5 * cross;
      }
      tau.put_block(block, tau_values);
      tilde_tau.put_block(block, tilde_values);
    }
    tau.synchronize();
    tilde_tau.synchronize();
    const auto owned_intermediate_outputs = checked_sum(
        owned_block_count(wmnij), owned_block_count(wmbej),
        "selected CCSD owned output block count overflows");
    auto tau_blocks = fetch_all(
        tau, options.maximum_phase_cache_bytes,
        result.memory.largest_phase_cache_bytes, iteration,
        "intermediates/tau", owned_intermediate_outputs,
        result.phase_statistics, options.phase_observer);
    auto tilde_blocks = fetch_all(
        tilde_tau, options.maximum_phase_cache_bytes,
        result.memory.largest_phase_cache_bytes, iteration,
        "intermediates/tilde_tau", owned_intermediate_outputs,
        result.phase_statistics, options.phase_observer);
    const IntermediateValues tau_values(tau_layout, tau_blocks);
    const IntermediateValues tilde_values(tau_layout, tilde_blocks);

    Matrix fae({virtuals, virtuals});
    Matrix fmi({occupied, occupied});
    Matrix fme({occupied, virtuals});
    const auto world = parallel::world();
    for (std::size_t flat = world.rank; flat < virtuals * virtuals;
         flat += world.size) {
      const auto a = flat / virtuals;
      const auto e = flat % virtuals;
      double value = a == e ? 0.0 : fock(occupied + a, occupied + e);
      for (std::size_t m = 0; m < occupied; ++m) {
        value -= 0.5 * fock(m, occupied + e) * t1(m, a);
        for (std::size_t q = 0; q < virtuals; ++q)
          value += t1(m, q) *
                   eri(m, occupied + a, occupied + q, occupied + e);
        for (std::size_t n = 0; n < occupied; ++n)
          for (std::size_t q = 0; q < virtuals; ++q)
            value -= 0.5 * tilde_values({m, n, a, q}) *
                     eri(m, n, occupied + e, occupied + q);
      }
      fae(a, e) = value;
    }
    parallel::sum_in_place(fae.data());
    for (std::size_t flat = world.rank; flat < occupied * occupied;
         flat += world.size) {
      const auto m = flat / occupied;
      const auto i = flat % occupied;
      double value = m == i ? 0.0 : fock(m, i);
      for (std::size_t e = 0; e < virtuals; ++e)
        value += 0.5 * t1(i, e) * fock(m, occupied + e);
      for (std::size_t n = 0; n < occupied; ++n)
        for (std::size_t e = 0; e < virtuals; ++e) {
          value += t1(n, e) * eri(m, n, i, occupied + e);
          for (std::size_t q = 0; q < virtuals; ++q)
            value += 0.5 * tilde_values({i, n, e, q}) *
                     eri(m, n, occupied + e, occupied + q);
        }
      fmi(m, i) = value;
    }
    parallel::sum_in_place(fmi.data());
    for (std::size_t flat = world.rank; flat < occupied * virtuals;
         flat += world.size) {
      const auto m = flat / virtuals;
      const auto e = flat % virtuals;
      double value = fock(m, occupied + e);
      for (std::size_t n = 0; n < occupied; ++n)
        for (std::size_t q = 0; q < virtuals; ++q)
          value += t1(n, q) * eri(m, n, occupied + e, occupied + q);
      fme(m, e) = value;
    }
    parallel::sum_in_place(fme.data());

    wmnij.zero();
    for (std::size_t block = 0; block < wmnij_layout.block_count(); ++block) {
      if (!wmnij.computation_block_owned(block))
        continue;
      std::vector<double> values(wmnij_layout.block(block).element_count);
      for (std::size_t flat = 0; flat < values.size(); ++flat) {
        const auto [m, n, i, j] =
            intermediate_orbitals(wmnij_layout, block, flat);
        double value = eri(m, n, i, j);
        for (std::size_t e = 0; e < virtuals; ++e) {
          value += t1(j, e) * eri(m, n, i, occupied + e) -
                   t1(i, e) * eri(m, n, j, occupied + e);
          for (std::size_t q = 0; q < virtuals; ++q)
            value += 0.25 * tau_values({i, j, e, q}) *
                     eri(m, n, occupied + e, occupied + q);
        }
        values[flat] = value;
      }
      wmnij.put_block(block, values);
    }
    wmnij.synchronize();
    wmbej.zero();
    for (std::size_t block = 0; block < wmbej_layout.block_count(); ++block) {
      if (!wmbej.computation_block_owned(block))
        continue;
      std::vector<double> values(wmbej_layout.block(block).element_count);
      for (std::size_t flat = 0; flat < values.size(); ++flat) {
        const auto [m, b, e, j] =
            intermediate_orbitals(wmbej_layout, block, flat);
        double value = eri(m, occupied + b, occupied + e, j);
        for (std::size_t q = 0; q < virtuals; ++q)
          value += t1(j, q) *
                   eri(m, occupied + b, occupied + e, occupied + q);
        for (std::size_t n = 0; n < occupied; ++n) {
          value -= t1(n, b) * eri(m, n, occupied + e, j);
          for (std::size_t q = 0; q < virtuals; ++q)
            value -= (0.5 * t2(j, n, q, b) +
                      t1(j, q) * t1(n, b)) *
                     eri(m, n, occupied + e, occupied + q);
        }
        values[flat] = value;
      }
      wmbej.put_block(block, values);
    }
    wmbej.synchronize();
    prepare_intermediate_phase_cache(
        wmnij_cache, wmnij, all_block_ids(wmnij_layout.block_count()),
        iteration, "doubles/wmnij_to_r2", owned_block_count(raw),
        result.memory.largest_phase_cache_bytes,
        result.phase_statistics, options.phase_observer);
    prepare_intermediate_phase_cache(
        wmbej_cache, wmbej, all_block_ids(wmbej_layout.block_count()),
        iteration, "doubles/wmbej_to_r2", owned_block_count(raw),
        result.memory.largest_phase_cache_bytes,
        result.phase_statistics, options.phase_observer);
    const auto& wmnij_values = wmnij_cache;
    const auto& wmbej_values = wmbej_cache;
    iteration_timing.intermediates_seconds += elapsed_seconds(
        intermediates_begin, SteadyClock::now());

    const auto singles_begin = SteadyClock::now();
    Matrix next_t1({occupied, virtuals});
    for (std::size_t flat = world.rank; flat < occupied * virtuals;
         flat += world.size) {
      const auto i = flat / virtuals;
      const auto a = flat % virtuals;
      if (selected_singles[flat] == 0U)
        continue;
      double value = fock(i, occupied + a);
      for (std::size_t e = 0; e < virtuals; ++e)
        value += t1(i, e) * fae(a, e);
      for (std::size_t m = 0; m < occupied; ++m) {
        value -= t1(m, a) * fmi(m, i);
        for (std::size_t e = 0; e < virtuals; ++e) {
          value += t2(i, m, a, e) * fme(m, e) -
                   t1(m, e) * eri(m, occupied + a, i, occupied + e);
          for (std::size_t q = 0; q < virtuals; ++q)
            value -= 0.5 * t2(i, m, e, q) *
                     eri(m, occupied + a, occupied + e, occupied + q);
          for (std::size_t n = 0; n < occupied; ++n)
            value -= 0.5 * t2(m, n, a, e) *
                     eri(n, m, occupied + e, i);
        }
      }
      next_t1(i, a) = value / d1(i, a);
    }
    parallel::sum_in_place(next_t1.data());
    iteration_timing.singles_seconds +=
        elapsed_seconds(singles_begin, SteadyClock::now());

    const auto doubles_begin = SteadyClock::now();
    raw.zero();
    for (std::size_t block = 0; block < t2_layout.block_count(); ++block) {
      if (!raw.computation_block_owned(block))
        continue;
      std::vector<double> values(t2_layout.block(block).element_count);
      for (std::size_t flat = 0; flat < values.size(); ++flat) {
        const auto [i, j, a, b] =
            doubles_orbitals(t2_layout, block, flat);
        const auto fae_term = [&](const std::size_t ii,
                                  const std::size_t jj,
                                  const std::size_t aa,
                                  const std::size_t bb) {
          double sum = 0.0;
          for (std::size_t e = 0; e < virtuals; ++e) {
            double intermediate = fae(bb, e);
            for (std::size_t m = 0; m < occupied; ++m)
              intermediate -= 0.5 * t1(m, bb) * fme(m, e);
            sum += t2(ii, jj, aa, e) * intermediate;
          }
          return sum;
        };
        const auto fmi_term = [&](const std::size_t ii,
                                  const std::size_t jj,
                                  const std::size_t aa,
                                  const std::size_t bb) {
          double sum = 0.0;
          for (std::size_t m = 0; m < occupied; ++m) {
            double intermediate = fmi(m, jj);
            for (std::size_t e = 0; e < virtuals; ++e)
              intermediate += 0.5 * t1(jj, e) * fme(m, e);
            sum += t2(ii, m, aa, bb) * intermediate;
          }
          return sum;
        };
        const auto wm_term = [&](const std::size_t ii,
                                 const std::size_t jj,
                                 const std::size_t aa,
                                 const std::size_t bb) {
          double sum = 0.0;
          for (std::size_t m = 0; m < occupied; ++m)
            for (std::size_t e = 0; e < virtuals; ++e)
              sum += t2(ii, m, aa, e) * wmbej_values({m, bb, e, jj}) -
                     t1(ii, e) * t1(m, aa) *
                         eri(m, occupied + bb, occupied + e, jj);
          return sum;
        };
        const auto abej = [&](const std::size_t ii,
                              const std::size_t jj,
                              const std::size_t aa,
                              const std::size_t bb) {
          double sum = 0.0;
          for (std::size_t e = 0; e < virtuals; ++e)
            sum += t1(ii, e) *
                   eri(occupied + aa, occupied + bb, occupied + e, jj);
          return sum;
        };
        const auto mbij = [&](const std::size_t ii,
                              const std::size_t jj,
                              const std::size_t aa,
                              const std::size_t bb) {
          double sum = 0.0;
          for (std::size_t m = 0; m < occupied; ++m)
            sum += t1(m, aa) * eri(m, occupied + bb, ii, jj);
          return sum;
        };
        double value = eri(i, j, occupied + a, occupied + b) +
                       fae_term(i, j, a, b) - fae_term(i, j, b, a) -
                       fmi_term(i, j, a, b) + fmi_term(j, i, a, b);
        for (std::size_t m = 0; m < occupied; ++m)
          for (std::size_t n = 0; n < occupied; ++n)
            value += 0.5 * tau_values({m, n, a, b}) *
                     wmnij_values({m, n, i, j});
        value += wm_term(i, j, a, b) - wm_term(j, i, a, b) -
                 wm_term(i, j, b, a) + wm_term(j, i, b, a) +
                 abej(i, j, a, b) - abej(j, i, a, b) -
                 mbij(i, j, a, b) + mbij(i, j, b, a);
        values[flat] = value / denominator_two(d1, i, j, a, b);
      }
      raw.put_block(block, values);
    }
    raw.synchronize();
    iteration_timing.doubles_seconds +=
        elapsed_seconds(doubles_begin, SteadyClock::now());

    // Build each streamed W_abef(a,b,:,:) block once, then reuse it for all
    // selected occupied pairs. Contributions owned by this rank are buffered
    // in selected-T2 block layout and accumulated collectively once per
    // phase. This avoids rebuilding W_abef inside every ijab residual.
    const auto wabef_begin = SteadyClock::now();
    const auto wabef_accumulator_bytes =
        block_bytes(t2_layout.distributed_layout());
    if (wabef_accumulator_bytes > options.maximum_phase_cache_bytes)
      throw std::runtime_error(
          "selected W_abef accumulator exceeds the explicit cache limit");
    result.memory.largest_phase_cache_bytes = std::max(
        result.memory.largest_phase_cache_bytes, wabef_accumulator_bytes);
    std::vector<std::vector<double>> wabef_contributions;
    wabef_contributions.reserve(t2_layout.block_count());
    for (std::size_t block = 0; block < t2_layout.block_count(); ++block)
      wabef_contributions.emplace_back(
          t2_layout.block(block).element_count, 0.0);
    const auto virtual_pairs = checked_product(
        virtuals, virtuals, "selected W_abef pair count overflows");
    for (std::size_t pair = world.rank; pair < virtual_pairs;
         pair += world.size) {
      const auto a = pair / virtuals;
      const auto b = pair % virtuals;
      bool required_output = false;
      for (std::size_t i = 0; i < occupied && !required_output; ++i)
        for (std::size_t j = 0; j < occupied; ++j)
          if (t2_layout.selected(i, j, a, b)) {
            required_output = true;
            break;
          }
      if (!required_output)
        continue;
      std::vector<double> wabef(virtual_pairs, 0.0);
      if (virtual_pairs >
          static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        throw std::overflow_error("selected W_abef block exceeds ptrdiff_t");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (std::ptrdiff_t raw_eq = 0;
           raw_eq < static_cast<std::ptrdiff_t>(virtual_pairs); ++raw_eq) {
        const auto eq = static_cast<std::size_t>(raw_eq);
        const auto e = eq / virtuals;
        const auto q = eq % virtuals;
        if (!wabef_layout.selected({a, b, e, q}))
          continue;
        double value = eri(occupied + a, occupied + b, occupied + e,
                           occupied + q);
        for (std::size_t m = 0; m < occupied; ++m) {
          value -= t1(m, b) *
                       eri(occupied + a, m, occupied + e, occupied + q) -
                   t1(m, a) *
                       eri(occupied + b, m, occupied + e, occupied + q);
          for (std::size_t n = 0; n < occupied; ++n)
            value += 0.25 * tau_values({m, n, a, b}) *
                     eri(m, n, occupied + e, occupied + q);
        }
        wabef[eq] = value;
      }
      const auto occupied_pairs = checked_product(
          occupied, occupied, "selected occupied-pair count overflows");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (std::ptrdiff_t raw_ij = 0;
           raw_ij < static_cast<std::ptrdiff_t>(occupied_pairs); ++raw_ij) {
        const auto ij = static_cast<std::size_t>(raw_ij);
        const auto i = ij / occupied;
        const auto j = ij % occupied;
        if (!t2_layout.selected(i, j, a, b))
          continue;
        double contribution = 0.0;
        for (std::size_t e = 0; e < virtuals; ++e)
          for (std::size_t q = 0; q < virtuals; ++q)
            contribution +=
                0.5 * tau_values({i, j, e, q}) *
                wabef[e * virtuals + q];
        const auto block = t2_layout.block_index(i, j, a, b);
        const auto element = t2_layout.element_index(i, j, a, b);
        wabef_contributions[block][element] +=
            contribution / denominator_two(d1, i, j, a, b);
      }
    }
    for (std::size_t block = 0; block < t2_layout.block_count(); ++block)
      raw.accumulate_block(block, wabef_contributions[block]);
    raw.synchronize();
    iteration_timing.wabef_seconds +=
        elapsed_seconds(wabef_begin, SteadyClock::now());

    const auto residual_begin = SteadyClock::now();
    auto raw_blocks = fetch_all(
        raw, options.maximum_phase_cache_bytes,
        result.memory.largest_phase_cache_bytes, iteration,
        "residual/raw_to_candidate", owned_block_count(candidate),
        result.phase_statistics, options.phase_observer);
    const DoublesValues raw_values(t2_layout, raw_blocks);
    candidate.zero();
    residual.zero();
    for (std::size_t block = 0; block < t2_layout.block_count(); ++block) {
      if (!candidate.computation_block_owned(block))
        continue;
      std::vector<double> next(t2_layout.block(block).element_count);
      for (std::size_t flat = 0; flat < next.size(); ++flat) {
        const auto [i, j, a, b] =
            doubles_orbitals(t2_layout, block, flat);
        next[flat] =
            0.25 * (raw_values(i, j, a, b) - raw_values(j, i, a, b) -
                    raw_values(i, j, b, a) + raw_values(j, i, b, a));
        next[flat] = t2(i, j, a, b) +
                     options.equations.damping *
                         (next[flat] - t2(i, j, a, b));
      }
      candidate.put_block(block, next);
    }
    candidate.synchronize();
    auto candidate_blocks = fetch_all(
        candidate, options.maximum_phase_cache_bytes,
        result.memory.largest_phase_cache_bytes, iteration,
        "residual/candidate_to_error", owned_block_count(residual),
        result.phase_statistics, options.phase_observer);
    const DoublesValues candidate_values(t2_layout, candidate_blocks);
    const auto raw_error = [&](const std::size_t i, const std::size_t j,
                               const std::size_t a,
                               const std::size_t b) {
      return denominator_two(d1, i, j, a, b) *
             (candidate_values(i, j, a, b) - t2(i, j, a, b)) /
             options.equations.damping;
    };
    for (std::size_t block = 0; block < t2_layout.block_count(); ++block) {
      if (!residual.computation_block_owned(block))
        continue;
      std::vector<double> error(t2_layout.block(block).element_count);
      for (std::size_t flat = 0; flat < error.size(); ++flat) {
        const auto [i, j, a, b] =
            doubles_orbitals(t2_layout, block, flat);
        error[flat] =
            0.25 * (raw_error(i, j, a, b) - raw_error(j, i, a, b) -
                    raw_error(i, j, b, a) + raw_error(j, i, b, a));
      }
      residual.put_block(block, error);
    }
    residual.synchronize();
    Matrix r1({occupied, virtuals});
    for (std::size_t index = 0; index < r1.size(); ++index) {
      r1.data()[index] = d1.data()[index] *
                         (next_t1.data()[index] - t1.data()[index]);
      next_t1.data()[index] = t1.data()[index] +
          options.equations.damping *
              (next_t1.data()[index] - t1.data()[index]);
    }
    result.r1_norm = norm(r1.data());
    result.r2_norm = std::sqrt(std::max(0.0, residual.dot(residual)));
    const auto r1_convergence_measure = ccsd_residual_measure(
        result.r1_norm, r1.size(), options.equations.residual_metric);
    const auto r2_convergence_measure = ccsd_residual_measure(
        result.r2_norm, t2_layout.selected_element_count(),
        options.equations.residual_metric);
    iteration_timing.residual_seconds +=
        elapsed_seconds(residual_begin, SteadyClock::now());

    const auto diis_begin = SteadyClock::now();
    if (iteration >= options.equations.diis_start)
      next_t1 = diis.extrapolate_distributed(
          next_t1, candidate, r1, residual, raw);
    else
      raw.copy_from(candidate);
    raw.synchronize();
    iteration_timing.diis_seconds +=
        elapsed_seconds(diis_begin, SteadyClock::now());

    const auto energy_begin = SteadyClock::now();
    auto next_blocks = fetch_all(
        raw, options.maximum_phase_cache_bytes,
        result.memory.largest_phase_cache_bytes, iteration,
        "energy/updated_amplitudes", 0, result.phase_statistics,
        options.phase_observer);
    const auto new_energy = selected_energy(
        fock, eri, next_t1, DoublesValues(t2_layout, next_blocks), occupied,
        virtuals);
    iteration_timing.energy_seconds +=
        elapsed_seconds(energy_begin, SteadyClock::now());
    const auto delta = new_energy - old_energy;
    result.history.push_back(
        {iteration, new_energy, delta, result.r1_norm, result.r2_norm});
    t1 = std::move(next_t1);
    current.copy_from(raw);
    old_energy = new_energy;
    result.converged = std::abs(delta) < options.equations.energy_tolerance &&
                       std::max(r1_convergence_measure,
                                r2_convergence_measure) <
                           options.equations.residual_tolerance;
    iteration_timing.iteration_seconds =
        elapsed_seconds(iteration_begin, SteadyClock::now());
    add_timing(result.cumulative_timing, iteration_timing);
    if (options.equations.iteration_observer)
      options.equations.iteration_observer(CcsdIterationReport{
          iteration, new_energy, delta, result.r1_norm, result.r2_norm,
          r1_convergence_measure, r2_convergence_measure, result.converged,
          CcsdAmplitudeBackend::global_arrays,
          CcsdWabefBackend::cpu, iteration_timing});
    if (options.observer &&
        (result.converged || options.equations.checkpoint_interval == 0 ||
         iteration % options.equations.checkpoint_interval == 0)) {
      result.correlation_energy = old_energy;
      result.t1 = t1;
      result.t2_blocks = next_blocks;
      options.observer(make_sparse_ccsd_checkpoint(
          result, occupied_labels, virtual_labels,
          options.maximum_monomers));
    }
    if (result.converged)
      break;
  }
  result.correlation_energy = old_energy;
  result.t1 = std::move(t1);
  result.t2_blocks = fetch_all(
      current, options.maximum_phase_cache_bytes,
      result.memory.largest_phase_cache_bytes,
      result.history.empty() ? completed_iterations
                             : result.history.back().iteration,
      "final/current_to_result", 0, result.phase_statistics,
      options.phase_observer);
  result.memory.selected_t2_elements =
      t2_layout.distributed_layout().scientific_element_count();
  result.memory.declared_spin_integral_elements =
      eri.declared_spin_element_count();
  result.memory.selected_spatial_integral_elements =
      eri.selected_element_count();
  result.memory.selected_spatial_integral_bytes = eri.selected_bytes();
  result.memory.selected_tau_elements =
      tau_layout.distributed_layout().scientific_element_count();
  result.memory.selected_wmnij_elements =
      wmnij_layout.distributed_layout().scientific_element_count();
  result.memory.selected_wmbej_elements =
      wmbej_layout.distributed_layout().scientific_element_count();
  for (const auto* tensor : {&current, &raw, &candidate, &residual, &tau,
                             &tilde_tau, &wmnij, &wmbej})
    result.memory.global_arrays_allocated_bytes = checked_sum(
        result.memory.global_arrays_allocated_bytes,
        tensor->memory().allocated_global_bytes,
        "sparse CCSD allocated byte count overflows");
  return result;
}

} // namespace localized_cc
