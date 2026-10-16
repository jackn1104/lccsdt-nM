#pragma once

#include "modernqc/mp2/localized_tiled_laplace_mp2.hpp"

#include <cstddef>
#include <optional>

namespace modernqc::mp2 {

struct LocalizedMp2MachineResources {
  std::size_t mpi_ranks{1};
  std::size_t ranks_per_node{1};
  std::size_t threads_per_rank{1};
  std::size_t available_memory_bytes_per_node{0};
};

struct LocalizedMp2Workload {
  std::size_t atom_count{0};
  std::size_t ao_count{0};
  std::size_t active_occupied_count{0};
  std::size_t virtual_count{0};
  std::size_t quadrature_points{0};
  NmExecutionMode execution_mode{NmExecutionMode::full};
};

struct LocalizedMp2ResourceOverrides {
  std::optional<std::size_t> occupied_tile_size;
  std::optional<std::size_t> virtual_tile_size;
  std::optional<std::size_t> memory_limit_bytes_per_rank;
  LocalizedLaplaceMp2Backend backend{
      LocalizedLaplaceMp2Backend::automatic};
};

struct LocalizedMp2ResourcePlan {
  std::size_t nodes{1};
  std::size_t usable_memory_bytes_per_rank{0};
  std::size_t occupied_tile_size{1};
  std::size_t virtual_tile_size{1};
  // Number of Laplace points sharing one AO shell-quartet traversal in the
  // pure four-center localized-direct-selected backend.
  std::size_t quadrature_batch_size{1};
  // Target AO lambda extent of the rank-local first-index intermediate.
  // Shells are never split, so an individual shell may exceed this target.
  std::size_t lambda_ao_block_size{1};
  // Rank-local left-occupied extent used only by the distributed separable
  // AO-to-MO transform. This is independent of the later localized-energy
  // occupied panel size above.
  std::size_t distributed_transform_occupied_count{0};
  std::size_t occupied_batches{1};
  std::size_t virtual_batches{1};
  std::size_t canonical_ovov_bytes{0};
  std::size_t localized_ovov_bytes{0};
  std::size_t estimated_transform_peak_bytes_per_rank{0};
  std::size_t estimated_total_peak_bytes_per_rank{0};
  LocalizedLaplaceMp2Backend backend{
      LocalizedLaplaceMp2Backend::automatic};
  bool memory_was_clamped_to_machine{false};
};

// Selects deterministic, balanced tiles inside a conservative node-local
// memory budget. The planner adapts to resources already granted to the
// process; it does not create MPI ranks or request scheduler resources.
[[nodiscard]] LocalizedMp2ResourcePlan plan_localized_mp2_resources(
    const LocalizedMp2MachineResources& machine,
    const LocalizedMp2Workload& workload,
    const LocalizedMp2ResourceOverrides& overrides = {});

}  // namespace modernqc::mp2
