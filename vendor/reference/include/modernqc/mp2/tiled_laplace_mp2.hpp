#pragma once

#include "modernqc/integrals/integral_provider.hpp"
#include "modernqc/linalg/matrix.hpp"
#include "modernqc/mp2/laplace_fit.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace modernqc::mp2 {

struct TiledLaplaceMp2Options {
  std::size_t occupied_tile_size{2};
  std::size_t virtual_tile_size{8};
  std::size_t maximum_additional_memory_bytes{
      512U * 1024U * 1024U};
  std::size_t checkpoint_interval_tiles{8};
  std::size_t threads{1};
  std::size_t mpi_rank{0};
  std::size_t mpi_ranks{1};
  double minimum_positive_denominator{1.0e-10};
  double eri_schwarz_threshold{0.0};
};

struct TiledLaplaceMp2Progress {
  std::uint64_t fingerprint{0};
  // Global tile-ordinal order. A completed entry has exactly one persisted
  // energy contribution, independent of its former MPI owner.
  std::vector<std::uint8_t> completed;
  std::vector<double> tile_energy_contributions;
};

using TiledLaplaceMp2Observer =
    std::function<void(const TiledLaplaceMp2Progress&)>;

struct TiledLaplaceMp2Result {
  double correlation_energy;
  double minimum_denominator;
  double maximum_denominator;
  double minimum_scaling_exponent;
  double maximum_scaling_exponent;
  std::size_t quadrature_points;
  std::size_t total_tiles;
  std::size_t initially_completed_tiles;
  std::size_t locally_computed_tiles;
  std::uint64_t minimum_tiles_per_rank;
  std::uint64_t maximum_tiles_per_rank;
  std::uint64_t direct_transform_count;
  std::uint64_t exchange_transform_count;
  std::uint64_t reused_exchange_tile_count;
  std::uint64_t ordered_shell_quartets_per_transform;
  std::uint64_t evaluated_shell_quartets;
  std::uint64_t schwarz_screened_shell_quartets;
  double eri_schwarz_threshold;
  std::size_t estimated_peak_additional_bytes;
  std::uint64_t fingerprint;
  TiledLaplaceMp2Progress progress;
};

// Production Phase 11 canonical-orbital Laplace MP2. AO integrals are
// transformed directly into bounded OVOV tiles; no AO or global OVOV
// four-index tensor is formed. MPI ownership is cyclic in global tile order.
[[nodiscard]] TiledLaplaceMp2Result compute_tiled_laplace_mp2(
    const integrals::Libint2IntegralProvider& provider,
    const linalg::Matrix& canonical_coefficients,
    const std::vector<double>& orbital_energies,
    const std::vector<std::size_t>& active_occupied_indices,
    const std::vector<std::size_t>& virtual_indices,
    const LaplaceFitResult& fit,
    const TiledLaplaceMp2Options& options = {},
    const TiledLaplaceMp2Progress* restart = nullptr,
    const TiledLaplaceMp2Observer& observer = {});

}  // namespace modernqc::mp2
