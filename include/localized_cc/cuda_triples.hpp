#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace localized_cc {

// One bounded TCE-style six-index tile. Axis order is
// (h1,h2,h3,p4,p5,p6), where h denotes an occupied spin-orbital range and p
// denotes a virtual spin-orbital range. The moments use row-major storage with
// p6 contiguous. Energies and results are in hartree.
struct TriplesEnergyTile {
  std::array<std::size_t, 6> dimensions{};
  std::array<std::vector<double>, 6> orbital_energies;
  std::vector<double> doubles_moment;
  std::vector<double> singles_moment;
  double symmetry_factor{1.0};
  double denominator_floor{1.0e-10};
};

struct TriplesEnergyPair {
  double bracket{};       // CCSD[T] tile contribution
  double parenthetical{}; // CCSD(T) tile contribution
  std::size_t workspace_bytes{};
  int device{-1};
  bool used_cuda{};
};

struct CudaTriplesRuntimeInfo {
  bool compiled{};
  bool available{};
  int visible_devices{};
  int selected_device{-1};
  std::string device_name;
  std::size_t free_bytes{};
  std::size_t total_bytes{};
};

// One occupied-triple/virtual-tile contraction. The three virtual axes may
// refer to different bounded ranges of the complete virtual space. For each
// occupied permutation, data are packed only for the ranges required by the
// six virtual permutations. No o^3 v^3 tensor is materialized.
struct TriplesMomentPermutationTile {
  // [axis][local_a, e]
  std::array<std::vector<double>, 3> t2_jk_axis_e;
  // [virtual permutation][local_b, local_c, e]
  std::array<std::vector<double>, 6> eri_pair_e;
  // [virtual permutation][m, local_b, local_c]
  std::array<std::vector<double>, 6> t2_m_pair;
  // [axis][m, local_a]
  std::array<std::vector<double>, 3> eri_m_axis;
  // [axis][local_a]
  std::array<std::vector<double>, 3> t1_axis;
  // [virtual permutation][local_b, local_c]
  std::array<std::vector<double>, 6> eri_pair_jk;
  // [axis][local_a]
  std::array<std::vector<double>, 3> fock_axis;
};

struct TriplesMomentTile {
  std::size_t occupied_count{};
  std::size_t virtual_count{};
  std::array<std::size_t, 3> virtual_offsets{};
  std::array<std::size_t, 3> virtual_dimensions{};
  std::array<double, 3> occupied_energies{};
  std::array<std::vector<double>, 3> virtual_energies;
  std::array<TriplesMomentPermutationTile, 6> occupied_permutations;
  // CPU-only aliases for eri_pair_e and t2_m_pair. All other operands remain
  // permutation-specific. Identity preserves the CUDA/TCE packing contract.
  std::array<std::size_t, 6> pair_operand_sources{0, 1, 2, 3, 4, 5};
  double denominator_floor{1.0e-10};
  // CPU-only external-index stride. Two denotes a homogeneous alpha/beta
  // axis in the interleaved spin convention. Internal sums remain complete.
  std::size_t virtual_stride{1};
  // CPU homogeneous-spin tiles from the spatial Hamiltonian only. This
  // permits skipping identically zero integral contractions, not screening.
  bool exact_spin_channels{false};
  // Set only by the spatial-Hamiltonian CPU packer: G(bc,ei)=-G(cb,ei).
  // Reuse the virtual-sum DGEMM, not the unrelated occupied-sum operand.
  bool antisymmetric_virtual_integrals{false};
  std::array<std::size_t,3> occupied_spins{};
};

struct TriplesMomentResult {
  double parenthetical{}; // CCSD(T) contribution only
  // Dense row-major virtual tile. Entries not satisfying global a<b<c are
  // zero. left_moment stores D+S, matching the parenthetical numerator.
  std::vector<double> doubles_moment;
  std::vector<double> left_moment;
  std::size_t workspace_bytes{};
  std::size_t evaluated_virtual_triples{};
  int device{-1};
  bool used_cuda{};
};

// Bounded CCSD W_abef contraction tile. tau is occupied-pair-major with a
// contiguous virtual-pair contraction dimension. wabef is task-major with the
// same contraction dimension. The result is task-major/occupied-pair-major.
// This accelerates only the dominant final W_abef dot products; construction
// of the W blocks and all other CCSD residual terms remain on the host.
struct CcsdWabefTile {
  std::size_t occupied_pairs{};
  std::size_t virtual_pair_elements{};
  std::size_t virtual_pair_tasks{};
  std::vector<double> tau;
  std::vector<double> wabef;
};

struct CcsdWabefResult {
  std::vector<double> contractions;
  std::size_t workspace_bytes{};
  int device{-1};
  bool used_cuda{};
};

[[nodiscard]] std::size_t
triples_energy_tile_size(const TriplesEnergyTile& tile);
void validate_triples_energy_tile(const TriplesEnergyTile& tile);

// Fixed row-major traversal used as the numerical oracle for the CUDA tile
// reduction. This preserves the NWChem TCE signs and denominators:
// D^2/Delta for CCSD[T] and D(D+S)/Delta for CCSD(T).
[[nodiscard]] TriplesEnergyPair
reduce_triples_energy_cpu(const TriplesEnergyTile& tile);

[[nodiscard]] CudaTriplesRuntimeInfo cuda_triples_runtime_info();

// device < 0 selects CCSD_CUDA_DEVICE when set, otherwise
// SLURM_LOCALID modulo the visible CUDA device count. Explicit CUDA requests
// fail rather than silently falling back to the CPU.
[[nodiscard]] TriplesEnergyPair
reduce_triples_energy_cuda(const TriplesEnergyTile& tile, int device = -1);

void validate_triples_moment_tile(const TriplesMomentTile& tile,
                                  bool allow_shared_pairs = false);

// Independent row-major oracle for the fused CUDA moment-construction and
// CCSD(T)-only energy kernel.
[[nodiscard]] TriplesMomentResult
contract_triples_moment_cpu(const TriplesMomentTile& tile);

// Forms the doubles and singles triples moments and immediately reduces the
// parenthetical CCSD(T) energy. An explicit CUDA request never falls back.
[[nodiscard]] TriplesMomentResult
contract_triples_moment_cuda(const TriplesMomentTile& tile, int device = -1);

// TCE-style binary64 GPU path. The packed input convention and returned
// moments are identical to contract_triples_moment_cpu. cuBLAS forms the
// bounded a-by-(bc) direct and exchange panels, after which a deterministic
// CUDA kernel applies P(ijk)P(abc), forms D+S, and reduces CCSD(T).
[[nodiscard]] TriplesMomentResult
contract_triples_moment_tce_cuda(const TriplesMomentTile& tile,
                                 int device = -1);

void validate_ccsd_wabef_tile(const CcsdWabefTile& tile);
[[nodiscard]] CcsdWabefResult
contract_ccsd_wabef_cpu(const CcsdWabefTile& tile);
[[nodiscard]] CcsdWabefResult
contract_ccsd_wabef_cuda(const CcsdWabefTile& tile, int device = -1);

} // namespace localized_cc
