#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace modernqc::mp2 {

constexpr std::size_t maximum_mp2_body_order = 4;

struct MonomerAssignment {
  int monomer_id;
  double confidence;
};

struct MonomerAssignmentQuality {
  std::size_t orbital_count;
  std::size_t low_confidence_count;
  double minimum_confidence;
};

enum class NmExecutionMode {
  full,
  one_two_monomer,
};

struct NmEnergyDecomposition {
  // Index zero is unused; entries 1--4 contain 1M--4M energies in hartree.
  std::array<double, maximum_mp2_body_order + 1> energy_hartree{};
  std::array<std::uint64_t, maximum_mp2_body_order + 1> term_count{};
  std::uint64_t low_confidence_terms{0};
  double confidence_threshold{0.8};

  [[nodiscard]] double total_energy_hartree() const noexcept;
  [[nodiscard]] std::uint64_t total_terms() const noexcept;
};

// Assigns each orbital centroid to its nearest monomer reference position.
// For water-cluster FM analysis, the caller supplies one oxygen position per
// monomer. This is postprocessing, not part of the FM functional.
[[nodiscard]] std::vector<MonomerAssignment>
assign_centroids_to_nearest_monomer_references(
    std::span<const std::array<double, 3>> orbital_centroids_bohr,
    std::span<const std::array<double, 3>> reference_positions_bohr,
    std::span<const int> reference_monomer_ids);

// Every orbital retains a hard label. This summary allows the application to
// reject a localization before energy evaluation when any monomer label is
// less confident than the requested scientific threshold.
[[nodiscard]] MonomerAssignmentQuality summarize_assignment_quality(
    std::span<const MonomerAssignment> assignments,
    double confidence_threshold);

// Returns the number of distinct, nonnegative monomer labels in the
// two-hole--two-particle excitation. Repeated labels count once.
[[nodiscard]] std::size_t excitation_body_order(
    int occupied_i_monomer, int occupied_j_monomer,
    int virtual_a_monomer, int virtual_b_monomer);

// A scheduling predicate only. It never reclassifies a computed term.
[[nodiscard]] bool should_compute_excitation(
    NmExecutionMode mode, int occupied_i_monomer,
    int occupied_j_monomer, int virtual_a_monomer,
    int virtual_b_monomer);

// Deterministically accumulates already-defined localized-orbital MP2 terms.
// Every assignment is hard; confidence below the threshold is diagnostic.
[[nodiscard]] NmEnergyDecomposition decompose_localized_mp2_terms(
    std::span<const MonomerAssignment> occupied_assignments,
    std::span<const MonomerAssignment> virtual_assignments,
    std::span<const double> term_energy_hartree,
    double confidence_threshold = 0.8);

}  // namespace modernqc::mp2
