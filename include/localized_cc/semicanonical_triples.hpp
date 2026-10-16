#pragma once

#include "localized_cc/canonical_triples.hpp"
#include "localized_cc/frozen_core.hpp"
#include "localized_cc/tensor.hpp"

#include <cstddef>
#include <limits>

namespace localized_cc {

// Exact streamed (T) evaluation for amplitudes converged in a localized
// occupied/virtual basis. The occupied and virtual Fock blocks are
// diagonalized independently, and the Hamiltonian and amplitudes are rotated
// within those unchanged orbital subspaces before calling the streamed
// diagonal-resolvent kernel. No canonical CCSD equations are solved.
struct SemicanonicalTriplesResult {
  CanonicalTriplesResult triples;
  double localized_fock_offdiagonal{};
  double semicanonical_fock_offdiagonal{};
  double transformed_t2_antisymmetry_error{};
  double rotation_peak_additional_mib{};
};

SemicanonicalTriplesResult solve_localized_triples_semicanonical(
    const SpatialRhfHamiltonian &localized_hamiltonian,
    const Matrix &localized_t1, const Tensor4 &localized_t2,
    const CanonicalTriplesOptions &options = {},
    std::size_t maximum_additional_memory_bytes =
        std::numeric_limits<std::size_t>::max());

} // namespace localized_cc
