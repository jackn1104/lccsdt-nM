#pragma once
#include "localized_cc/ccsd.hpp"
namespace localized_cc {
// W_mnij, mixed W_mbej, and W_abef contributions to spatial R2. No
// spin-expanded amplitudes or four-index spin intermediates are allocated.
Tensor4 spatial_tiled_pair_terms(const SpatialRhfHamiltonian& h,
    const SpatialRccsdAmplitudes& amplitudes, const CcsdOptions& options,
    CcsdIterationTiming& timing);
}
