#include "localized_cc/cuda_triples.hpp"

#include <stdexcept>

namespace localized_cc {

CudaTriplesRuntimeInfo cuda_triples_runtime_info() { return {}; }

TriplesEnergyPair reduce_triples_energy_cuda(const TriplesEnergyTile&, int) {
  throw std::runtime_error(
      "CUDA triples backend is unavailable in this CPU-only build");
}

TriplesMomentResult contract_triples_moment_cuda(const TriplesMomentTile&,
                                                  int) {
  throw std::runtime_error(
      "CUDA triples moment backend is unavailable in this CPU-only build");
}

TriplesMomentResult contract_triples_moment_tce_cuda(
    const TriplesMomentTile&, int) {
  throw std::runtime_error(
      "TCE-style CUDA triples moment backend is unavailable in this CPU-only build");
}

CcsdWabefResult contract_ccsd_wabef_cuda(const CcsdWabefTile&, int) {
  throw std::runtime_error(
      "CUDA CCSD W_abef backend is unavailable in this CPU-only build");
}

} // namespace localized_cc
