#include "localized_cc/cuda_triples.hpp"

#include <stdexcept>

namespace localized_cc {

CudaTriplesRuntimeInfo cuda_triples_runtime_info() { return {}; }

TriplesEnergyPair reduce_triples_energy_cuda(const TriplesEnergyTile&, int) {
  throw std::runtime_error(
      "CUDA triples backend was requested but this executable was built "
      "without CCSD_ENABLE_CUDA=ON");
}

TriplesMomentResult contract_triples_moment_cuda(const TriplesMomentTile&,
                                                  int) {
  throw std::runtime_error(
      "CUDA triples moment backend was requested but this executable was "
      "built without CCSD_ENABLE_CUDA=ON");
}

TriplesMomentResult contract_triples_moment_tce_cuda(
    const TriplesMomentTile&, int) {
  throw std::runtime_error(
      "TCE-style CUDA triples moment backend was requested but this "
      "executable was built without CCSD_ENABLE_CUDA=ON");
}

CcsdWabefResult contract_ccsd_wabef_cuda(const CcsdWabefTile&, int) {
  throw std::runtime_error(
      "CUDA CCSD W_abef backend was requested but this executable was built "
      "without CCSD_ENABLE_CUDA=ON");
}

} // namespace localized_cc
