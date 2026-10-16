#include "localized_cc/legacy_fortran.hpp"
#include "localized_cc/parallel.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(LOCALIZED_CC_HAVE_LEGACY_FORTRAN)
extern "C" {
void lccsdt_legacy_ccsd_energy(std::int64_t rank, std::int64_t ranks,
                               std::int64_t occupied, std::int64_t virtuals,
                               const double* fock, const double* eri,
                               const double* t1, const double* t2,
                               double* energy, int* status);
void lccsdt_legacy_triples_energy(std::int64_t rank, std::int64_t ranks,
                                  std::int64_t count,
                                  const double* left_moment,
                                  const double* response, double* energy,
                                  int* status);
void lccsdt_legacy_wabef(std::int64_t rank, std::int64_t ranks,
                         std::int64_t occupied, std::int64_t virtuals,
                         const double* eri, const double* t1,
                         const double* tau, double* wabef, int* status);
void lccsdt_legacy_wabef_block(std::int64_t occupied,
                               std::int64_t virtuals, std::int64_t a,
                               std::int64_t b, const double* eri,
                               const double* t1, const double* tau,
                               double* wabef, int* status);
void lccsdt_legacy_wabef_block_pair_scalars(
    std::int64_t occupied, std::int64_t virtuals, std::int64_t a,
    std::int64_t b, const double* eri, const double* t1,
    const double* tau_ab, double* wabef, int* status);
}
#endif

namespace localized_cc {
namespace {

#if defined(LOCALIZED_CC_HAVE_LEGACY_FORTRAN)
std::int64_t fortran_extent(const std::size_t value) {
  if (value == 0 ||
      value > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error("invalid legacy Fortran tensor extent");
  return static_cast<std::int64_t>(value);
}

std::int64_t fortran_index(const std::size_t value) {
  if (value >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error("legacy Fortran tensor index overflows");
  return static_cast<std::int64_t>(value);
}
#endif

} // namespace

bool legacy_fortran_backend_available() noexcept {
#if defined(LOCALIZED_CC_HAVE_LEGACY_FORTRAN)
  return true;
#else
  return false;
#endif
}

double legacy_fortran_ccsd_energy(const Matrix& fock, const Tensor4& eri,
                                  const Matrix& t1, const Tensor4& t2,
                                  const std::size_t occupied,
                                  const bool distributed) {
#if defined(LOCALIZED_CC_HAVE_LEGACY_FORTRAN)
  const auto fock_dimensions = fock.dimensions();
  const auto t1_dimensions = t1.dimensions();
  const auto orbitals = fock_dimensions[0];
  if (fock_dimensions[1] != orbitals || occupied >= orbitals ||
      t1_dimensions[0] != occupied ||
      t1_dimensions[1] != orbitals - occupied ||
      eri.dimensions() !=
          std::array<std::size_t, 4>{orbitals, orbitals, orbitals, orbitals} ||
      t2.dimensions() !=
          std::array<std::size_t, 4>{occupied, occupied, orbitals - occupied,
                                     orbitals - occupied})
    throw std::invalid_argument("legacy Fortran CCSD energy dimension mismatch");
  // Reject any allocation whose row-major offset cannot be represented by the
  // signed 64-bit indices used at the ISO_C_BINDING boundary.
  static_cast<void>(fortran_extent(fock.size()));
  static_cast<void>(fortran_extent(eri.size()));
  static_cast<void>(fortran_extent(t1.size()));
  static_cast<void>(fortran_extent(t2.size()));
  double energy = 0.0;
  int status = 0;
  const auto communicator = parallel::world();
  const auto rank = distributed ? communicator.rank : 0;
  const auto ranks = distributed ? communicator.size : 1;
  lccsdt_legacy_ccsd_energy(
      static_cast<std::int64_t>(rank), fortran_extent(ranks),
      fortran_extent(occupied), fortran_extent(orbitals - occupied),
      fock.data().data(), eri.data().data(), t1.data().data(),
      t2.data().data(), &energy, &status);
  if (status != 0)
    throw std::runtime_error("legacy Fortran CCSD energy kernel failed");
  if (distributed) {
    std::vector<double> reduced{energy};
    parallel::sum_in_place(reduced);
    return reduced.front();
  }
  return energy;
#else
  static_cast<void>(fock);
  static_cast<void>(eri);
  static_cast<void>(t1);
  static_cast<void>(t2);
  static_cast<void>(occupied);
  static_cast<void>(distributed);
  throw std::runtime_error(
      "legacy Fortran backend was not enabled at configure time");
#endif
}

double legacy_fortran_triples_energy(const Tensor6& left_moment,
                                     const Tensor6& response,
                                     const bool distributed) {
#if defined(LOCALIZED_CC_HAVE_LEGACY_FORTRAN)
  if (left_moment.dimensions() != response.dimensions())
    throw std::invalid_argument("legacy Fortran triples dimension mismatch");
  double energy = 0.0;
  int status = 0;
  const auto communicator = parallel::world();
  const auto rank = distributed ? communicator.rank : 0;
  const auto ranks = distributed ? communicator.size : 1;
  lccsdt_legacy_triples_energy(
      static_cast<std::int64_t>(rank), fortran_extent(ranks),
      fortran_extent(left_moment.size()), left_moment.data().data(),
      response.data().data(), &energy, &status);
  if (status != 0)
    throw std::runtime_error("legacy Fortran triples energy kernel failed");
  if (distributed) {
    std::vector<double> reduced{energy};
    parallel::sum_in_place(reduced);
    return reduced.front();
  }
  return energy;
#else
  static_cast<void>(left_moment);
  static_cast<void>(response);
  static_cast<void>(distributed);
  throw std::runtime_error(
      "legacy Fortran backend was not enabled at configure time");
#endif
}

Tensor4 legacy_fortran_wabef(const Tensor4& eri, const Matrix& t1,
                             const Tensor4& tau,
                             const std::size_t occupied,
                             const bool distributed) {
#if defined(LOCALIZED_CC_HAVE_LEGACY_FORTRAN)
  const auto t1_dimensions = t1.dimensions();
  const auto virtuals = t1_dimensions[1];
  const auto orbitals = occupied + virtuals;
  if (occupied == 0 || virtuals == 0 || t1_dimensions[0] != occupied ||
      eri.dimensions() !=
          std::array<std::size_t, 4>{orbitals, orbitals, orbitals, orbitals} ||
      tau.dimensions() !=
          std::array<std::size_t, 4>{occupied, occupied, virtuals, virtuals})
    throw std::invalid_argument(
        "legacy Fortran W_abef dimension mismatch");
  static_cast<void>(fortran_extent(eri.size()));
  static_cast<void>(fortran_extent(t1.size()));
  static_cast<void>(fortran_extent(tau.size()));
  Tensor4 wabef({virtuals, virtuals, virtuals, virtuals});
  static_cast<void>(fortran_extent(wabef.size()));
  int status = 0;
  const auto communicator = parallel::world();
  const auto rank = distributed ? communicator.rank : 0;
  const auto ranks = distributed ? communicator.size : 1;
  lccsdt_legacy_wabef(
      static_cast<std::int64_t>(rank), fortran_extent(ranks),
      fortran_extent(occupied), fortran_extent(virtuals), eri.data().data(),
      t1.data().data(), tau.data().data(), wabef.data().data(), &status);
  if (status != 0)
    throw std::runtime_error("legacy Fortran W_abef kernel failed");
  if (distributed)
    parallel::sum_in_place(wabef.data());
  return wabef;
#else
  static_cast<void>(eri);
  static_cast<void>(t1);
  static_cast<void>(tau);
  static_cast<void>(occupied);
  static_cast<void>(distributed);
  throw std::runtime_error(
      "legacy Fortran backend was not enabled at configure time");
#endif
}

void legacy_fortran_wabef_block(const Tensor4& eri, const Matrix& t1,
                                const Tensor4& tau,
                                const std::size_t occupied,
                                const std::size_t a, const std::size_t b,
                                Matrix& output) {
#if defined(LOCALIZED_CC_HAVE_LEGACY_FORTRAN)
  const auto t1_dimensions = t1.dimensions();
  const auto virtuals = t1_dimensions[1];
  const auto orbitals = occupied + virtuals;
  if (occupied == 0 || virtuals == 0 || a >= virtuals || b >= virtuals ||
      t1_dimensions[0] != occupied ||
      eri.dimensions() !=
          std::array<std::size_t, 4>{orbitals, orbitals, orbitals, orbitals} ||
      tau.dimensions() !=
          std::array<std::size_t, 4>{occupied, occupied, virtuals, virtuals} ||
      output.dimensions() !=
          std::array<std::size_t, 2>{virtuals, virtuals})
    throw std::invalid_argument(
        "legacy Fortran W_abef block dimension mismatch");
  static_cast<void>(fortran_extent(eri.size()));
  static_cast<void>(fortran_extent(t1.size()));
  static_cast<void>(fortran_extent(tau.size()));
  int status = 0;
  lccsdt_legacy_wabef_block(
      fortran_extent(occupied), fortran_extent(virtuals), fortran_index(a),
      fortran_index(b), eri.data().data(), t1.data().data(),
      tau.data().data(), output.data().data(), &status);
  if (status != 0)
    throw std::runtime_error("legacy Fortran W_abef block kernel failed");
#else
  static_cast<void>(eri);
  static_cast<void>(t1);
  static_cast<void>(tau);
  static_cast<void>(occupied);
  static_cast<void>(a);
  static_cast<void>(b);
  static_cast<void>(output);
  throw std::runtime_error(
      "legacy Fortran backend was not enabled at configure time");
#endif
}

void legacy_fortran_wabef_block_from_pair_scalars(
    const Tensor4& eri, const Matrix& t1, const Matrix& tau_ab,
    const std::size_t occupied, const std::size_t a, const std::size_t b,
    Matrix& output) {
#if defined(LOCALIZED_CC_HAVE_LEGACY_FORTRAN)
  const auto t1_dimensions = t1.dimensions();
  const auto virtuals = t1_dimensions[1];
  const auto orbitals = occupied + virtuals;
  if (occupied == 0 || virtuals == 0 || a >= virtuals || b >= virtuals ||
      t1_dimensions[0] != occupied ||
      eri.dimensions() !=
          std::array<std::size_t, 4>{orbitals, orbitals, orbitals, orbitals} ||
      tau_ab.dimensions() !=
          std::array<std::size_t, 2>{occupied, occupied} ||
      output.dimensions() !=
          std::array<std::size_t, 2>{virtuals, virtuals})
    throw std::invalid_argument(
        "legacy Fortran distributed W_abef block dimension mismatch");
  static_cast<void>(fortran_extent(eri.size()));
  static_cast<void>(fortran_extent(t1.size()));
  static_cast<void>(fortran_extent(tau_ab.size()));
  int status = 0;
  lccsdt_legacy_wabef_block_pair_scalars(
      fortran_extent(occupied), fortran_extent(virtuals), fortran_index(a),
      fortran_index(b), eri.data().data(), t1.data().data(),
      tau_ab.data().data(), output.data().data(), &status);
  if (status != 0)
    throw std::runtime_error(
        "legacy Fortran distributed W_abef block kernel failed");
#else
  static_cast<void>(eri);
  static_cast<void>(t1);
  static_cast<void>(tau_ab);
  static_cast<void>(occupied);
  static_cast<void>(a);
  static_cast<void>(b);
  static_cast<void>(output);
  throw std::runtime_error(
      "legacy Fortran backend was not enabled at configure time");
#endif
}

} // namespace localized_cc
