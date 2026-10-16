#include "modernqc/scf/rhf.hpp"

#include "modernqc/linalg/eigensolver.hpp"
#include "modernqc/linalg/matrix.hpp"
#include "modernqc/linalg/orthogonalization.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef MODERNQC_HAS_MPI
#include <mpi.h>
#endif

namespace modernqc::scf {
namespace {

struct Orbitals {
  linalg::Matrix ao_coefficients;
  linalg::Matrix orthonormal_coefficients;
  std::vector<double> energies;
};

[[nodiscard]] double matrix_dot(const linalg::Matrix& left,
                                const linalg::Matrix& right) {
  if(left.rows() != right.rows() || left.columns() != right.columns()) {
    throw std::invalid_argument(
        "matrix inner product requires equal dimensions");
  }
  double result = 0.0;
  for(std::size_t index = 0; index < left.size(); ++index) {
    result += left.values()[index] * right.values()[index];
  }
  if(!std::isfinite(result)) {
    throw std::runtime_error("matrix inner product is nonfinite");
  }
  return result;
}

[[nodiscard]] double rms(const linalg::Matrix& matrix) {
  if(matrix.empty()) {
    throw std::invalid_argument("RMS norm requires a nonempty matrix");
  }
  const double element_count = static_cast<double>(matrix.size());
  return linalg::frobenius_norm(matrix) / std::sqrt(element_count);
}

[[nodiscard]] linalg::Matrix commutator_error(
    const linalg::Matrix& fock, const linalg::Matrix& density,
    const linalg::Matrix& overlap) {
  const linalg::Matrix fock_density_overlap =
      linalg::multiply(linalg::multiply(fock, density), overlap);
  const linalg::Matrix overlap_density_fock =
      linalg::multiply(linalg::multiply(overlap, density), fock);
  return linalg::subtract(fock_density_overlap, overlap_density_fock);
}

[[nodiscard]] linalg::Matrix transform_symmetric(
    const linalg::Matrix& matrix, const linalg::Matrix& transformation) {
  linalg::Matrix result = linalg::multiply(
      linalg::transpose(transformation),
      linalg::multiply(matrix, transformation));
  linalg::symmetrize_in_place(result);
  return result;
}

[[nodiscard]] Orbitals diagonalize_fock(
    const linalg::Matrix& fock, const linalg::Matrix& orthogonalizer,
    const linalg::Matrix* previous_orthonormal_coefficients,
    std::size_t occupied_orbitals, double level_shift) {
  linalg::Matrix transformed =
      transform_symmetric(fock, orthogonalizer);
  if(level_shift > 0.0 && previous_orthonormal_coefficients != nullptr) {
    if(previous_orthonormal_coefficients->rows() != transformed.rows() ||
       previous_orthonormal_coefficients->columns() != transformed.rows()) {
      throw std::invalid_argument(
          "previous orthonormal orbital dimensions are inconsistent");
    }
    linalg::Matrix occupied_projector{transformed.rows(), transformed.rows()};
    for(std::size_t occupied = 0; occupied < occupied_orbitals; ++occupied) {
      for(std::size_t column = 0; column < transformed.columns(); ++column) {
        for(std::size_t row = 0; row < transformed.rows(); ++row) {
          occupied_projector(row, column) +=
              (*previous_orthonormal_coefficients)(row, occupied) *
              (*previous_orthonormal_coefficients)(column, occupied);
        }
      }
    }
    for(std::size_t column = 0; column < transformed.columns(); ++column) {
      for(std::size_t row = 0; row < transformed.rows(); ++row) {
        const double virtual_projector =
            (row == column ? 1.0 : 0.0) -
            occupied_projector(row, column);
        transformed(row, column) += level_shift * virtual_projector;
      }
    }
    linalg::symmetrize_in_place(transformed);
  }

  linalg::SymmetricEigendecomposition eigensystem =
      linalg::diagonalize_symmetric(transformed, 2.0e-12);
  linalg::Matrix ao_coefficients =
      linalg::multiply(orthogonalizer, eigensystem.eigenvectors);
  return Orbitals{
      .ao_coefficients = std::move(ao_coefficients),
      .orthonormal_coefficients = std::move(eigensystem.eigenvectors),
      .energies = std::move(eigensystem.eigenvalues),
  };
}

[[nodiscard]] linalg::Matrix density_from_coefficients(
    const linalg::Matrix& coefficients, std::size_t occupied_orbitals) {
  if(occupied_orbitals == 0 || occupied_orbitals > coefficients.columns()) {
    throw std::invalid_argument(
        "occupied orbital count is incompatible with coefficients");
  }
  linalg::Matrix occupied{coefficients.rows(), occupied_orbitals};
  for(std::size_t column = 0; column < occupied_orbitals; ++column) {
    for(std::size_t row = 0; row < coefficients.rows(); ++row) {
      occupied(row, column) = coefficients(row, column);
    }
  }
  linalg::Matrix density = linalg::scaled(
      linalg::multiply(occupied, linalg::transpose(occupied)), 2.0);
  linalg::symmetrize_in_place(density);
  return density;
}

[[nodiscard]] double electronic_energy(const linalg::Matrix& density,
                                       const linalg::Matrix& core,
                                       const linalg::Matrix& fock) {
  if(density.rows() != core.rows() || density.columns() != core.columns() ||
     density.rows() != fock.rows() || density.columns() != fock.columns()) {
    throw std::invalid_argument(
        "RHF energy matrices have inconsistent dimensions");
  }
  double result = 0.0;
  for(std::size_t index = 0; index < density.size(); ++index) {
    result += 0.5 * density.values()[index] *
              (core.values()[index] + fock.values()[index]);
  }
  if(!std::isfinite(result)) {
    throw std::runtime_error("RHF electronic energy is nonfinite");
  }
  return result;
}

class Diis {
 public:
  explicit Diis(std::size_t maximum_subspace)
      : maximum_subspace_{maximum_subspace} {}

  void add(linalg::Matrix fock, linalg::Matrix error) {
    fock_.push_back(std::move(fock));
    error_.push_back(std::move(error));
    while(fock_.size() > maximum_subspace_) {
      fock_.pop_front();
      error_.pop_front();
    }
  }

  [[nodiscard]] std::optional<linalg::Matrix> extrapolate() {
    while(fock_.size() >= 2) {
      const std::size_t count = fock_.size();
      linalg::Matrix coefficients{count + 1, count + 1};
      linalg::Matrix right{count + 1, 1};
      for(std::size_t column = 0; column < count; ++column) {
        for(std::size_t row = 0; row < count; ++row) {
          coefficients(row, column) =
              matrix_dot(error_[row], error_[column]);
        }
        coefficients(column, count) = 1.0;
        coefficients(count, column) = 1.0;
      }
      right(count, 0) = 1.0;
      try {
        const linalg::Matrix solution =
            linalg::solve_linear_system(std::move(coefficients),
                                        std::move(right));
        linalg::Matrix result{fock_.front().rows(),
                              fock_.front().columns()};
        for(std::size_t item = 0; item < count; ++item) {
          const double weight = solution(item, 0);
          for(std::size_t index = 0; index < result.size(); ++index) {
            result.values()[index] +=
                weight * fock_[item].values()[index];
          }
        }
        linalg::symmetrize_in_place(result);
        linalg::require_finite(result, "DIIS-extrapolated Fock matrix");
        return result;
      } catch(const std::runtime_error&) {
        fock_.pop_front();
        error_.pop_front();
      }
    }
    return std::nullopt;
  }

 private:
  std::size_t maximum_subspace_;
  std::deque<linalg::Matrix> fock_;
  std::deque<linalg::Matrix> error_;
};

void validate_options(const RhfOptions& options) {
  if(options.maximum_iterations <= 0 || options.diis_start < 1 ||
     options.diis_subspace < 2) {
    throw std::invalid_argument(
        "RHF iteration and DIIS integer settings are invalid");
  }
  for(const double tolerance :
      {options.energy_tolerance, options.density_rms_tolerance,
       options.commutator_rms_tolerance,
       options.linear_dependency_tolerance}) {
    if(!std::isfinite(tolerance) || tolerance <= 0.0) {
      throw std::invalid_argument(
          "RHF convergence tolerances must be finite and positive");
    }
  }
  if(!std::isfinite(options.damping) || options.damping < 0.0 ||
     options.damping >= 1.0) {
    throw std::invalid_argument("RHF damping must satisfy 0 <= damping < 1");
  }
  if(!std::isfinite(options.level_shift) || options.level_shift < 0.0) {
    throw std::invalid_argument(
        "RHF level shift must be finite and nonnegative");
  }
  if(!std::isfinite(options.eri_schwarz_threshold) ||
     options.eri_schwarz_threshold < 0.0 ||
     !std::isfinite(options.fock_contribution_threshold) ||
     options.fock_contribution_threshold < 0.0) {
    throw std::invalid_argument(
        "RHF screening thresholds must be finite and nonnegative");
  }
  if(options.mpi_ranks == 0 || options.mpi_rank >= options.mpi_ranks ||
     options.fock_threads == 0) {
    throw std::invalid_argument("RHF parallel topology is invalid");
  }
#ifdef MODERNQC_HAS_MPI
  if(options.mpi_ranks > 1) {
    int initialized = 0;
    if(MPI_Initialized(&initialized) != MPI_SUCCESS || initialized == 0) {
      throw std::runtime_error(
          "multi-rank RHF requires initialized MPI");
    }
    int communicator_rank = 0;
    int communicator_size = 1;
    if(MPI_Comm_rank(MPI_COMM_WORLD, &communicator_rank) != MPI_SUCCESS ||
       MPI_Comm_size(MPI_COMM_WORLD, &communicator_size) != MPI_SUCCESS) {
      throw std::runtime_error("unable to query MPI_COMM_WORLD topology");
    }
    if(communicator_rank < 0 || communicator_size <= 0 ||
       static_cast<std::size_t>(communicator_rank) != options.mpi_rank ||
       static_cast<std::size_t>(communicator_size) != options.mpi_ranks) {
      throw std::invalid_argument(
          "RHF topology does not match MPI_COMM_WORLD");
    }
  }
#else
  if(options.mpi_ranks != 1 || options.mpi_rank != 0) {
    throw std::invalid_argument(
        "multi-rank RHF requires MPI support");
  }
#endif
}

#ifdef MODERNQC_HAS_MPI
void check_mpi(int status, const char* operation) {
  if(status != MPI_SUCCESS) {
    throw std::runtime_error(std::string{"MPI failure while "} + operation);
  }
}

void propagate_collective_failure(bool local_failure) {
  int failed = local_failure ? 1 : 0;
  check_mpi(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD),
            "propagating a direct-Fock failure");
  if(failed != 0) {
    throw std::runtime_error(
        local_failure
            ? "direct Fock construction failed on this MPI rank"
            : "direct Fock construction failed on another MPI rank");
  }
}

void allreduce_matrix_sum(linalg::Matrix& matrix) {
  std::size_t offset = 0;
  while(offset < matrix.size()) {
    const std::size_t remaining = matrix.size() - offset;
    const int count = static_cast<int>(
        std::min(remaining,
                 static_cast<std::size_t>(
                     std::numeric_limits<int>::max())));
    check_mpi(MPI_Allreduce(MPI_IN_PLACE, matrix.data() + offset, count,
                            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
              "reducing a Fock matrix");
    offset += static_cast<std::size_t>(count);
  }
}

[[nodiscard]] integrals::DirectFockResult reduce_direct_fock(
    integrals::DirectFockResult local) {
  allreduce_matrix_sum(local.two_electron);
  std::array<unsigned long long, 5> counts{
      static_cast<unsigned long long>(local.owned_shell_quartets),
      static_cast<unsigned long long>(local.evaluated_shell_quartets),
      static_cast<unsigned long long>(local.screened_shell_quartets),
      static_cast<unsigned long long>(
          local.schwarz_screened_shell_quartets),
      static_cast<unsigned long long>(
          local.density_screened_shell_quartets),
  };
  check_mpi(MPI_Allreduce(MPI_IN_PLACE, counts.data(),
                          static_cast<int>(counts.size()),
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
            "reducing direct-Fock counters");
  check_mpi(
      MPI_Allreduce(MPI_IN_PLACE,
                    &local.screened_fock_maximum_error_bound, 1, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD),
      "reducing the screened-Fock error bound");
  local.owned_shell_quartets =
      static_cast<std::uint64_t>(counts[0]);
  local.evaluated_shell_quartets =
      static_cast<std::uint64_t>(counts[1]);
  local.screened_shell_quartets =
      static_cast<std::uint64_t>(counts[2]);
  local.schwarz_screened_shell_quartets =
      static_cast<std::uint64_t>(counts[3]);
  local.density_screened_shell_quartets =
      static_cast<std::uint64_t>(counts[4]);
  return local;
}

[[nodiscard]] std::pair<std::uint64_t, std::uint64_t>
owned_quartet_range(std::uint64_t local_owned) {
  unsigned long long minimum =
      static_cast<unsigned long long>(local_owned);
  unsigned long long maximum = minimum;
  check_mpi(MPI_Allreduce(MPI_IN_PLACE, &minimum, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD),
            "reducing minimum direct-Fock workload");
  check_mpi(MPI_Allreduce(MPI_IN_PLACE, &maximum, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD),
            "reducing maximum direct-Fock workload");
  return {static_cast<std::uint64_t>(minimum),
          static_cast<std::uint64_t>(maximum)};
}

[[nodiscard]] std::pair<double, double> fock_time_range(
    double local_seconds) {
  double minimum = local_seconds;
  double maximum = local_seconds;
  check_mpi(MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_DOUBLE, MPI_MIN,
                          MPI_COMM_WORLD),
            "reducing minimum direct-Fock time");
  check_mpi(MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD),
            "reducing maximum direct-Fock time");
  return {minimum, maximum};
}
#endif

}  // namespace

RhfResult run_rhf(
    const molecule::Molecule& molecule,
    const integrals::Libint2IntegralProvider& integral_provider,
    const RhfOptions& options,
    const std::optional<linalg::Matrix>& initial_density,
    const RhfObserver& observer,
    const RhfDensityObserver& density_observer) {
  validate_options(options);
  const integrals::OneElectronIntegrals one_electron =
      integral_provider.compute_one_electron();
  const linalg::Orthogonalization orthogonalization =
      linalg::build_orthogonalizer(
          one_electron.overlap,
          linalg::OrthogonalizationMethod::canonical,
          options.linear_dependency_tolerance);
  const std::size_t occupied_orbitals = molecule.occupied_orbitals_rhf();
  const std::size_t orbital_count =
      orthogonalization.transformation.columns();
  if(occupied_orbitals > orbital_count) {
    throw std::runtime_error(
        "linear-dependency removal leaves too few orbitals for the electrons");
  }

  Orbitals previous_orbitals =
      diagonalize_fock(one_electron.core_hamiltonian,
                       orthogonalization.transformation, nullptr,
                       occupied_orbitals, 0.0);
  linalg::Matrix density =
      density_from_coefficients(previous_orbitals.ao_coefficients,
                                occupied_orbitals);
  if(initial_density) {
    if(initial_density->rows() != density.rows() ||
       initial_density->columns() != density.columns()) {
      throw std::invalid_argument(
          "restart density dimensions do not match the AO basis");
    }
    linalg::require_finite(*initial_density, "restart density");
    if(linalg::maximum_asymmetry(*initial_density) > 1.0e-12) {
      throw std::invalid_argument(
          "restart density exceeds the symmetry tolerance");
    }
    density = *initial_density;
  }

  Diis diis{static_cast<std::size_t>(options.diis_subspace)};
  const double nuclear_repulsion = molecule.nuclear_repulsion_energy();
  double previous_energy = std::numeric_limits<double>::quiet_NaN();
  std::vector<RhfIteration> history;
  history.reserve(static_cast<std::size_t>(options.maximum_iterations));
  std::uint64_t evaluated_quartets_total = 0;
  std::uint64_t screened_quartets_total = 0;
  std::uint64_t schwarz_screened_quartets_total = 0;
  std::uint64_t density_screened_quartets_total = 0;
  std::uint64_t unique_quartets_per_fock = 0;
  double final_screened_fock_error_bound = 0.0;
  std::uint64_t minimum_owned_quartets_per_fock = 0;
  std::uint64_t maximum_owned_quartets_per_fock = 0;
  double direct_fock_wall_seconds_total = 0.0;
  double minimum_direct_fock_wall_seconds_last = 0.0;
  double maximum_direct_fock_wall_seconds_last = 0.0;

  for(int iteration = 1; iteration <= options.maximum_iterations;
      ++iteration) {
    std::optional<integrals::DirectFockResult> local_direct;
    std::exception_ptr direct_error;
    const auto fock_start = std::chrono::steady_clock::now();
    try {
      local_direct = integral_provider.build_rhf_two_electron(
          density,
          integrals::ScreeningOptions{
              .schwarz_threshold = options.eri_schwarz_threshold,
              .density_aware = options.density_screen,
              .fock_contribution_threshold =
                  options.fock_contribution_threshold,
          },
          integrals::ParallelBuildOptions{
              .rank = options.mpi_rank,
              .ranks = options.mpi_ranks,
              .threads = options.fock_threads,
          });
    } catch(...) {
      direct_error = std::current_exception();
    }
    const double local_fock_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                     fock_start)
            .count();
#ifdef MODERNQC_HAS_MPI
    if(options.mpi_ranks > 1) {
      propagate_collective_failure(direct_error != nullptr);
    }
#endif
    if(direct_error) {
      std::rethrow_exception(direct_error);
    }
    if(!local_direct) {
      throw std::runtime_error(
          "direct Fock construction returned no result");
    }
    const std::uint64_t local_owned =
        local_direct->owned_shell_quartets;
    integrals::DirectFockResult direct = std::move(*local_direct);
#ifdef MODERNQC_HAS_MPI
    if(options.mpi_ranks > 1) {
      const auto [minimum_owned, maximum_owned] =
          owned_quartet_range(local_owned);
      const auto [minimum_seconds, maximum_seconds] =
          fock_time_range(local_fock_seconds);
      minimum_owned_quartets_per_fock = minimum_owned;
      maximum_owned_quartets_per_fock = maximum_owned;
      minimum_direct_fock_wall_seconds_last = minimum_seconds;
      maximum_direct_fock_wall_seconds_last = maximum_seconds;
      direct_fock_wall_seconds_total += maximum_seconds;
      direct = reduce_direct_fock(std::move(direct));
    } else {
      minimum_owned_quartets_per_fock = local_owned;
      maximum_owned_quartets_per_fock = local_owned;
      minimum_direct_fock_wall_seconds_last = local_fock_seconds;
      maximum_direct_fock_wall_seconds_last = local_fock_seconds;
      direct_fock_wall_seconds_total += local_fock_seconds;
    }
#else
    minimum_owned_quartets_per_fock = local_owned;
    maximum_owned_quartets_per_fock = local_owned;
    minimum_direct_fock_wall_seconds_last = local_fock_seconds;
    maximum_direct_fock_wall_seconds_last = local_fock_seconds;
    direct_fock_wall_seconds_total += local_fock_seconds;
#endif
    unique_quartets_per_fock = direct.unique_shell_quartets;
    if(direct.owned_shell_quartets !=
       direct.unique_shell_quartets) {
      throw std::runtime_error(
          "global direct-Fock ownership does not cover every quartet");
    }
    final_screened_fock_error_bound =
        direct.screened_fock_maximum_error_bound;
    if(direct.screened_shell_quartets !=
       direct.schwarz_screened_shell_quartets +
           direct.density_screened_shell_quartets) {
      throw std::runtime_error(
          "direct Fock screening counters are inconsistent");
    }
    if(evaluated_quartets_total >
           std::numeric_limits<std::uint64_t>::max() -
               direct.evaluated_shell_quartets ||
       screened_quartets_total >
           std::numeric_limits<std::uint64_t>::max() -
               direct.screened_shell_quartets ||
       schwarz_screened_quartets_total >
           std::numeric_limits<std::uint64_t>::max() -
               direct.schwarz_screened_shell_quartets ||
       density_screened_quartets_total >
           std::numeric_limits<std::uint64_t>::max() -
               direct.density_screened_shell_quartets) {
      throw std::overflow_error("RHF shell-quartet counter overflow");
    }
    evaluated_quartets_total += direct.evaluated_shell_quartets;
    screened_quartets_total += direct.screened_shell_quartets;
    schwarz_screened_quartets_total +=
        direct.schwarz_screened_shell_quartets;
    density_screened_quartets_total +=
        direct.density_screened_shell_quartets;
    linalg::Matrix raw_fock =
        linalg::add(one_electron.core_hamiltonian, direct.two_electron);
    const linalg::Matrix ao_error =
        commutator_error(raw_fock, density, one_electron.overlap);
    const linalg::Matrix orthonormal_error =
        linalg::multiply(
            linalg::transpose(orthogonalization.transformation),
            linalg::multiply(ao_error,
                             orthogonalization.transformation));
    const double gradient_rms = rms(orthonormal_error);
    const double energy =
        electronic_energy(density, one_electron.core_hamiltonian, raw_fock);
    const double energy_change =
        iteration == 1 ? std::numeric_limits<double>::infinity()
                       : energy - previous_energy;

    linalg::Matrix diagonalization_fock = raw_fock;
    if(options.diis) {
      diis.add(raw_fock, orthonormal_error);
      if(iteration >= options.diis_start) {
        const std::optional<linalg::Matrix> extrapolated =
            diis.extrapolate();
        if(extrapolated) {
          diagonalization_fock = *extrapolated;
        }
      }
    }
    Orbitals next_orbitals =
        diagonalize_fock(diagonalization_fock,
                         orthogonalization.transformation,
                         &previous_orbitals.orthonormal_coefficients,
                         occupied_orbitals, options.level_shift);
    linalg::Matrix next_density =
        density_from_coefficients(next_orbitals.ao_coefficients,
                                  occupied_orbitals);
    if(options.damping > 0.0) {
      next_density =
          linalg::add(linalg::scaled(next_density, 1.0 - options.damping),
                      linalg::scaled(density, options.damping));
      linalg::symmetrize_in_place(next_density);
    }
    const double density_rms =
        rms(linalg::subtract(next_density, density));
    const RhfIteration record{
        .iteration = iteration,
        .electronic_energy = energy,
        .total_energy = energy + nuclear_repulsion,
        .energy_change = energy_change,
        .density_rms = density_rms,
        .commutator_rms = gradient_rms,
    };
    history.push_back(record);
    if(observer) {
      observer(record);
    }
    if(density_observer) {
      density_observer(record, next_density);
    }

    const bool converged =
        iteration > 1 &&
        std::abs(energy_change) <= options.energy_tolerance &&
        density_rms <= options.density_rms_tolerance &&
        gradient_rms <= options.commutator_rms_tolerance;
    if(converged) {
      // Store orbitals of the unshifted, non-DIIS Fock that corresponds to the
      // reported density and energy.
      Orbitals final_orbitals =
          diagonalize_fock(raw_fock, orthogonalization.transformation,
                           nullptr, occupied_orbitals, 0.0);
      std::vector<double> occupations(final_orbitals.energies.size(), 0.0);
      std::fill_n(occupations.begin(), occupied_orbitals, 2.0);
      return RhfResult{
          .converged = true,
          .iteration_count = iteration,
          .nuclear_repulsion_energy = nuclear_repulsion,
          .electronic_energy = energy,
          .total_energy = energy + nuclear_repulsion,
          .final_energy_change = energy_change,
          .final_density_rms = density_rms,
          .final_commutator_rms = gradient_rms,
          .overlap = one_electron.overlap,
          .core_hamiltonian = one_electron.core_hamiltonian,
          .orthogonalizer = orthogonalization.transformation,
          .density = std::move(density),
          .fock = std::move(raw_fock),
          .coefficients = std::move(final_orbitals.ao_coefficients),
          .orbital_energies = std::move(final_orbitals.energies),
          .occupations = std::move(occupations),
          .removed_overlap_vectors =
              orthogonalization.removed_vectors,
          .unique_shell_quartets_per_fock = unique_quartets_per_fock,
          .evaluated_shell_quartets_total =
              evaluated_quartets_total,
          .screened_shell_quartets_total =
              screened_quartets_total,
          .schwarz_screened_shell_quartets_total =
              schwarz_screened_quartets_total,
          .density_screened_shell_quartets_total =
              density_screened_quartets_total,
          .eri_schwarz_threshold = options.eri_schwarz_threshold,
          .density_screen = options.density_screen,
          .fock_contribution_threshold =
              options.fock_contribution_threshold,
          .final_screened_fock_maximum_error_bound =
              final_screened_fock_error_bound,
          .mpi_ranks = options.mpi_ranks,
          .fock_threads = options.fock_threads,
          .minimum_owned_shell_quartets_per_fock =
              minimum_owned_quartets_per_fock,
          .maximum_owned_shell_quartets_per_fock =
              maximum_owned_quartets_per_fock,
          .direct_fock_wall_seconds_total =
              direct_fock_wall_seconds_total,
          .minimum_direct_fock_wall_seconds_last =
              minimum_direct_fock_wall_seconds_last,
          .maximum_direct_fock_wall_seconds_last =
              maximum_direct_fock_wall_seconds_last,
          .history = std::move(history),
      };
    }

    previous_energy = energy;
    density = std::move(next_density);
    previous_orbitals = std::move(next_orbitals);
  }

  const RhfIteration& last = history.back();
  throw std::runtime_error(
      "RHF did not converge in " +
      std::to_string(options.maximum_iterations) +
      " iterations; final |dE|=" + std::to_string(std::abs(last.energy_change)) +
      ", density RMS=" + std::to_string(last.density_rms) +
      ", commutator RMS=" + std::to_string(last.commutator_rms));
}

}  // namespace modernqc::scf
