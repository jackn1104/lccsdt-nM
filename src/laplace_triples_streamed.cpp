#include "localized_cc/laplace_triples.hpp"

#include "localized_cc/parallel.hpp"
#include "localized_cc/pm_symmetry.hpp"
#include "localized_cc/selected_ccsd_storage.hpp"
#include "laplace_triples_quadrature.hpp"
#include "triples_block_screen.hpp"
#include "triples_monomer_patterns.hpp"
#include "triples_row_file.hpp"
#include "triples_occupied_panel.hpp"
#include "triples_virtual_axes.hpp"
#include "triples_sector_checkpoint.hpp"

#include "modernqc/linalg/eigensolver.hpp"
#include "modernqc/linalg/matrix.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <iomanip>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <type_traits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace localized_cc {
namespace {

using Clock = std::chrono::steady_clock;
using ModernMatrix = modernqc::linalg::Matrix;

extern "C" {
void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb, const double* beta, double* c,
            const int* ldc);
}

constexpr std::size_t sector_count = 4;
constexpr std::size_t missing_index =
    std::numeric_limits<std::size_t>::max();
constexpr std::array<std::array<std::size_t, 3>, 6> permutations{{
    {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}},
    {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}}}};
constexpr std::array<double, 6> permutation_sign{{1.0, -1.0, -1.0,
                                                  1.0, 1.0, -1.0}};

std::size_t checked_product(const std::size_t left,
                            const std::size_t right,
                            const char* label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(std::string(label) + " size overflows");
  return left * right;
}

std::size_t checked_sum(const std::size_t left, const std::size_t right,
                        const char* label) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error(std::string(label) + " size overflows");
  return left + right;
}

int blas_dimension(const std::size_t value, const char* label) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error(std::string(label) +
                              " exceeds the BLAS integer range");
  return static_cast<int>(value);
}

double seconds_since(const Clock::time_point begin) {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

std::size_t choose2(const std::size_t value) {
  return value < 2 ? 0 : checked_product(value, value - 1, "choose two") / 2;
}

std::size_t choose3(const std::size_t value) {
  return value < 3
      ? 0
      : checked_product(checked_product(value, value - 1, "choose three"),
                        value - 2, "choose three") /
            6;
}

std::size_t combination_rank(const std::size_t first,
                             const std::size_t second,
                             const std::size_t third,
                             const std::size_t extent) {
  if (!(first < second && second < third && third < extent))
    throw std::invalid_argument("invalid unique orbital triple");
  const auto before_first = choose3(extent) - choose3(extent - first);
  const auto second_count = second - first - 1;
  const auto before_second = second_count == 0
      ? 0
      : checked_product(second_count,
                        (extent - first - 2) + (extent - second),
                        "triple rank") /
            2;
  return before_first + before_second + third - second - 1;
}

struct TripleDescriptor {
  std::array<std::size_t, 3> spin{};
  // Canonical sector order: all alpha spatial indices followed by beta.
  std::array<std::size_t, 3> spatial{};
  double canonical_sign{1.0};
  std::size_t global_index{};
};

struct TripleSectors {
  std::array<std::vector<TripleDescriptor>, sector_count> sectors;
  std::vector<std::size_t> global_to_sector;
  std::vector<std::size_t> global_to_column;
};

TripleSectors make_triple_sectors(const std::size_t spin_extent) {
  if (spin_extent < 3 || spin_extent % 2 != 0)
    throw std::invalid_argument(
        "triples spin extent must be even and at least three");
  TripleSectors result;
  const auto triples = choose3(spin_extent);
  result.global_to_sector.assign(triples, missing_index);
  result.global_to_column.assign(triples, missing_index);
  const auto spatial_extent = spin_extent / 2;
  result.sectors[0].reserve(choose3(spatial_extent));
  result.sectors[3].reserve(choose3(spatial_extent));
  const auto mixed = checked_product(spatial_extent, choose2(spatial_extent), "mixed spin triples");
  result.sectors[1].reserve(mixed);
  result.sectors[2].reserve(mixed);
  std::size_t global = 0;
  for (std::size_t first = 0; first < spin_extent; ++first)
    for (std::size_t second = first + 1; second < spin_extent; ++second)
      for (std::size_t third = second + 1; third < spin_extent; ++third) {
        const std::array<std::size_t, 3> spin{first, second, third};
        std::array<std::size_t, 3> alpha{}, beta{};
        std::size_t alpha_count = 0, beta_count = 0;
        for (const auto orbital : spin) {
          if (orbital % 2 == 0) alpha[alpha_count++] = orbital / 2;
          else beta[beta_count++] = orbital / 2;
        }
        const auto sector = alpha_count;
        std::array<std::size_t, 3> spatial{};
        std::array<std::size_t, 3> canonical_spin{};
        std::size_t axis = 0;
        for (std::size_t index = 0; index < alpha_count; ++index) {
          const auto orbital = alpha[index];
          spatial[axis] = orbital;
          canonical_spin[axis++] = 2 * orbital;
        }
        for (std::size_t index = 0; index < beta_count; ++index) {
          const auto orbital = beta[index];
          spatial[axis] = orbital;
          canonical_spin[axis++] = 2 * orbital + 1;
        }
        std::size_t inversions = 0;
        for (std::size_t left = 0; left < 3; ++left)
          for (std::size_t right = left + 1; right < 3; ++right)
            inversions += canonical_spin[left] > canonical_spin[right];
        const auto column = result.sectors[sector].size();
        result.sectors[sector].push_back(
            TripleDescriptor{spin, spatial,
                             inversions % 2 == 0 ? 1.0 : -1.0, global});
        result.global_to_sector[global] = sector;
        result.global_to_column[global] = column;
        ++global;
      }
  if (global != triples)
    throw std::logic_error("unique triples enumeration is incomplete");
  return result;
}

struct SectorMoments {
  std::vector<std::size_t> owned_columns;
  std::vector<double> doubles;
  std::vector<double> left;
  std::vector<double> virtual_transformed;
  std::vector<double> response;
  std::unique_ptr<detail::TriplesRowFile> disk_doubles, disk_left;
  // Cache classes by distinct label sets, not by every six-index entry.
  std::vector<unsigned char> pattern_orders;
  std::vector<std::size_t> occupied_patterns;
  std::vector<std::size_t> virtual_patterns;
  std::size_t virtual_pattern_count{};
};

ModernMatrix square_block(const Matrix& source, const std::size_t first,
                          const std::size_t count) {
  const auto dimensions = source.dimensions();
  if (count == 0 || dimensions[0] != dimensions[1] ||
      first > dimensions[0] || count > dimensions[0] - first)
    throw std::invalid_argument("invalid streamed triples Fock block");
  ModernMatrix result(count, count);
  for (std::size_t row = 0; row < count; ++row)
    for (std::size_t column = 0; column < count; ++column)
      result(row, column) = source(first + row, first + column);
  return result;
}

ModernMatrix exponential_propagator(
    const modernqc::linalg::SymmetricEigendecomposition& eigensystem,
    const double exponent_factor) {
  const auto dimension = eigensystem.eigenvectors.rows();
  if (eigensystem.eigenvectors.columns() != dimension ||
      eigensystem.eigenvalues.size() != dimension)
    throw std::invalid_argument("invalid propagator eigensystem");
  ModernMatrix scaled(dimension, dimension);
  ModernMatrix result(dimension, dimension);
  for (std::size_t eigen = 0; eigen < dimension; ++eigen) {
    const auto factor =
        std::exp(exponent_factor * eigensystem.eigenvalues[eigen]);
    for (std::size_t row = 0; row < dimension; ++row)
      scaled(row, eigen) = eigensystem.eigenvectors(row, eigen) * factor;
  }
  const int n = blas_dimension(dimension, "propagator dimension");
  constexpr char no_transpose = 'N';
  constexpr char transpose = 'T';
  constexpr double alpha = 1.0;
  constexpr double beta = 0.0;
  dgemm_(&no_transpose, &transpose, &n, &n, &n, &alpha, scaled.data(), &n,
         eigensystem.eigenvectors.data(), &n, &beta, result.data(), &n);
  return result;
}

std::size_t tensor_index(const std::size_t first,
                         const std::size_t second,
                         const std::size_t third,
                         const std::size_t extent) {
  return first + extent * (second + extent * third);
}

void expand_sector_row(const double* input,
                       const std::vector<TripleDescriptor>& triples,
                       const std::size_t sector,
                       const std::size_t spatial_extent,
                       std::vector<double>& tensor) {
  std::fill(tensor.begin(), tensor.end(), 0.0);
  for (std::size_t column = 0; column < triples.size(); ++column) {
    const auto& triple = triples[column];
    const auto value = triple.canonical_sign * input[column];
    const auto& index = triple.spatial;
    if (sector == 0 || sector == 3) {
      for (std::size_t permutation = 0; permutation < permutations.size();
           ++permutation)
        tensor[tensor_index(index[permutations[permutation][0]],
                            index[permutations[permutation][1]],
                            index[permutations[permutation][2]],
                            spatial_extent)] =
            permutation_sign[permutation] * value;
    } else if (sector == 1) {
      tensor[tensor_index(index[0], index[1], index[2], spatial_extent)] =
          value;
      tensor[tensor_index(index[0], index[2], index[1], spatial_extent)] =
          -value;
    } else {
      tensor[tensor_index(index[0], index[1], index[2], spatial_extent)] =
          value;
      tensor[tensor_index(index[1], index[0], index[2], spatial_extent)] =
          -value;
    }
  }
}

void transform_tensor_axes(const std::vector<double>& source,
                           const ModernMatrix& propagator,
                           std::vector<double>& first,
                           std::vector<double>& second) {
  const auto extent = propagator.rows();
  const auto square = checked_product(extent, extent, "tensor square");
  if (propagator.columns() != extent ||
      source.size() != checked_product(square, extent, "tensor cube") ||
      first.size() != source.size() || second.size() != source.size())
    throw std::invalid_argument("streamed triples tensor dimensions mismatch");
  detail::transform_virtual_axes(source,
      std::span<const double>(propagator.data(),square),extent,first,second);
}

void compress_sector_row(const std::vector<double>& tensor,
                         const std::vector<TripleDescriptor>& triples,
                         const std::size_t spatial_extent, double* output) {
  for (std::size_t column = 0; column < triples.size(); ++column) {
    const auto& triple = triples[column];
    output[column] = triple.canonical_sign *
                     tensor[tensor_index(triple.spatial[0], triple.spatial[1],
                                         triple.spatial[2], spatial_extent)];
  }
}

void transform_virtual_rows(const std::vector<double>& input,
                            const std::size_t rows,
                            const std::vector<TripleDescriptor>& triples,
                            const std::size_t sector,
                            const ModernMatrix& propagator,
                            std::vector<double>& output,
                            const bool skip_zero_rows = false,
                            std::span<const unsigned char> active_rows = {}) {
  const auto columns = triples.size();
  if (input.size() != checked_product(rows, columns, "moment rows") ||
      output.size() != input.size() || (!active_rows.empty() && active_rows.size()!=rows))
    throw std::invalid_argument("streamed triples moment row mismatch");
  if (rows == 0) return;
  const auto extent = propagator.rows();
  if(sector>=sector_count || propagator.columns()!=extent)
    throw std::invalid_argument("virtual transform sector/propagator mismatch");
  for(const auto& triple:triples)
    for(const auto index:triple.spatial)
      if(index>=extent) throw std::out_of_range("virtual triple spatial index exceeds propagator");
  const auto cube = checked_product(
      checked_product(extent, extent, "propagator cube"), extent,
      "propagator cube");
#if defined(_OPENMP)
  const int active_threads = static_cast<int>(std::min(rows,
      static_cast<std::size_t>(omp_get_max_threads())));
#pragma omp parallel num_threads(active_threads)
#endif
  {
    std::vector<double> expanded(cube, 0.0);
    std::vector<double> first(cube, 0.0);
    std::vector<double> second(cube, 0.0);
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for (std::ptrdiff_t row = 0;
         row < static_cast<std::ptrdiff_t>(rows); ++row) {
      const auto offset = static_cast<std::size_t>(row) * columns;
      if (!active_rows.empty() && !active_rows[static_cast<std::size_t>(row)]) {
        std::fill_n(output.data()+offset,columns,0.0);
        continue;
      }
      if (skip_zero_rows && std::all_of(input.begin()+offset,
          input.begin()+offset+columns, [](double x) { return x==0.0; })) {
        std::fill(output.begin()+offset,output.begin()+offset+columns,0.0);
        continue;
      }
      expand_sector_row(input.data() + offset, triples, sector, extent,
                        expanded);
      transform_tensor_axes(expanded, propagator, first, second);
      compress_sector_row(first, triples, extent, output.data() + offset);
    }
  }
}

double spin_propagator_element(const ModernMatrix& spatial,
                               const std::size_t target,
                               const std::size_t source) {
  return target % 2 == source % 2
      ? spatial(target / 2, source / 2)
      : 0.0;
}

double compound_element(const ModernMatrix& spatial,
                        const TripleDescriptor& target,
                        const TripleDescriptor& source) {
  double matrix[3][3]{};
  for (std::size_t row = 0; row < 3; ++row)
    for (std::size_t column = 0; column < 3; ++column)
      matrix[row][column] = spin_propagator_element(
          spatial, target.spin[row], source.spin[column]);
  return matrix[0][0] *
             (matrix[1][1] * matrix[2][2] -
              matrix[1][2] * matrix[2][1]) -
         matrix[0][1] *
             (matrix[1][0] * matrix[2][2] -
              matrix[1][2] * matrix[2][0]) +
         matrix[0][2] *
             (matrix[1][0] * matrix[2][1] -
              matrix[1][1] * matrix[2][0]);
}

double maximum_block_offdiagonal(const Matrix& fock,
                                 const std::size_t occupied) {
  const auto orbitals = fock.dimensions()[0];
  double maximum = 0.0;
  for (std::size_t row = 0; row < orbitals; ++row)
    for (std::size_t column = 0; column < orbitals; ++column)
      if (row != column && ((row < occupied && column < occupied) ||
                            (row >= occupied && column >= occupied)))
        maximum = std::max(maximum, std::abs(fock(row, column)));
  return maximum;
}

} // namespace

template <typename T2>
StreamedLaplaceTriplesResult solve_localized_triples_laplace_streamed_impl(
    const SpatialRhfHamiltonian& hamiltonian, const Matrix& t1,
    const T2& t2, const std::vector<int>& occupied_monomers,
    const std::vector<int>& virtual_monomers,
    const StreamedLaplaceTriplesOptions& options) {
  detail::validate_laplace_triples_quadrature(
      options.quadrature_scheme, options.quadrature_points);
  if (options.symmetry_propagation && !options.pm_symmetry)
    throw std::invalid_argument("symmetry propagation requires a verified PM group");
  if (options.shared_pair_packing &&
      options.moment_backend != CanonicalTriplesBackend::tiled_cpu)
    throw std::invalid_argument("shared triples pair packing requires tiled CPU moments");
  if (options.spin_aware_construction &&
      options.moment_backend != CanonicalTriplesBackend::tiled_cpu)
    throw std::invalid_argument("spin-aware triples construction requires tiled CPU moments");
  if (!(options.requested_energy_error_hartree > 0.0) ||
      !std::isfinite(options.requested_energy_error_hartree) ||
      !(options.denominator_floor > 0.0) ||
      !std::isfinite(options.denominator_floor) ||
      options.validation_grid_size < 2 || options.virtual_tile_size == 0 ||
      options.propagation_row_batch == 0 ||
      options.moment_backend == CanonicalTriplesBackend::reference_cpu)
    throw std::invalid_argument("invalid streamed Laplace triples options");
  if (!(options.block_screening_budget_hartree >= 0.0) ||
      !std::isfinite(options.block_screening_budget_hartree))
    throw std::invalid_argument("invalid triples block screening budget");
  const bool screen_blocks=options.block_screening_budget_hartree>0.0;
  if(options.two_spin_sectors && (!options.sector_streaming || screen_blocks || options.pm_symmetry ||
      !options.initial_point_many_body_energies.empty() || !options.completed_points.empty()))
    throw std::invalid_argument("two spin sectors require unscreened sector streaming without PM symmetry or legacy point restart");
  if(!options.sector_checkpoint_path.empty() && !options.sector_streaming)
    throw std::invalid_argument("partial triples checkpoints require sector streaming");
  const bool disk_moments=!options.scratch_directory.empty();
  const bool durable_moments=!options.moment_checkpoint_directory.empty();
  if(durable_moments && (!disk_moments || !options.sector_streaming ||
      options.sector_checkpoint_identity.empty() || options.pm_symmetry || screen_blocks))
    throw std::invalid_argument("moment checkpoints require identified unscreened disk sector streaming without task symmetry");
  if (options.disk_parallel_panels && !disk_moments)
    throw std::invalid_argument("parallel disk panels require disk-backed triples");
  if(disk_moments && (!options.sector_streaming || options.pm_symmetry ||
      options.symmetry_propagation || screen_blocks || !options.maximum_scratch_bytes ||
      options.moment_backend != CanonicalTriplesBackend::tiled_cpu))
    throw std::invalid_argument("disk triples requires tiled CPU moments, sector streaming, explicit disk budget, no screening and no PM task symmetry");
  if (screen_blocks && (options.pm_symmetry ||
      !options.initial_point_many_body_energies.empty() ||
      !options.completed_points.empty() || options.checkpoint_observer))
    throw std::invalid_argument(
        "screened triples requires fresh points, no point checkpoint observer, "
        "and no moment-row symmetry reconstruction; CCSD checkpoints remain reusable");
  const auto spatial_occupied = hamiltonian.nocc;
  const auto spatial_virtuals = hamiltonian.nvirt;
  if (spatial_occupied < 2 || spatial_virtuals < 2 ||
      occupied_monomers.size() != spatial_occupied ||
      virtual_monomers.size() != spatial_virtuals)
    throw std::invalid_argument(
        "streamed Laplace triples orbital-label dimensions mismatch");
  if (std::any_of(occupied_monomers.begin(), occupied_monomers.end(),
                  [](const int label) { return label < 0; }) ||
      std::any_of(virtual_monomers.begin(), virtual_monomers.end(),
                  [](const int label) { return label < 0; }))
    throw std::invalid_argument(
        "streamed Laplace triples requires confident monomer labels");
  if (spatial_occupied > std::numeric_limits<std::size_t>::max() / 2 ||
      spatial_virtuals > std::numeric_limits<std::size_t>::max() / 2)
    throw std::overflow_error("streamed triples spin extent overflows");
  const auto occupied = 2 * spatial_occupied;
  const auto virtuals = 2 * spatial_virtuals;
  if (t1.dimensions() !=
          std::array<std::size_t, 2>{occupied, virtuals} ||
      t2.dimensions() !=
          std::array<std::size_t, 4>{occupied, occupied, virtuals, virtuals})
    throw std::invalid_argument(
        "streamed Laplace triples amplitude dimensions mismatch");

  const auto world = parallel::world();
  const auto progress = [&](const std::string& message) {
    if (world.rank == 0 && options.progress_observer)
      options.progress_observer(message);
  };
  progress("triples startup: symmetry validation and ownership planning started");
  if(options.two_spin_sectors) {
    if constexpr (std::is_same_v<std::remove_cvref_t<T2>, Tensor4>) {
      progress("closed-shell exact spin-flip amplitude gate started");
      bool invalid=false;
      for(std::size_t i=world.rank;i<occupied;i+=world.size) {
        for(std::size_t a=0;a<virtuals;++a) {
          const auto value=t1(i,a);
          invalid=invalid || !std::isfinite(value) || value!=t1(i^1U,a^1U) ||
              ((i%2!=a%2) && value!=0.);
        }
        for(std::size_t j=0;j<occupied;++j)
          for(std::size_t a=0;a<virtuals;++a) for(std::size_t b=0;b<virtuals;++b) {
            const auto value=t2(i,j,a,b);
            invalid=invalid || !std::isfinite(value) || value!=t2(i^1U,j^1U,a^1U,b^1U) ||
                ((i%2+j%2!=a%2+b%2) && value!=0.);
          }
      }
      if(parallel::sum_group(static_cast<std::uint64_t>(invalid),0))
        throw std::runtime_error("two-sector gate failed: saved amplitudes are not exactly closed-shell spin-flip invariant; use four sectors, no amplitudes changed");
      progress("closed-shell exact spin-flip amplitude gate passed; sectors=0,1; multiplicity=2; full M1-M6");
    } else throw std::invalid_argument("two-sector gate requires full dense T2 at triples boundary");
  }
  if (options.pm_symmetry) {
    if constexpr (std::is_same_v<std::remove_cvref_t<T2>, Tensor4>) {
      validate_pm_symmetry(hamiltonian, *options.pm_symmetry);
      validate_pm_symmetry_labels(*options.pm_symmetry, occupied_monomers, virtual_monomers);
      validate_pm_symmetry_amplitudes(*options.pm_symmetry, t1, t2);
    } else {
      throw std::invalid_argument("PM triples symmetry requires full dense T2, not selected amplitudes");
    }
  }
  const auto descriptor_budget = checked_product(
      checked_sum(choose3(occupied), choose3(virtuals), "triple descriptors"),
      2 * sizeof(TripleDescriptor) + 4 * sizeof(std::size_t), "descriptor budget");
  if (parallel::sum_group(static_cast<std::uint64_t>(
          descriptor_budget > options.maximum_additional_memory_bytes), 0) != 0)
    throw std::runtime_error("streamed triples metadata exceeds memory guard before descriptor allocation");
  const auto occupied_sectors = make_triple_sectors(occupied);
  const auto virtual_sectors = make_triple_sectors(virtuals);
  std::array<SectorMoments, sector_count> moments;
  std::vector<std::size_t> occupied_global_to_local(
      occupied_sectors.global_to_sector.size(), missing_index);
  std::size_t local_moment_elements = 0;
  std::size_t retained_moment_elements = 0;
  std::size_t global_moment_elements = 0;
  std::size_t maximum_panel_elements = 0;
  std::size_t maximum_local_sector_elements = 0;
  std::size_t maximum_compound_elements = 0;
  for (std::size_t sector = 0; sector < sector_count; ++sector) {
    for (std::size_t column = 0;
         column < occupied_sectors.sectors[sector].size(); ++column) {
      const auto global =
          occupied_sectors.sectors[sector][column].global_index;
      if (global % world.size == world.rank) {
        occupied_global_to_local[global] =
            moments[sector].owned_columns.size();
        moments[sector].owned_columns.push_back(column);
      }
    }
    const auto virtual_count = virtual_sectors.sectors[sector].size();
    const auto local = checked_product(
        moments[sector].owned_columns.size(), virtual_count,
        "local triples moments");
    local_moment_elements = checked_sum(local_moment_elements, local,
                                        "local triples moments");
    if(!options.two_spin_sectors || sector<2)
      retained_moment_elements=checked_sum(retained_moment_elements,local,"retained moments");
    maximum_local_sector_elements = std::max(maximum_local_sector_elements, local);
    global_moment_elements = checked_sum(
        global_moment_elements,
        checked_product(occupied_sectors.sectors[sector].size(),
                        virtual_count, "global triples moments"),
        "global triples moments");
    for (std::size_t root = 0; root < world.size; ++root) {
      std::size_t source_rows = 0;
      for (const auto& descriptor : occupied_sectors.sectors[sector])
        source_rows += descriptor.global_index % world.size == root;
      maximum_panel_elements = std::max(
          maximum_panel_elements,
          checked_product(std::min(source_rows, options.propagation_row_batch), virtual_count,
                          "triples broadcast panel"));
      maximum_compound_elements = std::max(maximum_compound_elements,
          checked_product(moments[sector].owned_columns.size(),
              std::min(source_rows, options.propagation_row_batch), "compound panel"));
    }
  }
  const auto spatial_cube = checked_product(
      checked_product(spatial_virtuals, spatial_virtuals,
                      "virtual spatial cube"),
      spatial_virtuals, "virtual spatial cube");
  std::size_t thread_count = 1;
#if defined(_OPENMP)
  thread_count = static_cast<std::size_t>(omp_get_max_threads());
#endif
  const auto resident_moment_elements = options.sector_streaming
      ? maximum_local_sector_elements : local_moment_elements;
  const auto scratch_bytes=disk_moments ? checked_product(
      durable_moments ? retained_moment_elements : maximum_local_sector_elements,
      2*sizeof(double),"triples scratch storage") : 0;
  if(parallel::sum_group(static_cast<std::uint64_t>(scratch_bytes>options.maximum_scratch_bytes && disk_moments),0)!=0)
    throw std::runtime_error("triples scratch plan exceeds explicit per-rank disk budget");
  auto peak_bytes = checked_product(
      checked_product(2, resident_moment_elements, "local moment workspace"),
      sizeof(double), "local moment workspace");
  peak_bytes = checked_sum(
      peak_bytes, checked_product(checked_product(options.sector_streaming ? 1 : 2, maximum_local_sector_elements,
          "sector propagation workspace"), sizeof(double), "sector propagation bytes"),
      "streamed triples peak");
  if (options.sector_streaming)
    peak_bytes = checked_sum(peak_bytes, checked_product(thread_count,
        (8 * 4096 + 7) * sizeof(double), "bounded response tiles"), "response tile peak");
  if(disk_moments) {
    std::size_t largest_virtual_sector=0;
    for(const auto& sector:virtual_sectors.sectors) largest_virtual_sector=std::max(largest_virtual_sector,sector.size());
    // Two construction rows; input/output propagation panels; one full left
    // row and one 4096-element response tile per worker. Later generic panel,
    // cube and metadata guards deliberately remain conservative.
    const auto row_buffers=checked_sum(checked_sum(2,thread_count,"disk row buffers"),
        checked_product(options.disk_parallel_panels ? 3 : 2,
                        options.propagation_row_batch,"disk panels"),"disk row buffers");
    peak_bytes=checked_product(checked_product(row_buffers,largest_virtual_sector,"disk row workspace"),sizeof(double),"disk row bytes");
    if(durable_moments) {
      peak_bytes=checked_sum(peak_bytes,checked_product(4*sizeof(double),largest_virtual_sector,"checkpoint row buffers"),"checkpoint peak");
      peak_bytes=checked_sum(peak_bytes,checked_product(128,choose3(occupied),"checkpoint manifests"),"checkpoint peak");
    }
    const auto packed_columns=checked_product(options.propagation_row_batch,4096,"disk occupied packing");
    peak_bytes=checked_sum(peak_bytes,checked_product(thread_count,
        checked_product(checked_sum(packed_columns,4096+7,"disk packed energy tiles"),sizeof(double),"disk energy bytes"),
        "disk energy tiles"),"disk energy peak");
  }
  // The compact response index/multiplicity vectors coexist with the mask.
  // Use the global row count as a conservative bound for vector capacity.
  peak_bytes=checked_sum(peak_bytes,checked_product(choose3(occupied),
      4*sizeof(std::size_t)+1,"propagation orbit metadata"),"propagation metadata peak");
  if (options.pm_symmetry) {
    std::size_t largest=0;
    for (const auto& sector:virtual_sectors.sectors) largest=std::max(largest,sector.size());
    peak_bytes=checked_sum(peak_bytes,
        checked_product(checked_product(2,largest,"symmetry rows"),sizeof(double),"symmetry rows"),
        "symmetry reconstruction peak");
    peak_bytes=checked_sum(peak_bytes, checked_product(
        checked_product(options.pm_symmetry->operations.size(), largest, "symmetry maps"),
        sizeof(std::size_t) + sizeof(double) + 1, "symmetry map bytes"), "symmetry maps peak");
  }
  peak_bytes = checked_sum(
      peak_bytes,
      checked_product(world.size == 1 ? 0 : maximum_panel_elements, sizeof(double),
                      "broadcast panel bytes"),
      "streamed triples peak");
  peak_bytes = checked_sum(peak_bytes, checked_product(maximum_compound_elements,
      sizeof(double), "compound panel bytes"), "compound peak");
  // Conservative bounded tile producer + CPU contraction scratch. Include
  // all 36 permutation operands/results, not only the persistent moments.
  const auto width = std::min(options.virtual_tile_size, virtuals);
  const auto pair = checked_product(width, width, "moment tile pair");
  const auto cube = checked_product(pair, width, "moment tile cube");
  const auto orbital_sum = checked_sum(occupied, virtuals, "moment orbital count");
  const auto packed = checked_sum(
      checked_product(18, checked_product(width, checked_sum(orbital_sum, 2, "tile axes"), "tile axes"), "tile axes"),
      checked_product(36, checked_product(pair, checked_sum(orbital_sum, 1, "tile pairs"), "tile pairs"), "tile pairs"), "packed moment tile");
  peak_bytes = checked_sum(peak_bytes,
      checked_product(checked_sum(checked_sum(packed, checked_product(3, width, "tile energies"), "packed energies"), checked_product(110, cube, "tile contraction scratch"), "tile scratch"),
                      sizeof(double), "tile scratch bytes"), "triples peak with packing");
  peak_bytes = checked_sum(
      peak_bytes,
      checked_product(checked_product(3, disk_moments ? std::min(thread_count,options.propagation_row_batch) : thread_count,
                                      "thread tensor workspace"),
                      checked_product(spatial_cube, sizeof(double),
                                      "thread tensor workspace"),
                      "thread tensor workspace"),
      "streamed triples peak");
  // Collective rejection BEFORE allocating any six-index moment array.
  peak_bytes=checked_sum(peak_bytes,checked_product(
      checked_product(2,checked_product(spatial_virtuals,spatial_virtuals,"virtual slice square"),"virtual slice pair"),
      checked_product(thread_count,sizeof(double),"virtual slice worker bytes"),"virtual slice packing"),
      "virtual slice peak");
  progress("triples startup: workspace plan bytes/rank=" + std::to_string(peak_bytes) +
           "; limit=" + std::to_string(options.maximum_additional_memory_bytes));
  if (parallel::sum_group(static_cast<std::uint64_t>(
          peak_bytes > options.maximum_additional_memory_bytes), 0) != 0)
    throw std::runtime_error(
        "streamed Laplace triples exceeds the configured memory guard");

  StreamedLaplaceTriplesResult result;
  // Fused response/classification has no separate wall-time interval.
  if (options.sector_streaming)
    result.partition_accumulation_seconds = std::numeric_limits<double>::quiet_NaN();
  result.many_body_energies.assign(6, 0.0);
  result.many_body_terms.assign(6, 0U);
  result.occupied_triples = choose3(occupied);
  result.spin_allowed_moment_elements = global_moment_elements;
  result.maximum_local_moment_bytes =
      checked_product(2, checked_product(disk_moments ? 0 : resident_moment_elements,
                                         sizeof(double),
                                         "local moment bytes"),
                      "local moment bytes");
  result.maximum_broadcast_panel_bytes = checked_product(
      maximum_panel_elements, sizeof(double), "broadcast panel bytes");

  progress("triples startup: compressed nM metadata started");
  std::array<std::uint64_t, 6> local_many_body_terms{};
  for (std::size_t sector = 0; sector < sector_count; ++sector) {
    const detail::TripleMonomerPatterns op(occupied_sectors.sectors[sector], occupied_monomers);
    const detail::TripleMonomerPatterns vp(virtual_sectors.sectors[sector], virtual_monomers);
    auto& data = moments[sector];
    data.virtual_patterns = vp.columns;
    data.virtual_pattern_count = vp.patterns.size();
    const auto pattern_elements = checked_product(op.patterns.size(), vp.patterns.size(), "nM patterns");
    if (parallel::sum_group(static_cast<std::uint64_t>(pattern_elements >
            options.maximum_additional_memory_bytes - peak_bytes), 0) != 0)
      throw std::runtime_error("triples label-pattern table exceeds memory budget");
    data.pattern_orders.resize(pattern_elements);
    for (std::size_t o = 0; o < op.patterns.size(); ++o)
      for (std::size_t v = 0; v < vp.patterns.size(); ++v)
        data.pattern_orders[o * vp.patterns.size() + v] =
            detail::triple_pattern_span(op.patterns[o], vp.patterns[v]);
    for (std::size_t local = 0;
         local < moments[sector].owned_columns.size(); ++local) {
      const auto occupied_column = moments[sector].owned_columns[local];
      const auto pattern = op.columns[occupied_column];
      data.occupied_patterns.push_back(pattern);
      for (std::size_t column = 0; column < vp.patterns.size(); ++column) {
        const auto order = data.pattern_orders[pattern * vp.patterns.size() + column];
        if (order == 0 || order > local_many_body_terms.size())
          throw std::logic_error("invalid triples monomer span");
        local_many_body_terms[order - 1] += vp.counts[column];
      }
    }
  }
  for (std::size_t order = 0; order < local_many_body_terms.size(); ++order) {
    const auto global = parallel::sum_group(local_many_body_terms[order], 0);
    if (global > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("triples monomer-class count overflows");
    result.many_body_terms[order] = static_cast<std::size_t>(global);
  }

  // Coverage check.
  //
  // The per-class counts above are only meaningful if the owned columns tile
  // the full column set exactly.  A column claimed by no rank drops its
  // configurations out of every class at once, and no downstream comparison of
  // class sums can notice, because the reported total is *defined* as the sum
  // of the classes.  The count the sector structure demands is independent of
  // ownership, so comparing against it is the check that can actually fail:
  // it catches an unowned column, a doubly owned one, and any mismatch between
  // the classification pass and the ownership map.
  {
    std::uint64_t classified = 0;
    for (const auto count : local_many_body_terms) classified += count;
    const auto global_classified = parallel::sum_group(classified, 0);

    std::uint64_t required = 0;
    for (std::size_t sector = 0; sector < sector_count; ++sector)
      required += static_cast<std::uint64_t>(
                      occupied_sectors.sectors[sector].size()) *
                  static_cast<std::uint64_t>(
                      virtual_sectors.sectors[sector].size());

    if (global_classified != required)
      throw std::logic_error(
          "streamed Laplace triples classified " +
          std::to_string(global_classified) + " of " + std::to_string(required) +
          " configurations; the owned columns do not tile the sectors");
  }

  std::size_t metadata_bytes = 0;
  for (const auto& sector : moments) {
    metadata_bytes = checked_sum(metadata_bytes, sector.pattern_orders.size(), "nM metadata");
    metadata_bytes = checked_sum(metadata_bytes, checked_product(
        checked_sum(sector.occupied_patterns.size(), sector.virtual_patterns.size(), "pattern columns"),
        sizeof(std::size_t), "pattern column bytes"), "nM metadata");
  }
  // Sector descriptors, lookup indices and ownership maps also remain live.
  metadata_bytes = checked_sum(metadata_bytes, checked_product(
      checked_sum(choose3(occupied), choose3(virtuals), "triple descriptors"),
      2 * sizeof(TripleDescriptor) + 4 * sizeof(std::size_t), "descriptor bytes"), "triples metadata");
  peak_bytes = checked_sum(peak_bytes, metadata_bytes, "triples peak with metadata");
  if (parallel::sum_group(static_cast<std::uint64_t>(
          peak_bytes > options.maximum_additional_memory_bytes), 0) != 0)
    throw std::runtime_error("streamed Laplace triples exceeds memory guard including metadata and packing");
  result.planned_peak_additional_bytes = peak_bytes;
  result.planned_scratch_bytes=scratch_bytes;
  if(disk_moments) progress("triples disk-backed plan; scratch_bytes/rank="+std::to_string(scratch_bytes)+
      "; additional_RAM_bytes/rank="+std::to_string(peak_bytes)+"; full nM; temporary files are not checkpoints");
  progress("triples startup: compressed nM metadata complete; storage=" +
      std::string(options.sector_streaming ? "sector-streamed bounded response" : "all-sector moments") +
      "; planned_bytes/rank=" + std::to_string(peak_bytes));
  progress("triples startup: quadrature construction started");

  const auto spin_fock = expand_spatial_one_body_to_spin_orbital(
      hamiltonian.fock);
  // Build the actual quadrature before screening so the error bound includes
  // its complete matrix-exponential propagation, not diagonal denominators.
  const auto diagonalize_fock = [&](std::size_t first, std::size_t count,
                                    const char* label) {
    const auto block=square_block(hamiltonian.fock,first,count);
    std::ostringstream audit;
    audit<<std::scientific<<std::setprecision(12)
         <<"triples "<<label<<" Fock eigensolver; dimension="<<count
         <<"; maximum_asymmetry_hartree="<<modernqc::linalg::maximum_asymmetry(block)
         <<"; symmetry_tolerance_hartree=1e-12";
    progress(audit.str());
    try {
      return modernqc::linalg::diagonalize_symmetric(block,1e-12);
    } catch(const std::exception& e) {
      throw std::runtime_error(audit.str()+"; "+e.what());
    }
  };
  const auto occupied_eigen=diagonalize_fock(0,spatial_occupied,"occupied");
  const auto virtual_eigen=diagonalize_fock(spatial_occupied,spatial_virtuals,"virtual");
  const auto [occupied_minimum, occupied_maximum] = std::minmax_element(
      occupied_eigen.eigenvalues.begin(), occupied_eigen.eigenvalues.end());
  const auto [virtual_minimum, virtual_maximum] = std::minmax_element(
      virtual_eigen.eigenvalues.begin(), virtual_eigen.eigenvalues.end());
  auto denominator_minimum = 3.0 * (*virtual_minimum - *occupied_maximum);
  const auto denominator_maximum = 3.0 * (*virtual_maximum - *occupied_minimum);
  if (!(denominator_minimum > options.denominator_floor) ||
      !(denominator_maximum > denominator_minimum))
    throw std::runtime_error("streamed triples denominator interval is invalid");
  if (options.quadrature_scheme == LaplaceTriplesQuadratureScheme::minimax) {
    if (denominator_maximum / denominator_minimum < 2.0)
      denominator_minimum = denominator_maximum / 2.0;
    if (denominator_maximum / denominator_minimum > 100.0)
      throw std::runtime_error("streamed triples denominator ratio exceeds the validated minimax interval");
  }
  result.denominator_minimum = denominator_minimum;
  result.denominator_maximum = denominator_maximum;
  result.localized_fock_offdiagonal = maximum_block_offdiagonal(spin_fock, occupied);
  std::vector<double> fit_diagnostics(1, 0.0);
  if (world.rank == 0) {
    const auto quadrature = detail::build_laplace_triples_quadrature(
        denominator_minimum, denominator_maximum, options.quadrature_scheme,
        options.quadrature_points, options.validation_grid_size);
    result.nodes = quadrature.nodes;
    result.weights = quadrature.weights;
    fit_diagnostics[0] = quadrature.maximum_reciprocal_error;
  } else {
    result.nodes.resize(options.quadrature_points);
    result.weights.resize(options.quadrature_points);
  }
  parallel::broadcast_in_place(result.nodes);
  parallel::broadcast_in_place(result.weights);
  parallel::broadcast_in_place(fit_diagnostics);
  result.maximum_reciprocal_error = fit_diagnostics[0];
  result.quadrature_points = result.nodes.size();

  detail::TriplesBlockScreenPlan screen_plan;
  if (screen_blocks) {
    if (world.rank==0 && options.screening_observer)
      options.screening_observer("triples preconstruction block bounds started; full nM retained");
    const auto begin=Clock::now();
    long double knorm=0;
    for (std::size_t q=0;q<result.nodes.size();++q) {
      if (!(result.nodes[q]>=0) || !std::isfinite(result.nodes[q]) ||
          !std::isfinite(result.weights[q]))
        throw std::runtime_error("invalid screening quadrature node or weight");
      knorm+=std::abs(static_cast<long double>(result.weights[q]))*
             std::exp(-static_cast<long double>(denominator_minimum)*result.nodes[q]);
    }
    screen_plan=detail::plan_triples_block_screen(
        hamiltonian,t1,t2,options.virtual_tile_size,
        std::max(static_cast<double>(knorm),1.0/denominator_minimum)*(1.0+1.e-10),
        options.block_screening_budget_hartree,
        options.maximum_additional_memory_bytes-peak_bytes);
    result.block_screening_bound_hartree=screen_plan.energy_bound;
    result.block_screening_candidates=screen_plan.candidates;
    result.block_screening_skipped=screen_plan.skipped;
    result.block_screening_plan_bytes=screen_plan.workspace_bytes;
    result.block_screening_seconds=seconds_since(begin);
    if (world.rank==0 && options.screening_observer) {
      std::ostringstream message;
      message.precision(17);
      message << "triples preconstruction block bounds complete; candidates="
              << screen_plan.candidates << "; skipped=" << screen_plan.skipped
              << "; total_and_each_nM_bound_hartree=" << screen_plan.energy_bound
              << "; seconds=" << result.block_screening_seconds;
      options.screening_observer(message.str());
    }
  }
  const auto calculation_begin = Clock::now();
  std::vector<double> norms{0.0, 0.0};
  std::ostringstream moment_key;
  moment_key << std::setprecision(17) << options.sector_checkpoint_identity
      << ':' << world.size << ':' << world.rank << ':' << occupied << ':' << virtuals
      << ':' << options.virtual_tile_size << ':' << options.spin_aware_construction
      << ':' << options.shared_pair_packing << ':' << options.denominator_floor;
  for(auto m:occupied_monomers) moment_key << ':' << m;
  for(auto m:virtual_monomers) moment_key << ':' << m;
  moment_key << ":two-spin=" << options.two_spin_sectors;
  const auto build_moments = [&](const std::size_t sector_first, const std::size_t sector_end) {
  const auto moment_begin = Clock::now();
  progress(std::string(durable_moments ? "opening durable moment checkpoints; sectors=" :
      disk_moments ? "reserving two temporary moment files; sectors=" : "allocating two persistent moment arrays; sectors=") + std::to_string(sector_first) +
      ".." + std::to_string(sector_end - 1));
  std::string moment_error;
  try {
  for (std::size_t sector = sector_first; sector < sector_end; ++sector) {
    const auto elements = checked_product(moments[sector].owned_columns.size(),
        virtual_sectors.sectors[sector].size(), "moment allocation");
    if(disk_moments) {
      const auto rows=moments[sector].owned_columns.size(),columns=virtual_sectors.sectors[sector].size();
      const auto directory=durable_moments ? options.moment_checkpoint_directory : options.scratch_directory;
      const auto stem="rank"+std::to_string(world.rank)+"-sector"+std::to_string(sector);
      moments[sector].disk_doubles=std::make_unique<detail::TriplesRowFile>(directory,rows,columns,
          durable_moments ? stem+"-D.bin" : "",moment_key.str()+":"+std::to_string(sector)+":D");
      moments[sector].disk_left=std::make_unique<detail::TriplesRowFile>(directory,rows,columns,
          durable_moments ? stem+"-L.bin" : "",moment_key.str()+":"+std::to_string(sector)+":L");
      if(durable_moments) progress("moment checkpoint verified rank-zero rows="+
          std::to_string(std::min(moments[sector].disk_doubles->completed_rows(),
                                 moments[sector].disk_left->completed_rows()))+"/"+std::to_string(rows)+
          "; sector="+std::to_string(sector));
    } else {
      moments[sector].doubles.assign(elements, 0.0);
      moments[sector].left.assign(elements, 0.0);
    }
  }
  } catch(const std::exception& e) { moment_error=e.what(); }
  if(parallel::sum_group(static_cast<std::uint64_t>(!moment_error.empty()),0)!=0)
    throw std::runtime_error("moment storage initialization failed on a rank: "+moment_error);
  progress("triples moment packing and contraction started");
  LocalizedTriplesMomentStreamOptions stream_options;
  stream_options.backend = options.moment_backend;
  stream_options.virtual_tile_size = options.virtual_tile_size;
  stream_options.denominator_floor = options.denominator_floor;
  stream_options.cuda_device = options.cuda_device;
  stream_options.shared_pair_packing = options.shared_pair_packing;
  stream_options.spin_aware_construction = options.spin_aware_construction;
  if(durable_moments) stream_options.completed_row_observer = [&](const std::array<std::size_t,3>& triple) {
    const auto global=combination_rank(triple[0],triple[1],triple[2],occupied);
    const auto sector=occupied_sectors.global_to_sector[global];
    const auto row=occupied_global_to_local[global];
    moments[sector].disk_doubles->commit_row(row);
    moments[sector].disk_left->commit_row(row);
    if(row==0 || (row+1)%16==0 || row+1==moments[sector].owned_columns.size())
      progress("moment checkpoint durable rank-zero rows="+std::to_string(row+1)+
          "; sector="+std::to_string(sector));
  };
  stream_options.timing_observer = [&](double packing, double contraction) {
    result.moment_packing_seconds += packing;
    result.moment_contraction_seconds += contraction;
  };
  stream_options.row_observer = [&](std::size_t done, std::size_t total) {
    progress("triples moments rank-zero selected rows=" + std::to_string(done) + "/" +
        std::to_string(total) + "; seconds=" + std::to_string(seconds_since(moment_begin)));
  };
  if (screen_blocks) {
    stream_options.retain_block = [&](const std::array<std::size_t,3>& occ,
                                      const std::array<std::size_t,3>& offsets) {
      const auto row=combination_rank(occ[0],occ[1],occ[2],occupied);
      const auto width=std::min(options.virtual_tile_size,virtuals);
      const auto nt=screen_plan.tiles;
      const auto slot=((row*nt+offsets[0]/width)*nt+offsets[1]/width)*nt+offsets[2]/width;
      return screen_plan.keep.at(slot)!=0.0;
    };
  }
  stream_options.occupied_representative = [&](const std::array<std::size_t,3>& triple) {
      const auto global = combination_rank(triple[0], triple[1], triple[2], occupied);
      const auto sector = occupied_sectors.global_to_sector[global];
      if (sector < sector_first || sector >= sector_end) return false;
      if(durable_moments && occupied_global_to_local[global] <
          std::min(moments[sector].disk_doubles->completed_rows(),
                   moments[sector].disk_left->completed_rows())) return false;
      if (options.pm_symmetry) {
        for (const auto& g:options.pm_symmetry->operations)
          if (pm_map_triple(g,triple,0).first < triple) return false;
      }
      return true;
  };
  stream_options.observer = [&](LocalizedTriplesMomentBlock&& block) {
    const auto tile_elements = checked_product(checked_product(
        block.virtual_dimensions[0], block.virtual_dimensions[1], "moment block"),
        block.virtual_dimensions[2], "moment block");
    if (block.doubles_moment.size() != tile_elements ||
        block.left_moment.size() != tile_elements ||
        (block.virtual_stride != 1 && block.virtual_stride != 2))
      throw std::logic_error("invalid streamed moment block storage or stride");
    for (std::size_t axis = 0; axis < 3; ++axis)
      if (block.virtual_dimensions[axis] == 0 || block.virtual_offsets[axis] >= virtuals ||
          block.virtual_dimensions[axis] - 1 >
              (virtuals - 1 - block.virtual_offsets[axis]) / block.virtual_stride)
        throw std::logic_error("streamed moment block virtual range exceeds orbital space");
    const auto occupied_global = combination_rank(
        block.occupied_indices[0], block.occupied_indices[1],
        block.occupied_indices[2], occupied);
    const auto sector = occupied_sectors.global_to_sector[occupied_global];
    const auto local_row = occupied_global_to_local[occupied_global];
    if (sector < sector_first || sector >= sector_end ||
        local_row == missing_index ||
        local_row >= moments[sector].owned_columns.size())
      throw std::logic_error(
          "streamed triples moment arrived on the wrong MPI rank");
    const auto virtual_count = virtual_sectors.sectors[sector].size();
    for (std::size_t x0 = 0; x0 < block.virtual_dimensions[0]; ++x0)
      for (std::size_t x1 = 0; x1 < block.virtual_dimensions[1]; ++x1)
        for (std::size_t x2 = 0; x2 < block.virtual_dimensions[2]; ++x2) {
          const std::array<std::size_t, 3> global{
              block.virtual_offsets[0] + block.virtual_stride * x0,
              block.virtual_offsets[1] + block.virtual_stride * x1,
              block.virtual_offsets[2] + block.virtual_stride * x2};
          if (!(global[0] < global[1] && global[1] < global[2])) continue;
          const auto virtual_global = combination_rank(
              global[0], global[1], global[2], virtuals);
          if (virtual_sectors.global_to_sector[virtual_global] != sector)
            continue;
          const auto column =
              virtual_sectors.global_to_column[virtual_global];
          const auto tile_index =
              (x0 * block.virtual_dimensions[1] + x1) *
                  block.virtual_dimensions[2] +
              x2;
          const auto destination = local_row * virtual_count + column;
          if(disk_moments) {
            moments[sector].disk_doubles->set(local_row,column,block.doubles_moment[tile_index]);
            moments[sector].disk_left->set(local_row,column,block.left_moment[tile_index]);
          } else {
            moments[sector].doubles[destination] = block.doubles_moment[tile_index];
            moments[sector].left[destination] = block.left_moment[tile_index];
          }
        }
  };
  try { stream_localized_triples_moments(hamiltonian, t1, t2, stream_options); }
  catch(const std::exception& e) { moment_error=e.what(); }
  if(parallel::sum_group(static_cast<std::uint64_t>(!moment_error.empty()),0)!=0)
    throw std::runtime_error("moment construction/checkpoint failed on a rank: "+moment_error);
  // Rank-zero row progress alone is not a global phase boundary: a faster
  // rank used to enter propagation while rank zero was still constructing.
  // Keep the diagnostic boundary outside all OpenMP and contraction loops.
  if (disk_moments) {
    parallel::barrier();
    progress("triples moment construction complete on all ranks; disk sealing started");
  }
  if(disk_moments) for(std::size_t s=sector_first;s<sector_end;++s) {
    moments[s].disk_doubles->finish_writes(); moments[s].disk_left->finish_writes();
  }
  if (disk_moments) {
    parallel::barrier();
    progress("triples moment disk sealing complete on all ranks");
  }
  if (options.pm_symmetry) {
    progress("triples moment symmetry reconstruction started");
    // Compute a representative row once globally. Reconstruct its orbit on
    // the established cyclic owners. Broadcast two bounded rows at a time,
    // never the full six-index tensor. Stabilizers must not double count.
    for (std::size_t sector=sector_first; sector<sector_end; ++sector) {
      const auto nv=virtual_sectors.sectors[sector].size();
      std::vector<double> panel(checked_product(2,nv,"PM symmetry moment rows"));
      // Cache each virtual permutation once per sector, not once per
      // occupied row. Validate bijectivity before parallel scatter.
      std::vector<std::vector<std::size_t>> mapped_columns;
      std::vector<std::vector<double>> mapped_signs;
      for (const auto& g : options.pm_symmetry->operations) {
        std::vector<std::size_t> columns(nv);
        std::vector<double> signs(nv);
        std::vector<unsigned char> seen(nv, 0U);
        for (std::size_t column=0; column<nv; ++column) {
          const auto [target, sign] = pm_map_triple(g, virtual_sectors.sectors[sector][column].spin, spatial_occupied);
          const auto global = combination_rank(target[0], target[1], target[2], virtuals);
          const auto destination = virtual_sectors.global_to_column[global];
          if (virtual_sectors.global_to_sector[global] != sector || destination >= nv || seen[destination])
            throw std::logic_error("PM virtual triple map is not a sector bijection");
          seen[destination] = 1U;
          columns[column] = destination;
          signs[column] = sign;
        }
        mapped_columns.push_back(std::move(columns));
        mapped_signs.push_back(std::move(signs));
      }
      for (const auto& source:occupied_sectors.sectors[sector]) {
        if (!stream_options.occupied_representative(source.spin)) continue;
        ++result.symmetry_unique_moment_rows;
        const auto root=source.global_index%world.size;
        if (world.rank==root) {
          const auto local=occupied_global_to_local[source.global_index];
          std::copy_n(moments[sector].doubles.data()+local*nv,nv,panel.data());
          std::copy_n(moments[sector].left.data()+local*nv,nv,panel.data()+nv);
        }
        const auto communication_begin = Clock::now();
        parallel::broadcast_in_place(panel,root);
        result.communication_seconds += seconds_since(communication_begin);
        std::array<std::size_t,4> visited{};
        std::size_t count=0;
        for (std::size_t operation=0; operation<options.pm_symmetry->operations.size(); ++operation) {
          const auto& g = options.pm_symmetry->operations[operation];
          const auto mapped_triple=pm_map_triple(g,source.spin,0);
          const auto& target=mapped_triple.first;
          const int osign=mapped_triple.second;
          const auto global=combination_rank(target[0],target[1],target[2],occupied);
          const auto end=visited.begin()+static_cast<std::ptrdiff_t>(count);
          if (std::find(visited.begin(),end,global)!=end) continue;
          visited[count++]=global;
          if (global%world.size!=world.rank) continue;
          const auto local=occupied_global_to_local[global];
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
          for (std::size_t column=0;column<nv;++column) {
            const auto vsign = mapped_signs[operation][column];
            const auto destination=local*nv+mapped_columns[operation][column];
            moments[sector].doubles[destination]=osign*vsign*panel[column];
            moments[sector].left[destination]=osign*vsign*panel[nv+column];
          }
        }
      }
      progress("triples moment symmetry sector " + std::to_string(sector) + " complete");
    }
  }
  progress("triples sector moments complete; seconds=" + std::to_string(seconds_since(moment_begin)));
  progress("triples moment norms and quadrature bound started");
  for (std::size_t s = sector_first; s < sector_end; ++s) {
    const auto& sector = moments[s];
    double doubles_norm = 0.0, left_norm = 0.0;
    if(disk_moments) {
      std::vector<double> d(virtual_sectors.sectors[s].size()),l(d.size());
      for(std::size_t row=0;row<sector.owned_columns.size();++row) {
        sector.disk_doubles->read(row,d); sector.disk_left->read(row,l);
        for(std::size_t c=0;c<d.size();++c) { doubles_norm+=d[c]*d[c]; left_norm+=l[c]*l[c]; }
      }
    } else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+:doubles_norm,left_norm)
#endif
    for (std::size_t i = 0; i < sector.doubles.size(); ++i) {
      doubles_norm += sector.doubles[i] * sector.doubles[i];
      left_norm += sector.left[i] * sector.left[i];
    }
    }
    norms[0] += doubles_norm;
    norms[1] += left_norm;
  }
  result.moment_construction_seconds += seconds_since(moment_begin);
  if (disk_moments) {
    parallel::barrier();
    progress("triples moment norms complete on all ranks; propagation may begin");
  }
  };
  const auto certify_moments = [&]() {
  parallel::sum_in_place(norms);
  if(options.two_spin_sectors) for(auto& norm:norms) norm*=2.;

  result.certified_quadrature_error_bound =
      result.maximum_reciprocal_error *
      std::sqrt(std::max(0.0, norms[0])) *
      std::sqrt(std::max(0.0, norms[1]));
  if (!std::isfinite(norms[0]) || !std::isfinite(norms[1]) ||
      !std::isfinite(result.certified_quadrature_error_bound))
    throw std::runtime_error("non-finite streamed triples norm or quadrature bound");
  if (result.certified_quadrature_error_bound >
      options.requested_energy_error_hartree) {
    std::ostringstream message;
    message << "streamed Laplace triples "
            << laplace_triples_quadrature_scheme_name(
                   options.quadrature_scheme)
            << " q" << options.quadrature_points
            << " fails the requested energy bound: reciprocal_error="
            << result.maximum_reciprocal_error
            << " Eh^-1, certified_bound="
            << result.certified_quadrature_error_bound
            << " Eh, requested=" << options.requested_energy_error_hartree
            << " Eh, denominator_interval=[" << result.denominator_minimum
            << "," << result.denominator_maximum << "] Eh";
    throw std::runtime_error(message.str());
  }
  };

  const auto point_energy_count = checked_product(
      result.quadrature_points, std::size_t{6},
      "streamed triples point energies");
  const bool restarting = !options.initial_point_many_body_energies.empty() ||
                          !options.completed_points.empty();
  if (restarting &&
      (options.initial_point_many_body_energies.size() != point_energy_count ||
       options.completed_points.size() != result.quadrature_points))
    throw std::invalid_argument(
        "streamed Laplace triples restart dimensions mismatch");
  result.point_many_body_energies =
      restarting ? options.initial_point_many_body_energies
                 : std::vector<double>(point_energy_count, 0.0);
  result.completed_points =
      restarting ? options.completed_points
                 : std::vector<unsigned char>(result.quadrature_points, 0U);
  // Total accumulated without consulting the many-body classification, and the
  // number of points it covers.  Restored points contribute to the class sums
  // from the checkpoint but are not recomputed here, so the reconstruction
  // identity is only asserted over points this run actually evaluated.
  double unclassified_total = 0.0;
  std::size_t unclassified_points = 0;

  for (std::size_t point = 0; point < result.quadrature_points; ++point) {
    if (result.completed_points[point] > 1U)
      throw std::invalid_argument(
          "invalid streamed Laplace triples completion flag");
    if (result.completed_points[point] == 0U) {
      for (std::size_t order = 0; order < 6; ++order)
        result.point_many_body_energies[6 * point + order] = 0.0;
    } else {
      ++result.restarted_points;
      for (std::size_t order = 0; order < 6; ++order)
        if (!std::isfinite(
                result.point_many_body_energies[6 * point + order]))
          throw std::invalid_argument(
              "non-finite streamed Laplace triples restart energy");
    }
  }

  constexpr char no_transpose = 'N';
  constexpr double alpha = 1.0;
  constexpr double beta_one = 1.0;
  std::array<std::vector<std::vector<std::size_t>>, sector_count>
      source_columns_by_root;
  for (std::size_t sector = 0; sector < sector_count; ++sector) {
    source_columns_by_root[sector].resize(world.size);
    for (std::size_t column = 0;
         column < occupied_sectors.sectors[sector].size(); ++column) {
      const auto root =
          occupied_sectors.sectors[sector][column].global_index % world.size;
      source_columns_by_root[sector][root].push_back(column);
    }
  }
  // Reconstruct source rows after transforming only occupied-orbit representatives.
  // Signed maps include fermionic sorting parity. All source rows still enter
  // occupied propagation; no coupling or monomer class is truncated.
  const auto reconstruct_virtual_sources = [&](std::size_t sector) {
    if (!options.symmetry_propagation) return;
    const auto nv=virtual_sectors.sectors[sector].size();
    std::vector<std::vector<std::size_t>> maps;
    std::vector<std::vector<int>> phases;
    for (const auto& g:options.pm_symmetry->operations) {
      std::vector<std::size_t> map(nv);
      std::vector<int> phase(nv);
      for (std::size_t c=0;c<nv;++c) {
        const auto [target,sign]=pm_map_triple(g,virtual_sectors.sectors[sector][c].spin,spatial_occupied);
        const auto global=combination_rank(target[0],target[1],target[2],virtuals);
        map[c]=virtual_sectors.global_to_column[global];
        if (map[c]>=nv || virtual_sectors.sectors[sector][map[c]].global_index!=global)
          throw std::runtime_error("PM virtual propagation map leaves its spin sector");
        phase[c]=sign;
      }
      auto unique=map;
      std::sort(unique.begin(),unique.end());
      if (std::adjacent_find(unique.begin(),unique.end())!=unique.end())
        throw std::runtime_error("PM virtual propagation map is not bijective");
      maps.push_back(std::move(map)); phases.push_back(std::move(phase));
    }
    std::vector<double> panel(nv);
    for (const auto& source:occupied_sectors.sectors[sector]) {
      bool representative=true;
      for (const auto& g:options.pm_symmetry->operations)
        representative=representative && !(pm_map_triple(g,source.spin,0).first<source.spin);
      if (!representative) continue;
      const auto root=source.global_index%world.size;
      if (world.rank==root)
        std::copy_n(moments[sector].virtual_transformed.data()+occupied_global_to_local[source.global_index]*nv,nv,panel.data());
      const auto begin=Clock::now();
      parallel::broadcast_in_place(panel,root);
      result.communication_seconds+=seconds_since(begin);
      std::array<std::size_t,4> seen{};
      std::size_t count=0;
      for (std::size_t op=0;op<options.pm_symmetry->operations.size();++op) {
        const auto mapped_triple=pm_map_triple(options.pm_symmetry->operations[op],source.spin,0);
        const auto& target=mapped_triple.first;
        const int sign=mapped_triple.second;
        const auto global=combination_rank(target[0],target[1],target[2],occupied);
        if (std::find(seen.begin(),seen.begin()+count,global)!=seen.begin()+count) continue;
        seen[count++]=global;
        if (global%world.size!=world.rank) continue;
        const auto row=occupied_global_to_local[global];
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::size_t c=0;c<nv;++c)
          moments[sector].virtual_transformed[row*nv+maps[op][c]]=sign*phases[op][c]*panel[c];
      }
    }
  };
  const auto propagate_sector = [&](const std::size_t point, const std::size_t sector,
                                    std::vector<double>& point_energies) {
    progress("Laplace triples point " + std::to_string(point + 1) + "/" +
        std::to_string(result.nodes.size()) + "; sector=" + std::to_string(sector) + " started");
    const auto occupied_propagator = exponential_propagator(
        occupied_eigen, result.nodes[point]);
    const auto virtual_propagator = exponential_propagator(
        virtual_eigen, -result.nodes[point]);
    std::vector<std::size_t> target_rows;
    std::vector<std::size_t> orbit_sizes;
    std::vector<unsigned char> active_rows(moments[sector].owned_columns.size(),1);
    for (std::size_t row=0;row<active_rows.size();++row) {
      const auto& triple=occupied_sectors.sectors[sector][moments[sector].owned_columns[row]];
      std::array<std::size_t,4> seen{};
      std::size_t count=1;
      seen[0]=triple.global_index;
      if (options.symmetry_propagation) {
        for (const auto& g:options.pm_symmetry->operations) {
          const auto mapped=pm_map_triple(g,triple.spin,0).first;
          if (mapped<triple.spin) active_rows[row]=0;
          const auto global=combination_rank(mapped[0],mapped[1],mapped[2],occupied);
          if (std::find(seen.begin(),seen.begin()+count,global)==seen.begin()+count) seen[count++]=global;
        }
      }
      if (active_rows[row]) { target_rows.push_back(row); orbit_sizes.push_back(count); }
    }
    if (options.symmetry_propagation) {
      // The group must commute with each one-particle propagator, not merely
      // with the input Hamiltonian. Fail closed on numerical covariance loss.
      for (const auto& g:options.pm_symmetry->operations)
        for (bool virt:{false,true}) {
          const auto& p=virt?virtual_propagator:occupied_propagator;
          const auto offset=virt?spatial_occupied:0;
          for (std::size_t i=0;i<p.rows();++i) for (std::size_t j=0;j<p.rows();++j) {
            const auto x=p(i,j),y=g.sign[offset+i]*g.sign[offset+j]*p(g.image[offset+i]-offset,g.image[offset+j]-offset);
            if (!std::isfinite(x) || !std::isfinite(y) || std::abs(x-y)>1e-10*std::max({1.0,std::abs(x),std::abs(y)}))
              throw std::runtime_error("PM propagator covariance failed");
          }
        }
      progress("PM propagation representative rows rank0="+std::to_string(target_rows.size())+
          "/"+std::to_string(active_rows.size())+"; full source sum and nM multiplicities retained");
    }
      progress("Laplace triples sector " + std::to_string(sector) + " virtual propagation started");
      if(!disk_moments) {
      moments[sector].virtual_transformed.resize(moments[sector].doubles.size());
      transform_virtual_rows(
          moments[sector].doubles,
          moments[sector].owned_columns.size(),
          virtual_sectors.sectors[sector], sector, virtual_propagator,
          moments[sector].virtual_transformed, screen_blocks, active_rows);
      reconstruct_virtual_sources(sector);
      }

      auto& sector_moments = moments[sector];
      if (!options.sector_streaming)
        sector_moments.response.assign(checked_product(target_rows.size(),virtual_sectors.sectors[sector].size(),"representative response"), 0.0);
      progress("Laplace triples sector " + std::to_string(sector) + " occupied propagation started");
      const auto virtual_count = virtual_sectors.sectors[sector].size();
      const auto target_count = target_rows.size();
      // Check metadata once before entering worker hot loops or raw BLAS calls.
      if(sector_moments.virtual_patterns.size()!=virtual_count||
          sector_moments.occupied_patterns.size()!=sector_moments.owned_columns.size())
        throw std::logic_error("triples propagation label dimensions disagree");
      for(auto p:sector_moments.virtual_patterns)
        if(p>=sector_moments.virtual_pattern_count) throw std::logic_error("triples virtual pattern out of range");
      for(auto row:target_rows) {
        if(virtual_count==0) continue;
        const auto p=sector_moments.occupied_patterns.at(row);
        if(!sector_moments.virtual_pattern_count||
            p>=sector_moments.pattern_orders.size()/sector_moments.virtual_pattern_count)
          throw std::logic_error("triples occupied pattern out of range");
      }
      for(auto order:sector_moments.pattern_orders)
        if(order<1||order>6) throw std::logic_error("triples nM order out of range");
      std::vector<double> panel;
      std::vector<double> compound;
      // Each worker owns its energy bins across source panels. The response
      // is linear in each source panel, so L dot sum(response) is evaluated as
      // sum(L dot response_tile), with no full response allocation.
      std::vector<std::array<double, 7>> fused_energies(thread_count, std::array<double, 7>{});
      // Reuse large disk-read buffers across source panels. Allocating a
      // virtual-sector row for every worker at every panel is unnecessary.
      std::vector<std::vector<double>> disk_left_rows;
      if(disk_moments) {
        disk_left_rows.resize(thread_count);
        for(auto& row:disk_left_rows) row.resize(virtual_count);
      }
      // At most one task per source row. Existing orbit metadata guard covers
      // these two indices/row; it also covers target rows and orbit sizes.
      struct PanelTask { std::size_t root, first; };
      std::vector<PanelTask> panel_tasks;
      panel_tasks.reserve(occupied_sectors.sectors[sector].size());
      for (std::size_t root = 0; root < world.size; ++root)
        for (std::size_t first = 0; first < source_columns_by_root[sector][root].size();
             first += options.propagation_row_batch)
          panel_tasks.push_back({root, first});
      if (options.disk_parallel_panels)
        std::sort(panel_tasks.begin(), panel_tasks.end(), [](const auto& a, const auto& b) {
          return a.first != b.first ? a.first < b.first : a.root < b.root;
        });
      std::vector<double> ready_panel;
      auto prepared_first = std::numeric_limits<std::size_t>::max();
      std::size_t panel_number=0;
      for (const auto task : panel_tasks) {
        ++panel_number;
        const bool trace_panel=disk_moments&&(panel_number==1||panel_number%32==0||panel_number==panel_tasks.size());
        const auto panel_label="disk propagation point="+std::to_string(point+1)+" sector="+std::to_string(sector)+
            " panel="+std::to_string(panel_number)+"/"+std::to_string(panel_tasks.size());
        if(trace_panel) progress(panel_label+" preparation started");
        const auto root = task.root, first = task.first;
        const auto& all_sources = source_columns_by_root[sector][root];
        if (options.disk_parallel_panels && prepared_first != first) {
          // Every rank enters once per wave, even ranks with no source rows.
          // Each owns its disk file; all virtual transforms run concurrently.
          bool failed = false;
          try {
            const auto local_rows = source_columns_by_root[sector][world.rank].size();
            const auto count = first < local_rows
                ? std::min(options.propagation_row_batch, local_rows - first) : 0;
            const auto elements = checked_product(count, virtual_count, "ready disk panel");
            ready_panel.resize(elements);
            if (count) {
              std::vector<double> input(elements);
              const auto read_begin = Clock::now();
              sector_moments.disk_doubles->read(first, input);
              result.disk_panel_read_seconds += seconds_since(read_begin);
              const auto transform_begin = Clock::now();
              transform_virtual_rows(input, count, virtual_sectors.sectors[sector],
                  sector, virtual_propagator, ready_panel);
              result.disk_panel_transform_seconds += seconds_since(transform_begin);
            }
          } catch (const std::exception&) { failed = true; }
          const auto ready_begin = Clock::now();
          const auto failures = parallel::sum_group(static_cast<std::uint64_t>(failed), 0);
          result.communication_seconds += seconds_since(ready_begin);
          if (failures)
            throw std::runtime_error("parallel disk panel preparation failed; point not checkpointed");
          prepared_first = first;
          ++result.disk_panel_waves;
        }
        const auto source_columns = std::span<const std::size_t>(all_sources).subspan(
            first, std::min(options.propagation_row_batch, all_sources.size() - first));
        const auto panel_elements = checked_product(
            source_columns.size(), virtual_count, "source moment panel");
        if (world.size != 1 || disk_moments) panel.assign(panel_elements, 0.0);
        if (world.rank == root) {
          if(disk_moments) {
            if (options.disk_parallel_panels) {
              if (ready_panel.size() != panel_elements)
                throw std::logic_error("prepared disk panel dimension mismatch");
              panel.swap(ready_panel);
            } else {
            // Transform one owned source panel, broadcast it once, then
            // consume it for every target. No full transformed tensor.
            std::vector<double> input(panel_elements);
            const auto read_begin = Clock::now();
            if(trace_panel) progress(panel_label+" source read started");
            sector_moments.disk_doubles->read(first,input);
            if(trace_panel) progress(panel_label+" source read complete; virtual transform started");
            result.disk_panel_read_seconds += seconds_since(read_begin);
            const auto transform_begin = Clock::now();
            transform_virtual_rows(input,source_columns.size(),virtual_sectors.sectors[sector],
                sector,virtual_propagator,panel);
            result.disk_panel_transform_seconds += seconds_since(transform_begin);
            if(trace_panel) progress(panel_label+" virtual transform complete");
            }
          } else {
          if (checked_product(all_sources.size(), virtual_count, "owned moment size") != sector_moments.virtual_transformed.size())
            throw std::logic_error(
                "owned triples moment panel dimensions disagree");
          if (world.size != 1) std::copy_n(
              sector_moments.virtual_transformed.data() + first * virtual_count, panel_elements, panel.data());
          }
        }
        const auto communication_begin = Clock::now();
        if(trace_panel) progress(panel_label+" broadcast started");
        if (world.size != 1) parallel::broadcast_in_place(panel, root);
        if(trace_panel) progress(panel_label+" broadcast complete; occupied contraction started");
        result.communication_seconds += seconds_since(communication_begin);
        if (target_count == 0 || source_columns.empty() ||
            virtual_count == 0)
          continue;
        const auto compound_elements = checked_product(
            target_count, source_columns.size(), "occupied compound panel");
        compound.resize(compound_elements);
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static) \
    if(compound_elements >= 4096)
#endif
        for (std::ptrdiff_t target = 0;
             target < static_cast<std::ptrdiff_t>(target_count); ++target)
          for (std::ptrdiff_t source = 0;
               source < static_cast<std::ptrdiff_t>(source_columns.size());
               ++source)
            compound[static_cast<std::size_t>(target) *
                         source_columns.size() +
                     static_cast<std::size_t>(source)] =
                compound_element(
                    occupied_propagator,
                    occupied_sectors.sectors[sector]
                        [sector_moments.owned_columns[
                            target_rows[static_cast<std::size_t>(target)]]],
                    occupied_sectors.sectors[sector]
                        [source_columns[static_cast<std::size_t>(source)]]);
        const int rows = blas_dimension(virtual_count, "virtual triple count");
        const int contraction =
            blas_dimension(source_columns.size(), "source occupied triples");
        const auto* source = world.size == 1 && !disk_moments
            ? sector_moments.virtual_transformed.data() + first * virtual_count : panel.data();
        // Serial BLAS per OpenMP-owned output block, no MPI inside workers.
        const std::size_t output_block = disk_moments ? 1 : 8;
        const auto output_blocks = (target_count + output_block - 1) / output_block;
        if (options.sector_streaming) {
          constexpr std::size_t column_tile = 4096;
          constexpr double beta_zero = 0.0;
          std::atomic<bool> disk_read_failed{false};
          std::vector<std::string> worker_failures(thread_count);
#if defined(_OPENMP)
#pragma omp parallel num_threads(static_cast<int>(thread_count))
#endif
          {
            std::size_t thread = 0;
#if defined(_OPENMP)
            thread = static_cast<std::size_t>(omp_get_thread_num());
#endif
            std::vector<double> response_tile(output_block * column_tile);
            std::vector<double> packed_source;
            std::vector<double> unused_left_row;
            auto& left_row=disk_moments ? disk_left_rows[thread] : unused_left_row;
            auto energies = fused_energies[thread];
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (std::size_t block = 0; block < output_blocks; ++block) {
              try {
              const auto target = block * output_block;
              const int columns = static_cast<int>(std::min(output_block, target_count - target));
              if(disk_moments) {
                try { sector_moments.disk_left->read(target_rows[target],left_row); }
                catch(const std::exception&) { disk_read_failed.store(true); continue; }
              }
              for (std::size_t vfirst = 0; vfirst < virtual_count; vfirst += column_tile) {
                const int tile_rows = static_cast<int>(std::min(column_tile, virtual_count - vfirst));
                if(disk_moments) {
                  detail::contract_disk_occupied_panel(panel,virtual_count,
                      std::span<const double>(compound).subspan(target*source_columns.size(),source_columns.size()),
                      vfirst,std::span<double>(response_tile).first(static_cast<std::size_t>(tile_rows)),packed_source);
                } else dgemm_(&no_transpose, &no_transpose, &tile_rows, &columns, &contraction,
                    &alpha, source + vfirst, &rows,
                    compound.data() + target * source_columns.size(), &contraction,
                    &beta_zero, response_tile.data(), &tile_rows);
                for (int r = 0; r < columns; ++r) {
                  const auto row = target + static_cast<std::size_t>(r);
                  const auto length = static_cast<std::size_t>(tile_rows);
                  const auto* orders = sector_moments.pattern_orders.data() +
                      sector_moments.occupied_patterns[target_rows[row]] * sector_moments.virtual_pattern_count;
                  detail::accumulate_triples_energy_row(
                      {disk_moments ? left_row.data()+vfirst : sector_moments.left.data() + target_rows[row] * virtual_count + vfirst, length},
                      {response_tile.data() + static_cast<std::size_t>(r) * length, length},
                      {sector_moments.virtual_patterns.data() + vfirst, length},
                      orders, result.weights[point]*static_cast<double>(orbit_sizes[row]), energies);
                }
              }
              } catch(const std::exception& e) {worker_failures[thread]=e.what();disk_read_failed.store(true);}
            }
            fused_energies[thread] = energies;
          }
          if(disk_read_failed.load()) {
            for(const auto& failure:worker_failures) if(!failure.empty())
              throw std::runtime_error("triples disk occupied worker failed: "+failure);
            throw std::runtime_error("triples scratch left-row read failed; energy not checkpointed");
          }
        } else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::size_t block = 0; block < output_blocks; ++block) {
          const auto target = block * output_block;
          const int columns = static_cast<int>(std::min(output_block, target_count - target));
          dgemm_(&no_transpose, &no_transpose, &rows, &columns, &contraction,
                 &alpha, source, &rows, compound.data() + target * source_columns.size(), &contraction,
                 &beta_one, sector_moments.response.data() + target * virtual_count, &rows);
        }
        }
        if(trace_panel) progress(panel_label+" occupied contraction complete");
      }
    // Seven slots: the six many-body classes and, in the last, a total
    // accumulated without consulting the classification at all.  Keeping the
    // two paths separate is what lets the reconstruction identity below be a
    // measurement rather than a restatement.
      if (options.sector_streaming) {
        for (const auto& energies : fused_energies)
          for (std::size_t order = 0; order < point_energies.size(); ++order)
            point_energies[order] += energies[order];
      } else {
      const auto elements = sector_moments.response.size();
      const auto partition_begin = Clock::now();
      std::vector<double> thread_energies(
          checked_product(thread_count, std::size_t{7},
                          "thread triples energies"),
          0.0);
#if defined(_OPENMP)
#pragma omp parallel
#endif
      {
        // Private bins avoid workers repeatedly writing adjacent cache lines.
        std::array<double, 7> energies{};
        std::size_t thread = 0;
#if defined(_OPENMP)
        thread = static_cast<std::size_t>(omp_get_thread_num());
#pragma omp for schedule(static)
#endif
        for (std::size_t block = 0;
             block < elements / 4096 + (elements % 4096 != 0); ++block) {
          auto index = block * 4096;
          const auto end = index + std::min(std::size_t{4096}, elements - index);
          while (index < end) {
            const auto row = index / virtual_count;
            const auto column = index % virtual_count;
            const auto length = std::min(end - index, virtual_count - column);
            const auto* orders = sector_moments.pattern_orders.data() +
                sector_moments.occupied_patterns[target_rows[row]] * sector_moments.virtual_pattern_count;
            detail::accumulate_triples_energy_row(
                {moments[sector].left.data() + target_rows[row]*virtual_count+column, length},
                {moments[sector].response.data() + index, length},
                {sector_moments.virtual_patterns.data() + column, length},
                orders, result.weights[point]*static_cast<double>(orbit_sizes[row]), energies);
            index += length;
          }
        }
        std::copy(energies.begin(), energies.end(), thread_energies.begin() + 7 * thread);
      }
      for (std::size_t thread = 0; thread < thread_count; ++thread)
        for (std::size_t order = 0; order < point_energies.size(); ++order)
          point_energies[order] += thread_energies[7 * thread + order];
      result.partition_accumulation_seconds += seconds_since(partition_begin);
      }
      std::vector<double>().swap(sector_moments.virtual_transformed);
      std::vector<double>().swap(sector_moments.response);
      progress("Laplace triples sector " + std::to_string(sector) + " complete");
  };
  const auto finish_point = [&](const std::size_t point, std::vector<double>& point_energies,
                                const double elapsed) {
    parallel::sum_in_place(point_energies);
    if (!std::all_of(point_energies.begin(), point_energies.end(),
                     [](double energy) { return std::isfinite(energy); }))
      throw std::runtime_error("non-finite triples point energy; checkpoint not written");
    // Only the six classes enter the checkpointed layout; the unclassified
    // total is a per-run diagnostic and is accumulated separately.
    for (std::size_t order = 0; order < 6; ++order)
      result.point_many_body_energies[6 * point + order] =
          point_energies[order];
    unclassified_total += point_energies[6];
    unclassified_points += 1;
    result.completed_points[point] = 1U;
    if (options.checkpoint_observer) {
      progress("Laplace triples point checkpoint started");
      options.checkpoint_observer(StreamedLaplaceTriplesProgress{
          result.point_many_body_energies, result.completed_points});
      progress("Laplace triples point checkpoint complete");
    }
    if (world.rank == 0 && options.observer)
      options.observer(StreamedLaplaceTriplesPoint{
          .point = point + 1,
          .points = result.nodes.size(),
          .node = result.nodes[point],
          .weight = result.weights[point],
          .seconds = elapsed,
          .many_body_energies = std::vector<double>(point_energies.begin(), point_energies.begin() + 6)});
  };
  if (options.sector_streaming) {
    const bool persistent=!options.sector_checkpoint_path.empty();
    if(persistent && (options.sector_checkpoint_identity.empty() || restarting || options.pm_symmetry ||
        options.block_screening_budget_hartree!=0))
      throw std::invalid_argument("sector checkpoints require identity, fresh certified-point state, no symmetry or screening");
    std::ostringstream key;
    key << std::setprecision(17) << options.sector_checkpoint_identity << ' ' << world.size
        << ' ' << occupied << ' ' << virtuals << ' ' << result.quadrature_points
        << ' ' << options.requested_energy_error_hartree << ' ' << options.denominator_floor;
    for(auto x:result.nodes) key << ' ' << x;
    for(auto x:result.weights) key << ' ' << x;
    for(auto x:occupied_monomers) key << ' ' << x;
    for(auto x:virtual_monomers) key << ' ' << x;
    key << " two-spin=" << options.two_spin_sectors;
    const auto sector_stride=2+8*result.quadrature_points;
    std::vector<double> saved(4*sector_stride,0.);
    bool load_failed=false;
    if(persistent && world.rank==0) try {
      saved=detail::read_sector_checkpoint(options.sector_checkpoint_path,key.str(),saved.size());
    } catch(const std::exception& e) { progress(e.what()); load_failed=true; }
    if(parallel::sum_group(static_cast<std::uint64_t>(load_failed),0))
      throw std::runtime_error("triples sector checkpoint load rejected");
    if(persistent && world.size!=1) parallel::broadcast_in_place(saved,0);
    for(std::size_t s=0;s<4;++s) {
      if(saved[s*sector_stride]<0 || saved[s*sector_stride+1]<0)
        throw std::runtime_error("invalid triples checkpoint norms");
      for(std::size_t p=0;p<result.quadrature_points;++p) {
        const auto flag=saved[s*sector_stride+2+8*p];
        if(flag!=0. && flag!=1.) throw std::runtime_error("invalid triples sector completion flag");
      }
    }
    progress(options.two_spin_sectors ? "two-sector triples: sectors 0/1 represent 3/2; full nM; multiplicity 2" :
        "sector-streamed triples: full sectors retained; partial-sector records are not certified final energies");
    std::vector<std::vector<double>> point_energies(result.quadrature_points, std::vector<double>(7, 0.0));
    std::vector<double> point_seconds(result.quadrature_points, 0.0);
    for (std::size_t sector = 0; sector < (options.two_spin_sectors ? 2U : sector_count); ++sector) {
      const auto base=sector*sector_stride;
      bool pending=false;
      for(std::size_t p=0;p<result.quadrature_points;++p)
        pending=pending || (result.completed_points[p]==0 && saved[base+2+8*p]==0.);
      if(pending) {
        const auto previous=norms;
        build_moments(sector, sector + 1);
        std::vector<double> delta{norms[0]-previous[0],norms[1]-previous[1]};
        parallel::sum_in_place(delta);
        saved[base]=delta[0]; saved[base+1]=delta[1];
      } else if(persistent) {
        if(world.rank==0) { norms[0]+=saved[base]; norms[1]+=saved[base+1]; }
        progress("triples sector "+std::to_string(sector)+" restored; moment construction skipped");
      } else build_moments(sector,sector+1);
      for (std::size_t point = 0; point < result.quadrature_points; ++point) {
        if (result.completed_points[point] != 0U) continue;
        const auto entry=base+2+8*point;
        if(saved[entry]==1.) {
          if(world.rank==0) for(std::size_t m=0;m<7;++m)
            point_energies[point][m]+=(options.two_spin_sectors ? 2. : 1.)*saved[entry+1+m];
          progress("triples sector "+std::to_string(sector)+" point "+std::to_string(point+1)+" restored; propagation skipped");
          continue;
        }
        const auto begin = Clock::now();
        std::vector<double> contribution(7,0.);
        propagate_sector(point, sector, contribution);
        if(persistent) {
          parallel::sum_in_place(contribution);
          saved[entry]=1.;
          std::copy(contribution.begin(),contribution.end(),saved.begin()+entry+1);
          bool failed=false;
          if(world.rank==0) try {
            detail::write_sector_checkpoint(options.sector_checkpoint_path,key.str(),saved);
          } catch(const std::exception& e) { progress(e.what()); failed=true; }
          if(parallel::sum_group(static_cast<std::uint64_t>(failed),0))
            throw std::runtime_error("triples sector checkpoint publication failed");
          progress("triples sector checkpoint durable; sector="+std::to_string(sector)+" point="+std::to_string(point+1));
          if(options.sector_checkpoint_observer) options.sector_checkpoint_observer(sector,point);
        }
        if(!persistent || world.rank==0) for(std::size_t m=0;m<7;++m)
          point_energies[point][m]+=(options.two_spin_sectors ? 2. : 1.)*contribution[m];
        point_seconds[point] += seconds_since(begin);
      }
      std::vector<double>().swap(moments[sector].doubles);
      std::vector<double>().swap(moments[sector].left);
      moments[sector].disk_doubles.reset(); moments[sector].disk_left.reset();
      progress("triples sector " + std::to_string(sector) + " finished all pending points; moments released");
    }
    certify_moments();
    for (std::size_t point = 0; point < result.quadrature_points; ++point)
      if (result.completed_points[point] == 0U)
        finish_point(point, point_energies[point], point_seconds[point]);
  } else {
    build_moments(0, sector_count);
    certify_moments();
    for (std::size_t point = 0; point < result.quadrature_points; ++point) {
      if (result.completed_points[point] != 0U) continue;
      const auto point_begin = Clock::now();
      std::vector<double> point_energies(7, 0.0);
      for (std::size_t sector = 0; sector < sector_count; ++sector)
        propagate_sector(point, sector, point_energies);
      finish_point(point, point_energies, seconds_since(point_begin));
    }
  }
  result.propagator_seconds = seconds_since(calculation_begin) -
                              result.moment_construction_seconds;
  for (std::size_t point = 0; point < result.quadrature_points; ++point) {
    if (result.completed_points[point] != 1U)
      throw std::logic_error(
          "streamed Laplace triples left an incomplete quadrature point");
    for (std::size_t order = 0; order < 6; ++order)
      result.many_body_energies[order] +=
          result.point_many_body_energies[6 * point + order];
  }
  for (const auto energy : result.many_body_energies)
    result.parenthetical_energy += energy;

  // Reconstruction check.  The class sum is compared against the total
  // accumulated independently of the classification.  When every quadrature
  // point was evaluated in this run the two cover the same contributions and
  // the difference is a real measurement; when some points came from a
  // checkpoint there is no independent total for them, so the identity is not
  // asserted and the error is reported as not-a-number rather than as a
  // spurious zero.
  if (unclassified_points == result.quadrature_points) {
    double partition_sum = 0.0;
    for (const auto energy : result.many_body_energies) partition_sum += energy;
    result.partition_error = partition_sum - unclassified_total;
  } else {
    result.partition_error = std::numeric_limits<double>::quiet_NaN();
  }
  return result;
}

StreamedLaplaceTriplesResult solve_localized_triples_laplace_streamed(
    const SpatialRhfHamiltonian& hamiltonian, const Matrix& t1,
    const Tensor4& t2, const std::vector<int>& occupied_monomers,
    const std::vector<int>& virtual_monomers,
    const StreamedLaplaceTriplesOptions& options) {
  return solve_localized_triples_laplace_streamed_impl(
      hamiltonian, t1, t2, occupied_monomers, virtual_monomers, options);
}

StreamedLaplaceTriplesResult solve_localized_triples_laplace_streamed(
    const SpatialRhfHamiltonian& hamiltonian, const Matrix& t1,
    const SelectedDoublesView& t2,
    const std::vector<int>& occupied_monomers,
    const std::vector<int>& virtual_monomers,
    const StreamedLaplaceTriplesOptions& options) {
  return solve_localized_triples_laplace_streamed_impl(
      hamiltonian, t1, t2, occupied_monomers, virtual_monomers, options);
}

} // namespace localized_cc
