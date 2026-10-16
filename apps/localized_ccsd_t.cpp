#include "localized_cc/basis_projection.hpp"
#include "localized_cc/canonical_triples.hpp"
#include "localized_cc/checkpoint.hpp"
#include "localized_cc/cuda_triples.hpp"
#include "localized_cc/distributed_pair_tensor.hpp"
#include "localized_cc/ccsd.hpp"
#include "localized_cc/frozen_core.hpp"
#include "localized_cc/hamiltonian_rotation.hpp"
#include "localized_cc/laplace_triples.hpp"
#include "localized_cc/lmp2_domains.hpp"
#include "localized_cc/lmp2_preintegral.hpp"
#include "localized_cc/lmp2_adaptive.hpp"
#include "localized_cc/parallel.hpp"
#include "localized_cc/pm_symmetry.hpp"
#include "localized_cc/pm_adapted.hpp"
#include "localized_cc/partition.hpp"
#include "localized_cc/selected_ccsd_manifold.hpp"
#include "localized_cc/selected_ccsd_storage.hpp"
#include "localized_cc/selected_semidirect_hamiltonian.hpp"
#include "localized_cc/semidirect_hamiltonian.hpp"
#include "localized_cc/semicanonical_triples.hpp"
#include "localized_cc/sparse_ccsd.hpp"
#include "localized_cc/triples.hpp"
#include "localized_cc/water_zmatrix.hpp"
#include "modernqc/integrals/integral_provider.hpp"
#include "modernqc/localization/orbital_localization.hpp"
#include "modernqc/localization/pipek_mezey.hpp"
#include "modernqc/molecule/molecule.hpp"
#include "modernqc/scf/rhf.hpp"
#include "../src/hermitian_one_body.hpp"

#include <chrono>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

enum class RunMode { plan, canonical, localized, both };
enum class CoreMode { atomic, none };
enum class RestartMode { automatic, required, none };
enum class Localizer { pm, fm2, fm4 };
enum class LocalizedTriplesRepresentation { semicanonical, laplace };
enum class ExecutionStage { full, ccsd_only };
enum class SelectedCcsdExecutor {
  dense_oracle,
  selected_kernel,
  block_sparse,
};
enum class SelectedIntegralBackend { dense_oracle, ao_direct_selected };
enum class LocalizedHamiltonianBackend { ao_semidirect, canonical_rotation };

// The full CCSD equations are invariant to separate unitary rotations of the
// occupied and virtual spaces.  The canonical option is an execution choice:
// solve the untruncated equations in the diagonal-Fock RHF basis, then rotate
// the converged amplitudes into the localized basis before any nM partition or
// localized triples calculation.  It is deliberately rejected for a
// truncated selected manifold.
enum class CcsdSolverBasis { localized, canonical };
enum class CcsdSolverBackend { spin_orbital, legacy_spatial };

double elapsed_seconds(const std::chrono::steady_clock::time_point begin,
                       const std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

struct ParsedGeometry {
  std::vector<modernqc::molecule::Atom> atoms;
  std::string point_group;
};

ParsedGeometry read_xyz(const std::string& path) {
  std::ifstream input(path);
  std::size_t atom_count = 0;
  if (!input || !(input >> atom_count))
    throw std::runtime_error("invalid XYZ file: " + path);
  input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  std::string comment;
  std::getline(input, comment);
  std::vector<modernqc::molecule::Atom> atoms;
  atoms.reserve(atom_count);
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    std::string symbol;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!(input >> symbol >> x >> y >> z))
      throw std::runtime_error("truncated XYZ file: " + path);
    const int atomic_number = symbol == "H" ? 1 : (symbol == "O" ? 8 : 0);
    if (atomic_number == 0)
      throw std::runtime_error("the molecular driver supports only H and O");
    atoms.push_back({symbol, atomic_number,
                     {x * localized_cc::angstrom_to_bohr,
                      y * localized_cc::angstrom_to_bohr,
                      z * localized_cc::angstrom_to_bohr}});
  }
  return {std::move(atoms), "not assigned (XYZ input)"};
}

ParsedGeometry read_water_zmatrix(const std::string& path) {
  const auto zmat = localized_cc::read_water_zmatrix(path);
  std::vector<modernqc::molecule::Atom> atoms;
  for (const auto& atom : zmat.cartesian_atoms())
    atoms.push_back({atom.symbol, atom.atomic_number, atom.position_bohr});
  return {std::move(atoms), zmat.point_group()};
}

ParsedGeometry read_geometry(const std::string& path) {
  if (std::string_view(path).ends_with(".zmat"))
    return read_water_zmatrix(path);
  return read_xyz(path);
}

localized_cc::Matrix copy_matrix(const modernqc::linalg::Matrix& source) {
  localized_cc::Matrix result({source.rows(), source.columns()});
  for (std::size_t row = 0; row < source.rows(); ++row)
    for (std::size_t column = 0; column < source.columns(); ++column)
      result(row, column) = source(row, column);
  return result;
}

modernqc::linalg::Matrix copy_matrix(const localized_cc::Matrix& source) {
  const auto dimensions = source.dimensions();
  modernqc::linalg::Matrix result(dimensions[0], dimensions[1]);
  for (std::size_t row = 0; row < dimensions[0]; ++row)
    for (std::size_t column = 0; column < dimensions[1]; ++column)
      result(row, column) = source(row, column);
  return result;
}

localized_cc::RhfCheckpointState checkpoint_rhf(
    const modernqc::scf::RhfResult& result) {
  return {result.nuclear_repulsion_energy,
          result.electronic_energy,
          result.total_energy,
          copy_matrix(result.overlap),
          copy_matrix(result.core_hamiltonian),
          copy_matrix(result.density),
          copy_matrix(result.fock),
          copy_matrix(result.coefficients),
          result.orbital_energies,
          result.occupations};
}

modernqc::scf::RhfResult restore_rhf(
    const localized_cc::RhfCheckpointState& state,
    const std::size_t expected_ao_count) {
  const auto square =
      std::array<std::size_t, 2>{expected_ao_count, expected_ao_count};
  if (state.overlap.dimensions() != square ||
      state.core_hamiltonian.dimensions() != square ||
      state.density.dimensions() != square || state.fock.dimensions() != square ||
      state.coefficients.dimensions() != square ||
      state.orbital_energies.size() != expected_ao_count ||
      state.occupations.size() != expected_ao_count)
    throw std::runtime_error("RHF checkpoint dimensions do not match basis");
  if (!std::isfinite(state.nuclear_repulsion_energy) ||
      !std::isfinite(state.electronic_energy) ||
      !std::isfinite(state.total_energy) ||
      std::any_of(state.orbital_energies.begin(), state.orbital_energies.end(),
                  [](const double value) { return !std::isfinite(value); }) ||
      std::any_of(state.occupations.begin(), state.occupations.end(),
                  [](const double value) { return !std::isfinite(value); }))
    throw std::runtime_error("RHF checkpoint contains non-finite data");
  modernqc::scf::RhfResult result{};
  result.converged = true;
  result.nuclear_repulsion_energy = state.nuclear_repulsion_energy;
  result.electronic_energy = state.electronic_energy;
  result.total_energy = state.total_energy;
  result.overlap = copy_matrix(state.overlap);
  result.core_hamiltonian = copy_matrix(state.core_hamiltonian);
  result.density = copy_matrix(state.density);
  result.fock = copy_matrix(state.fock);
  result.coefficients = copy_matrix(state.coefficients);
  result.orbital_energies = state.orbital_energies;
  result.occupations = state.occupations;
  return result;
}

std::string calculation_fingerprint(const ParsedGeometry& geometry,
                                    const std::string& basis_name,
                                    const CoreMode core_mode) {
  std::ostringstream text;
  text << "localized-ccsd-t-v1|spherical|charge=0|multiplicity=1|basis="
       << basis_name << "|core="
       << (core_mode == CoreMode::atomic ? "atomic" : "none")
       << "|atoms=" << geometry.atoms.size();
  text << std::hexfloat;
  for (const auto& atom : geometry.atoms)
    text << '|' << atom.atomic_number << ':' << atom.position_bohr[0] << ','
         << atom.position_bohr[1] << ',' << atom.position_bohr[2];
  return text.str();
}

std::string fingerprint_basis(const std::string& fingerprint) {
  constexpr std::string_view key = "|basis=";
  const auto begin = fingerprint.find(key);
  if (begin == std::string::npos)
    throw std::runtime_error(
        "projection-source checkpoint fingerprint has no basis field");
  const auto value_begin = begin + key.size();
  const auto end = fingerprint.find('|', value_begin);
  if (end == value_begin)
    throw std::runtime_error(
        "projection-source checkpoint fingerprint has an empty basis field");
  return fingerprint.substr(value_begin, end - value_begin);
}

std::string fingerprint_without_basis(const std::string& fingerprint) {
  constexpr std::string_view key = "|basis=";
  const auto begin = fingerprint.find(key);
  if (begin == std::string::npos)
    throw std::runtime_error("checkpoint fingerprint has no basis field");
  const auto end = fingerprint.find('|', begin + key.size());
  if (end == std::string::npos)
    return fingerprint.substr(0, begin);
  return fingerprint.substr(0, begin) + fingerprint.substr(end);
}

void validate_projection_source_rhf(
    const localized_cc::RhfCheckpointState& state,
    const std::size_t occupied_count) {
  const auto overlap_dimensions = state.overlap.dimensions();
  if (overlap_dimensions[0] == 0 ||
      overlap_dimensions[0] != overlap_dimensions[1] ||
      state.coefficients.dimensions() != overlap_dimensions ||
      state.occupations.size() != overlap_dimensions[0] ||
      overlap_dimensions[0] <= occupied_count)
    throw std::runtime_error(
        "projection-source RHF checkpoint dimensions are inconsistent");
  for (std::size_t orbital = 0; orbital < state.occupations.size(); ++orbital) {
    const double expected = orbital < occupied_count ? 2.0 : 0.0;
    if (!std::isfinite(state.occupations[orbital]) ||
        std::abs(state.occupations[orbital] - expected) > 1.0e-12)
      throw std::runtime_error(
          "projection-source RHF occupations are not closed-shell canonical");
  }
}

double metric_orthonormality_error(
    const modernqc::linalg::Matrix& coefficients,
    const modernqc::linalg::Matrix& overlap) {
  if (coefficients.empty() || overlap.rows() != overlap.columns() ||
      coefficients.rows() != overlap.rows())
    throw std::invalid_argument(
        "metric-orthonormality check dimensions mismatch");
  const auto gram = modernqc::linalg::multiply(
      modernqc::linalg::transpose(coefficients),
      modernqc::linalg::multiply(overlap, coefficients));
  return modernqc::linalg::maximum_absolute_value(
      modernqc::linalg::subtract(
          gram,
          modernqc::linalg::Matrix::identity(coefficients.columns())));
}

struct MonomerAssignments {
  std::vector<int> labels;
  std::vector<double> confidence;
  std::size_t confident{};
};

MonomerAssignments assign_water_monomers(
    const modernqc::linalg::Matrix& coefficients,
    const modernqc::linalg::Matrix& overlap,
    const std::vector<std::vector<std::size_t>>& centers,
    const std::size_t atom_count, const double threshold) {
  if (atom_count == 0 || atom_count % 3 != 0 || centers.size() != atom_count)
    throw std::invalid_argument(
        "water monomer assignment requires consecutive O-H-H atom triples");
  if (!(threshold > 0.0 && threshold <= 1.0) || !std::isfinite(threshold))
    throw std::invalid_argument("assignment threshold must be in (0,1]");
  const auto populations = modernqc::localization::lowdin_center_populations(
      coefficients, overlap, centers);
  MonomerAssignments result;
  result.labels.resize(coefficients.columns());
  result.confidence.resize(coefficients.columns());
  const auto monomers = atom_count / 3;
  for (std::size_t orbital = 0; orbital < coefficients.columns(); ++orbital) {
    double best = -1.0;
    std::size_t best_monomer = 0;
    for (std::size_t monomer = 0; monomer < monomers; ++monomer) {
      double population = 0.0;
      for (std::size_t atom = 3 * monomer; atom < 3 * monomer + 3; ++atom)
        population += populations(atom, orbital);
      if (population > best) {
        best = population;
        best_monomer = monomer;
      }
    }
    constexpr double population_roundoff_tolerance = 1.0e-10;
    if (!std::isfinite(best) || best < -population_roundoff_tolerance ||
        best > 1.0 + population_roundoff_tolerance)
      throw std::runtime_error(
          "Löwdin monomer confidence is outside its physical [0,1] domain");
    best = std::clamp(best, 0.0, 1.0);
    result.labels[orbital] =
        best >= threshold ? static_cast<int>(best_monomer) : -1;
    result.confidence[orbital] = best;
    if (best >= threshold)
      ++result.confident;
  }
  return result;
}

modernqc::linalg::Matrix columns(const modernqc::linalg::Matrix& source,
                                 const std::size_t first,
                                 const std::size_t count) {
  modernqc::linalg::Matrix result(source.rows(), count);
  for (std::size_t row = 0; row < source.rows(); ++row)
    for (std::size_t column = 0; column < count; ++column)
      result(row, column) = source(row, first + column);
  return result;
}

std::vector<std::vector<std::size_t>> atom_centers(
    const modernqc::basis::BasisSet& basis, const std::size_t atom_count) {
  std::vector<std::vector<std::size_t>> result(atom_count);
  for (const auto& shell : basis.shells)
    for (std::size_t function = 0; function < shell.function_count; ++function)
      result[shell.atom_index].push_back(shell.first_ao + function);
  return result;
}

localized_cc::CcsdResult run_ccsd(
    const localized_cc::SpatialRhfHamiltonian& hamiltonian,
    localized_cc::CcsdOptions options = {}) {
  options.max_iterations = std::max<std::size_t>(
      options.max_iterations, options.completed_iterations + 2);
  auto result = localized_cc::solve_ccsd(hamiltonian, options);
  if (!result.converged)
    throw std::runtime_error("CCSD failed to converge");
  return result;
}

localized_cc::CcsdResult run_legacy_spatial_ccsd(
    const localized_cc::SpatialRhfHamiltonian& hamiltonian,
    localized_cc::CcsdOptions options = {}) {
  options.max_iterations = std::max<std::size_t>(
      options.max_iterations, options.completed_iterations + 2);
  if (options.projection_mask)
    throw std::invalid_argument(
        "legacy-spatial CCSD is already the exact closed-shell manifold and "
        "does not accept a spin projection mask");
  auto spatial = localized_cc::solve_spatial_rccsd_reference(
      hamiltonian, options);
  if (!spatial.converged)
    throw std::runtime_error("legacy-spatial CCSD failed to converge");
  localized_cc::CcsdResult result;
  result.correlation_energy = spatial.correlation_energy;
  result.t1 = localized_cc::expand_spatial_rccsd_t1_to_spin_orbital(
      spatial.amplitudes.t1);
  result.t2 = localized_cc::expand_spatial_rccsd_t2_to_spin_orbital(
      spatial.amplitudes.t2);
  result.r1_norm = spatial.r1_norm;
  result.r2_norm = spatial.r2_norm;
  result.history = std::move(spatial.history);
  result.converged = spatial.converged;
  result.amplitude_backend = localized_cc::CcsdAmplitudeBackend::replicated;
  result.wabef_backend = options.wabef_backend;
  result.wabef_packing_backend = options.wabef_packing_backend;
  result.wabef_invariant_cache_bytes =
      spatial.wabef_invariant_cache_bytes;
  result.wabef_invariant_cache_build_seconds =
      spatial.wabef_invariant_cache_build_seconds;
  result.spatial_residual_workspace_bytes =
      spatial.residual_workspace_bytes;
  result.spatial_residual_workspace_build_seconds =
      spatial.residual_workspace_build_seconds;
  result.cumulative_timing = spatial.cumulative_timing;
  return result;
}

localized_cc::CcsdResult run_ccsd(
    const localized_cc::SelectedSpatialHamiltonian& hamiltonian,
    localized_cc::CcsdOptions options = {}) {
  options.max_iterations = std::max<std::size_t>(
      options.max_iterations, options.completed_iterations + 2);
  auto result = localized_cc::solve_ccsd(
      hamiltonian.fock, hamiltonian.nocc, hamiltonian.nvirt,
      hamiltonian.integrals, options);
  if (!result.converged)
    throw std::runtime_error("selected-integral CCSD failed to converge");
  return result;
}

localized_cc::CcsdAmplitudeBackend restored_ccsd_backend(
    const localized_cc::CcsdOptions& options,
    const localized_cc::CcsdCheckpointState& state,
    const bool global_arrays_active) {
  if (options.amplitude_backend !=
      localized_cc::CcsdAmplitudeBackend::automatic)
    return options.amplitude_backend;
  if (state.t2.size() >
      std::numeric_limits<std::size_t>::max() / sizeof(double))
    throw std::overflow_error("saved CCSD T2 byte count overflows");
  const auto t2_bytes = state.t2.size() * sizeof(double);
  return global_arrays_active && options.use_global_arrays_diis &&
                 t2_bytes > options.replicated_t2_max_bytes
             ? localized_cc::CcsdAmplitudeBackend::global_arrays
             : localized_cc::CcsdAmplitudeBackend::replicated;
}

std::filesystem::path ccsd_stage_checkpoint_path(
    const std::filesystem::path& checkpoint_path) {
  const auto extension = checkpoint_path.extension();
  if (extension.empty())
    return std::filesystem::path(checkpoint_path.string() + ".ccsd");
  return checkpoint_path.parent_path() /
         (checkpoint_path.stem().string() + ".ccsd" + extension.string());
}

RunMode parse_mode(const std::string& mode) {
  if (mode == "plan")
    return RunMode::plan;
  if (mode == "canonical")
    return RunMode::canonical;
  if (mode == "localized")
    return RunMode::localized;
  if (mode == "both")
    return RunMode::both;
  throw std::invalid_argument(
      "MODE must be plan, canonical, localized, or both");
}

CoreMode parse_core_mode(const std::string& mode) {
  if (mode == "atomic")
    return CoreMode::atomic;
  if (mode == "none")
    return CoreMode::none;
  throw std::invalid_argument("CORE must be atomic or none");
}

RestartMode parse_restart_mode(const std::string& mode) {
  if (mode == "auto")
    return RestartMode::automatic;
  if (mode == "required")
    return RestartMode::required;
  if (mode == "none")
    return RestartMode::none;
  throw std::invalid_argument("RESTART must be auto, required, or none");
}

Localizer parse_localizer(const std::string& method) {
  if (method == "pm")
    return Localizer::pm;
  if (method == "fm2" || method == "pm2")
    return Localizer::fm2;
  if (method == "fm4" || method == "pm4")
    return Localizer::fm4;
  throw std::invalid_argument("LOCALIZER must be pm, fm2, or fm4");
}

const char* localizer_name(const Localizer method) {
  if (method == Localizer::pm)
    return "pm";
  return method == Localizer::fm2 ? "fm2" : "fm4";
}

const char* localization_checkpoint_method(const Localizer method) {
  if (method == Localizer::pm)
    return "pm-pyscf-ciah-meta-lowdin";
  return localizer_name(method);
}

double many_body_energy(const localized_cc::EnergyPartition& partition,
                        const std::size_t order) {
  const auto found = partition.many_body_classes.find(
      "M" + std::to_string(order));
  return found == partition.many_body_classes.end() ? 0.0 : found->second;
}

localized_cc::CanonicalTriplesBackend triples_backend_from_environment() {
  const char* value = std::getenv("CCSD_TRIPLES_BACKEND");
  if (value == nullptr || *value == '\0' || std::string_view(value) == "cpu")
    return localized_cc::CanonicalTriplesBackend::reference_cpu;
  if (std::string_view(value) == "tiled-cpu")
    return localized_cc::CanonicalTriplesBackend::tiled_cpu;
  if (std::string_view(value) == "cuda")
    return localized_cc::CanonicalTriplesBackend::cuda;
  if (std::string_view(value) == "tce-cuda")
    return localized_cc::CanonicalTriplesBackend::tce_cuda;
  throw std::invalid_argument(
      "CCSD_TRIPLES_BACKEND must be cpu, tiled-cpu, cuda, or tce-cuda");
}

LocalizedTriplesRepresentation
localized_triples_representation_from_environment() {
  const char* value = std::getenv("CCSD_LOCALIZED_TRIPLES_REPRESENTATION");
  if (value == nullptr || *value == '\0' ||
      std::string_view(value) == "laplace")
    return LocalizedTriplesRepresentation::laplace;
  if (std::string_view(value) == "semicanonical")
    return LocalizedTriplesRepresentation::semicanonical;
  throw std::invalid_argument(
      "CCSD_LOCALIZED_TRIPLES_REPRESENTATION must be semicanonical or "
      "laplace");
}

ExecutionStage execution_stage_from_environment() {
  const char* value = std::getenv("CCSD_EXECUTION_STAGE");
  if (value == nullptr || *value == '\0' ||
      std::string_view(value) == "full")
    return ExecutionStage::full;
  if (std::string_view(value) == "ccsd-only")
    return ExecutionStage::ccsd_only;
  throw std::invalid_argument(
      "CCSD_EXECUTION_STAGE must be full or ccsd-only");
}

localized_cc::LaplaceTriplesQuadratureScheme
laplace_triples_quadrature_scheme_from_environment() {
  const char* value = std::getenv("CCSD_LAPLACE_TRIPLES_SCHEME");
  if (value == nullptr || *value == '\0' ||
      std::string_view(value) == "minimax")
    return localized_cc::LaplaceTriplesQuadratureScheme::minimax;
  if (std::string_view(value) == "scaled-gauss-laguerre")
    return localized_cc::LaplaceTriplesQuadratureScheme::
        scaled_gauss_laguerre;
  throw std::invalid_argument(
      "CCSD_LAPLACE_TRIPLES_SCHEME must be minimax or "
      "scaled-gauss-laguerre");
}

double positive_double_from_environment(const char* name,
                                        const double default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return default_value;
  const std::string text(value);
  std::size_t parsed = 0;
  const auto result = std::stod(text, &parsed);
  if (parsed != text.size() || !std::isfinite(result) || result <= 0.0)
    throw std::invalid_argument(std::string(name) +
                                " must be finite and positive");
  return result;
}

double nonnegative_double_from_environment(const char* name,
                                           const double default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return default_value;
  const std::string text(value);
  std::size_t parsed = 0;
  const auto result = std::stod(text, &parsed);
  if (parsed != text.size() || !std::isfinite(result) || result < 0.0)
    throw std::invalid_argument(std::string(name) +
                                " must be finite and non-negative");
  return result;
}

// Cauchy-Schwarz bound on the atomic-orbital integrals:
//
//     |(mu nu | lambda sigma)| <= sqrt((mu nu | mu nu)) sqrt((la si | la si))
//
// Shell quartets whose bound falls below the threshold are skipped entirely.
// The bound is rigorous, so the discarded contributions are known to be
// smaller than the threshold rather than assumed to be; this is an integral
// screen and is unrelated to the monomer-span selection, which truncates the
// amplitude manifold instead.
//
// Zero is the default and disables the screen, which keeps the strict
// canonical gate exact.  Note the ceiling on what this can buy: screening
// applies only to stages that evaluate atomic-orbital quartets, which on the
// recorded four-node W2/aVDZ run were RHF, the AO integrals, and the MO
// Hamiltonian -- together about 72 s of 1591 s.  It does not touch the CCSD
// iterations that dominate the runtime.
double ao_schwarz_threshold_from_environment() {
  return nonnegative_double_from_environment("CCSD_AO_SCHWARZ_THRESHOLD", 0.0);
}

std::size_t selected_maximum_monomers_from_environment() {
  const char* value = std::getenv("CCSD_SELECTED_MAXIMUM_MONOMERS");
  if (value == nullptr || *value == '\0' ||
      std::string_view(value) == "full")
    return std::numeric_limits<std::size_t>::max();
  const std::string text(value);
  std::size_t parsed = 0;
  const auto result = std::stoull(text, &parsed);
  if (parsed != text.size() || result == 0 ||
      result > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument(
        "CCSD_SELECTED_MAXIMUM_MONOMERS must be full or a positive integer");
  return static_cast<std::size_t>(result);
}

SelectedCcsdExecutor selected_ccsd_executor_from_environment() {
  const char* value = std::getenv("CCSD_SELECTED_EXECUTOR");
  if (value == nullptr || *value == '\0' ||
      std::string_view(value) == "selected-kernel")
    return SelectedCcsdExecutor::selected_kernel;
  if (std::string_view(value) == "dense-oracle")
    return SelectedCcsdExecutor::dense_oracle;
  if (std::string_view(value) == "block-sparse")
    return SelectedCcsdExecutor::block_sparse;
  if (std::string_view(value) == "sparse-localized")
    return SelectedCcsdExecutor::block_sparse;
  if (std::string_view(value) == "selected-kernel")
    return SelectedCcsdExecutor::selected_kernel;
  throw std::invalid_argument(
      "CCSD_SELECTED_EXECUTOR must be dense-oracle, selected-kernel, "
      "sparse-localized, or block-sparse");
}

SelectedIntegralBackend selected_integral_backend_from_environment() {
  const char* value = std::getenv("CCSD_SELECTED_INTEGRAL_BACKEND");
  if (value == nullptr || *value == '\0' ||
      std::string_view(value) == "dense-oracle")
    return SelectedIntegralBackend::dense_oracle;
  if (std::string_view(value) == "ao-direct-selected")
    return SelectedIntegralBackend::ao_direct_selected;
  throw std::invalid_argument(
      "CCSD_SELECTED_INTEGRAL_BACKEND must be dense-oracle or "
      "ao-direct-selected");
}

LocalizedHamiltonianBackend localized_hamiltonian_backend_from_environment() {
  const char* value = std::getenv("CCSD_LOCALIZED_HAMILTONIAN_BACKEND");
  if (value == nullptr || *value == '\0' ||
      std::string_view(value) == "ao-semidirect")
    return LocalizedHamiltonianBackend::ao_semidirect;
  if (std::string_view(value) == "canonical-rotation")
    return LocalizedHamiltonianBackend::canonical_rotation;
  throw std::invalid_argument(
      "CCSD_LOCALIZED_HAMILTONIAN_BACKEND must be ao-semidirect or "
      "canonical-rotation");
}

std::string_view localized_hamiltonian_backend_name(
    const LocalizedHamiltonianBackend backend) {
  switch (backend) {
  case LocalizedHamiltonianBackend::ao_semidirect:
    return "independent AO-to-localized-MO semidirect transform";
  case LocalizedHamiltonianBackend::canonical_rotation:
    return "exact canonical-to-PM four-index rotation";
  }
  throw std::invalid_argument("unknown localized Hamiltonian backend");
}

CcsdSolverBasis ccsd_solver_basis_from_environment() {
  const char* value = std::getenv("CCSD_SOLVER_BASIS");
  if (value == nullptr || *value == '\0' ||
      std::string_view(value) == "localized")
    return CcsdSolverBasis::localized;
  if (std::string_view(value) == "canonical")
    return CcsdSolverBasis::canonical;
  throw std::invalid_argument(
      "CCSD_SOLVER_BASIS must be localized or canonical");
}

std::string_view ccsd_solver_basis_name(const CcsdSolverBasis basis) {
  switch (basis) {
  case CcsdSolverBasis::localized:
    return "localized";
  case CcsdSolverBasis::canonical:
    return "canonical RHF; amplitudes rotated to localized basis";
  }
  throw std::invalid_argument("unknown CCSD solver basis");
}

CcsdSolverBackend ccsd_solver_backend_from_environment() {
  const char* value = std::getenv("CCSD_SOLVER_BACKEND");
  if (value == nullptr || *value == '\0' ||
      std::string_view(value) == "spin-orbital")
    return CcsdSolverBackend::spin_orbital;
  if (std::string_view(value) == "legacy-spatial")
    return CcsdSolverBackend::legacy_spatial;
  throw std::invalid_argument(
      "CCSD_SOLVER_BACKEND must be spin-orbital or legacy-spatial");
}

std::string_view ccsd_solver_backend_name(const CcsdSolverBackend backend) {
  switch (backend) {
  case CcsdSolverBackend::spin_orbital:
    return "spin-orbital";
  case CcsdSolverBackend::legacy_spatial:
    return "legacy closed-shell spatial admission";
  }
  throw std::invalid_argument("unknown CCSD solver backend");
}

std::size_t triples_tile_size_from_environment() {
  const char* value = std::getenv("CCSD_TRIPLES_TILE_SIZE");
  if (value == nullptr || *value == '\0')
    return 16;
  const std::string text(value);
  std::size_t parsed = 0;
  const auto result = std::stoull(text, &parsed);
  if (parsed != text.size() || result == 0 ||
      result > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument(
        "CCSD_TRIPLES_TILE_SIZE must be a positive integer");
  return static_cast<std::size_t>(result);
}

localized_cc::CcsdAmplitudeBackend ccsd_backend_from_environment() {
  const char* value = std::getenv("CCSD_AMPLITUDE_BACKEND");
  if (value == nullptr || *value == '\0' || std::string_view(value) == "auto")
    return localized_cc::CcsdAmplitudeBackend::automatic;
  if (std::string_view(value) == "replicated")
    return localized_cc::CcsdAmplitudeBackend::replicated;
  if (std::string_view(value) == "ga")
    return localized_cc::CcsdAmplitudeBackend::global_arrays;
  throw std::invalid_argument(
      "CCSD_AMPLITUDE_BACKEND must be auto, replicated, or ga");
}

localized_cc::CcsdWabefBackend ccsd_wabef_backend_from_environment() {
  const char* value = std::getenv("CCSD_WABEF_BACKEND");
  if (value == nullptr || *value == '\0' ||
      std::string_view(value) == "tiled-cpu")
    return localized_cc::CcsdWabefBackend::tiled_cpu;
  if (std::string_view(value) == "cpu")
    return localized_cc::CcsdWabefBackend::cpu;
  if (std::string_view(value) == "cuda")
    return localized_cc::CcsdWabefBackend::cuda;
  throw std::invalid_argument(
      "CCSD_WABEF_BACKEND must be cpu, tiled-cpu, or cuda");
}

localized_cc::CcsdWabefPackingBackend
ccsd_wabef_packing_backend_from_environment() {
  const char* value = std::getenv("CCSD_WABEF_PACKING_BACKEND");
  if (value == nullptr || *value == '\0' ||
      std::string_view(value) == "cached-invariants")
    return localized_cc::CcsdWabefPackingBackend::cached_invariants;
  if (std::string_view(value) == "direct")
    return localized_cc::CcsdWabefPackingBackend::direct;
  throw std::invalid_argument(
      "CCSD_WABEF_PACKING_BACKEND must be direct or cached-invariants");
}

bool tiled_ccsd_doubles_from_environment() {
  const char* value = std::getenv("CCSD_DOUBLES_BACKEND");
  if (value == nullptr || *value == '\0' ||
      std::string_view(value) == "tiled-cpu")
    return true;
  if (std::string_view(value) == "scalar") return false;
  throw std::invalid_argument(
      "CCSD_DOUBLES_BACKEND must be scalar or tiled-cpu");
}

bool tiled_ccsd_intermediates_from_environment() {
  const char* value = std::getenv("CCSD_INTERMEDIATES_BACKEND");
  if (value == nullptr || *value == '\0' ||
      std::string_view(value) == "tiled-cpu")
    return true;
  if (std::string_view(value) == "scalar") return false;
  throw std::invalid_argument(
      "CCSD_INTERMEDIATES_BACKEND must be scalar or tiled-cpu");
}

bool incremental_ccsd_diis_gram_from_environment() {
  const char* value = std::getenv("CCSD_DIIS_GRAM_BACKEND");
  if (value == nullptr || *value == '\0' ||
      std::string_view(value) == "incremental")
    return true;
  if (std::string_view(value) == "recomputed") return false;
  throw std::invalid_argument(
      "CCSD_DIIS_GRAM_BACKEND must be recomputed or incremental");
}

bool root_replicated_ccsd_diis_from_environment() {
  const char* value = std::getenv("CCSD_REPLICATED_DIIS_EXECUTOR");
  if (value == nullptr || *value == '\0' ||
      std::string_view(value) == "root-broadcast")
    return true;
  if (std::string_view(value) == "all-ranks") return false;
  throw std::invalid_argument(
      "CCSD_REPLICATED_DIIS_EXECUTOR must be all-ranks or root-broadcast");
}

bool distributed_replicated_ccsd_energy_from_environment() {
  const char* value = std::getenv("CCSD_REPLICATED_ENERGY_EXECUTOR");
  if (value == nullptr || *value == '\0' ||
      std::string_view(value) == "distributed-openmp")
    return true;
  if (std::string_view(value) == "all-ranks-serial") return false;
  throw std::invalid_argument(
      "CCSD_REPLICATED_ENERGY_EXECUTOR must be all-ranks-serial or "
      "distributed-openmp");
}

localized_cc::CcsdResidualMetric ccsd_residual_metric_from_environment() {
  const char* value = std::getenv("CCSD_RESIDUAL_METRIC");
  if (value == nullptr || *value == '\0' || std::string_view(value) == "rms")
    return localized_cc::CcsdResidualMetric::rms;
  if (std::string_view(value) == "l2")
    return localized_cc::CcsdResidualMetric::l2;
  throw std::invalid_argument("CCSD_RESIDUAL_METRIC must be rms or l2");
}

bool boolean_from_environment(const char* name,
                              const bool default_value = false) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return default_value;
  const std::string_view text(value);
  if (text == "1" || text == "true" || text == "yes" || text == "on")
    return true;
  if (text == "0" || text == "false" || text == "no" || text == "off")
    return false;
  throw std::invalid_argument(std::string(name) +
                              " must be a boolean value");
}

void report_lmo_shell_support(
    const char* space,
    const modernqc::linalg::Matrix& coefficients,
    const modernqc::basis::BasisSet& basis) {
  if (coefficients.rows() != basis.number_of_aos ||
      coefficients.columns() == 0)
    throw std::invalid_argument(
        "LMO shell-support coefficient dimensions are inconsistent");
  constexpr std::array<double, 6> thresholds{
      0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4};
  const auto total_blocks = basis.shells.size() * coefficients.columns();
  const auto saved_flags = std::cout.flags();
  const auto saved_precision = std::cout.precision();
  for (const double threshold : thresholds) {
    std::size_t omitted_blocks = 0;
    long double omitted_squared_norm = 0.0L;
    for (const auto& shell : basis.shells) {
      if (shell.first_ao > coefficients.rows() ||
          shell.function_count > coefficients.rows() - shell.first_ao)
        throw std::invalid_argument(
            "LMO shell-support basis metadata are inconsistent");
      for (std::size_t orbital = 0; orbital < coefficients.columns();
           ++orbital) {
        long double squared_norm = 0.0L;
        for (std::size_t local_ao = 0;
             local_ao < shell.function_count; ++local_ao) {
          const double coefficient =
              coefficients(shell.first_ao + local_ao, orbital);
          if (!std::isfinite(coefficient))
            throw std::invalid_argument(
                "LMO shell-support coefficients must be finite");
          squared_norm += static_cast<long double>(coefficient) *
                          static_cast<long double>(coefficient);
        }
        const long double threshold_squared =
            static_cast<long double>(threshold) *
            static_cast<long double>(threshold);
        if (squared_norm <= threshold_squared) {
          ++omitted_blocks;
          omitted_squared_norm += squared_norm;
        }
      }
    }
    std::cout << "LMO shell support          " << space
              << "; raw_L2_threshold=" << std::scientific
              << std::setprecision(3) << threshold
              << "; retained_blocks=" << (total_blocks - omitted_blocks)
              << "/" << total_blocks
              << "; omitted_blocks=" << omitted_blocks
              << "; omitted_raw_coefficient_L2="
              << static_cast<double>(std::sqrt(omitted_squared_norm))
              << "\n";
  }
  std::cout.flags(saved_flags);
  std::cout.precision(saved_precision);
}

std::size_t mib_from_environment(const char* name,
                                 const std::size_t default_mib,
                                 const bool allow_zero) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0')
    return default_mib;
  const std::string text(value);
  std::size_t parsed = 0;
  const auto result = std::stoull(text, &parsed);
  if (parsed != text.size() || (!allow_zero && result == 0) ||
      result > std::numeric_limits<std::size_t>::max() / (1024U * 1024U))
    throw std::invalid_argument(std::string(name) +
                                " must be a valid MiB integer");
  return static_cast<std::size_t>(result);
}

int positive_int_from_environment(const char* name,
                                  const int default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0')
    return default_value;
  const std::string text(value);
  std::size_t parsed = 0;
  const auto result = std::stoull(text, &parsed);
  if (parsed != text.size() || result == 0 ||
      result > static_cast<unsigned long long>(
                   std::numeric_limits<int>::max()))
    throw std::invalid_argument(std::string(name) +
                                " must be a positive integer");
  return static_cast<int>(result);
}

std::size_t openmp_thread_count() {
#if defined(_OPENMP)
  return static_cast<std::size_t>(omp_get_max_threads());
#else
  return 1;
#endif
}

const char* triples_backend_name(
    const localized_cc::CanonicalTriplesBackend backend) {
  if (backend == localized_cc::CanonicalTriplesBackend::tce_cuda)
    return "TCE-style CUDA binary64 cuBLAS D/S tiles";
  if (backend == localized_cc::CanonicalTriplesBackend::cuda)
    return "CUDA fused D/S moment tiles";
  if (backend == localized_cc::CanonicalTriplesBackend::tiled_cpu)
    return "CPU bounded D/S moment tiles";
  return "CPU reference streamed triples";
}

} // namespace

int main(int argc, char** argv) {
  localized_cc::parallel::Runtime parallel_runtime(argc, argv);
  try {
    localized_cc::GlobalArraysRuntime global_arrays_runtime;
    if (argc < 2 || argc > 15) {
      std::cerr
          << "usage: localized_ccsd_t GEOMETRY.xyz [BASIS=sto-3g] "
             "[MEMORY_MiB=2048] [MODE=localized|canonical|both|plan] "
             "[OCCUPIED_TASK_GRAIN=1] "
             "[CORE=atomic] [CHECKPOINT=localized_ccsd_t.chk] "
             "[RESTART=auto|required|none] [LOCALIZER=pm|fm2|fm4] "
             "[ASSIGNMENT_THRESHOLD=0.8] [PARTITION_PREFIX=none] "
             "[CONTRIBUTION_THRESHOLD=0] [MAX_OMITTED_Eh=1e-6] "
             "[PROJECT_FROM_CHECKPOINT=none]\n";
      return 2;
    }
    const std::string basis_name = argc > 2 ? argv[2] : "sto-3g";
    const std::size_t memory_mib = argc > 3 ? std::stoull(argv[3]) : 2048;
    const auto mode = parse_mode(argc > 4 ? argv[4] : "localized");
    const bool run_canonical =
        mode == RunMode::canonical || mode == RunMode::both;
    const bool run_localized =
        mode == RunMode::localized || mode == RunMode::both;
    const std::size_t task_grain = argc > 5 ? std::stoull(argv[5]) : 1;
    const auto core_mode = parse_core_mode(argc > 6 ? argv[6] : "atomic");
    const std::filesystem::path checkpoint_path =
        argc > 7 ? argv[7] : "localized_ccsd_t.chk";
    const auto ccsd_stage_path = ccsd_stage_checkpoint_path(checkpoint_path);
    const auto restart_mode =
        parse_restart_mode(argc > 8 ? argv[8] : "auto");
    const auto localizer = parse_localizer(argc > 9 ? argv[9] : "pm");
    const double assignment_threshold =
        argc > 10 ? std::stod(argv[10]) : 0.8;
    const std::string partition_prefix = argc > 11 ? argv[11] : "none";
    const localized_cc::PartitionScreeningOptions partition_screening{
        argc > 12 ? std::stod(argv[12]) : 0.0,
        argc > 13 ? std::stod(argv[13]) : 1.0e-6};
    const std::filesystem::path projection_source_path =
        argc > 14 ? argv[14] : "none";
    const bool projection_requested = projection_source_path != "none";
    const auto communicator = parallel_runtime.world();
    const auto triples_backend = triples_backend_from_environment();
    const auto triples_tile_size = triples_tile_size_from_environment();
    const auto localized_triples_representation =
        localized_triples_representation_from_environment();
    const auto execution_stage = execution_stage_from_environment();
    if (execution_stage == ExecutionStage::ccsd_only &&
        (!run_localized || run_canonical))
      throw std::invalid_argument(
          "CCSD_EXECUTION_STAGE=ccsd-only requires MODE=localized");
    const auto laplace_triples_points = static_cast<std::size_t>(
        positive_int_from_environment("CCSD_LAPLACE_TRIPLES_POINTS", 7));
    const auto laplace_triples_scheme =
        laplace_triples_quadrature_scheme_from_environment();
    const auto laplace_triples_error_tolerance =
        positive_double_from_environment(
            "CCSD_LAPLACE_TRIPLES_ERROR_TOLERANCE", 1.0e-9);
    // Explicit zero requests an unscreened, fresh-point comparison arm.
    const bool compare_triples_screen =
        std::getenv("CCSD_LAPLACE_TRIPLES_BLOCK_SCREEN_BUDGET") != nullptr;
    const double triples_screen_budget = nonnegative_double_from_environment(
        "CCSD_LAPLACE_TRIPLES_BLOCK_SCREEN_BUDGET", 0.0);
    if (triples_screen_budget > 8.0e-7)
      throw std::invalid_argument("triples screening budget must be <=8e-7 Eh; reserve the remaining 1e-6 budget for quadrature/numerics");
    if (triples_screen_budget>0 &&
        triples_screen_budget+laplace_triples_error_tolerance>1.e-6)
      throw std::invalid_argument("triples screening plus quadrature budgets exceed 1e-6 Eh");
    const auto selected_maximum_monomers =
        selected_maximum_monomers_from_environment();
    const auto selected_ccsd_executor =
        selected_ccsd_executor_from_environment();
    const auto selected_integral_backend =
        selected_integral_backend_from_environment();
    const auto localized_hamiltonian_backend =
        localized_hamiltonian_backend_from_environment();
    const auto ccsd_solver_basis = ccsd_solver_basis_from_environment();
    const bool lmp2_domain_audit =
        boolean_from_environment("CCSD_LMP2_DOMAIN_AUDIT_ONLY");
    const bool lmp2_preintegral = boolean_from_environment("CCSD_LMP2_PREINTEGRAL");
    const bool lmp2_adaptive = boolean_from_environment("CCSD_LMP2_ADAPTIVE");
    if(lmp2_adaptive&&(!lmp2_domain_audit||lmp2_preintegral))
      throw std::invalid_argument("adaptive MP2 requires audit-only and cannot combine with preintegral pilot");
    if(lmp2_preintegral&&!lmp2_domain_audit)
      throw std::invalid_argument("preintegral LMP2 currently requires CCSD_LMP2_DOMAIN_AUDIT_ONLY=1; no production mask is admitted");
    if (lmp2_domain_audit &&
        (mode != RunMode::localized || localizer != Localizer::pm ||
         ccsd_solver_basis != CcsdSolverBasis::localized ||
         selected_maximum_monomers != std::numeric_limits<std::size_t>::max() ||
         selected_ccsd_executor != SelectedCcsdExecutor::dense_oracle ||
         selected_integral_backend != SelectedIntegralBackend::dense_oracle ||
         localized_hamiltonian_backend != LocalizedHamiltonianBackend::ao_semidirect ||
         execution_stage != ExecutionStage::ccsd_only || projection_requested))
      throw std::invalid_argument(
          "LMP2 domain audit requires full PM, localized solver basis, dense-oracle "
          "four-center Hamiltonian, ccsd-only stage, and no basis projection; "
          "CCSD itself is not executed");
    if (lmp2_domain_audit && std::filesystem::exists(
            checkpoint_path.string()+".lmp2-domains"))
      throw std::invalid_argument("LMP2 audit output already exists; use a new checkpoint copy/path");
    const auto ccsd_solver_backend = ccsd_solver_backend_from_environment();
    const bool native_spatial_pairs =
        boolean_from_environment("CCSD_NATIVE_SPATIAL_PAIRS");
    const bool compact_spatial_doubles =
        boolean_from_environment("CCSD_NATIVE_SPATIAL_COMPACT_DOUBLES");
    const bool audit_compact_spatial_doubles =
        boolean_from_environment("CCSD_NATIVE_SPATIAL_COMPACT_AUDIT");
    if (audit_compact_spatial_doubles && !compact_spatial_doubles)
      throw std::invalid_argument("CCSD_NATIVE_SPATIAL_COMPACT_AUDIT requires compact spatial doubles");
    if (compact_spatial_doubles && !native_spatial_pairs)
      throw std::invalid_argument("CCSD_NATIVE_SPATIAL_COMPACT_DOUBLES requires native spatial pairs");
    if (native_spatial_pairs && ccsd_solver_backend != CcsdSolverBackend::legacy_spatial)
      throw std::invalid_argument("CCSD_NATIVE_SPATIAL_PAIRS requires legacy-spatial");
    const auto ccsd_backend = ccsd_backend_from_environment();
    // The closed-shell spatial solver is also available on the localized
    // basis.  Its residual keeps the (1 - delta) Fock terms in F_ae and F_mi,
    // so a non-diagonal localized Fock block is exact; the previous
    // canonical-only restriction reflected where the backend had been wired
    // rather than a limit of the equations.
    //
    // Two combinations remain genuinely unsupported.  The spatial solver is
    // already the exact closed-shell manifold and rejects a projection mask,
    // so it cannot express a monomer-truncated manifold; and it does not
    // consume the selected AO-direct integral Hamiltonian.
    if (ccsd_solver_backend == CcsdSolverBackend::legacy_spatial &&
        ccsd_solver_basis != CcsdSolverBasis::canonical &&
        ccsd_solver_basis != CcsdSolverBasis::localized)
      throw std::invalid_argument(
          "CCSD_SOLVER_BACKEND=legacy-spatial requires "
          "CCSD_SOLVER_BASIS=canonical or localized");
    if (ccsd_solver_backend == CcsdSolverBackend::legacy_spatial &&
        ccsd_solver_basis == CcsdSolverBasis::localized &&
        selected_maximum_monomers != std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument(
          "CCSD_SOLVER_BACKEND=legacy-spatial is the exact closed-shell "
          "manifold and cannot represent a monomer-truncated selection; "
          "set CCSD_SELECTED_MAXIMUM_MONOMERS=full");
    if (localized_hamiltonian_backend ==
            LocalizedHamiltonianBackend::canonical_rotation &&
        ccsd_solver_basis != CcsdSolverBasis::canonical)
      throw std::invalid_argument(
          "CCSD_LOCALIZED_HAMILTONIAN_BACKEND=canonical-rotation requires "
          "CCSD_SOLVER_BASIS=canonical");
    const auto ccsd_wabef_backend = ccsd_wabef_backend_from_environment();
    const auto ccsd_wabef_packing_backend =
        ccsd_wabef_packing_backend_from_environment();
    const bool use_wabef_pair_symmetry =
        boolean_from_environment("CCSD_WABEF_PAIR_SYMMETRY");
    const bool use_pm_task_symmetry =
        boolean_from_environment("CCSD_PM_TASK_SYMMETRY");
    const bool pm_residual_orbits = boolean_from_environment("CCSD_PM_RESIDUAL_ORBITS");
    const bool pm_propagation_orbits = boolean_from_environment("CCSD_PM_PROPAGATION_ORBITS");
    const bool pm_residual_compare = boolean_from_environment("CCSD_PM_RESIDUAL_COMPARE");
    if ((pm_residual_orbits || pm_residual_compare) &&
        (!use_pm_task_symmetry || !native_spatial_pairs))
      throw std::invalid_argument("PM residual orbits/comparison require PM task symmetry and native spatial pairs");
    if (pm_propagation_orbits && (!use_pm_task_symmetry || compare_triples_screen))
      throw std::invalid_argument("PM propagation orbits require unscreened PM task symmetry");
    const bool use_adapted_pm = use_pm_task_symmetry ||
        boolean_from_environment("CCSD_PM_ADAPTED_LOCALIZATION");
    if (use_adapted_pm && localizer != Localizer::pm)
      throw std::invalid_argument("cyclic symmetry-adapted localization requires PM");
    if (use_pm_task_symmetry &&
        (!run_localized || run_canonical ||
         ccsd_solver_backend != CcsdSolverBackend::legacy_spatial ||
         ccsd_solver_basis != CcsdSolverBasis::localized ||
         selected_maximum_monomers != std::numeric_limits<std::size_t>::max() ||
         localized_triples_representation != LocalizedTriplesRepresentation::laplace))
      throw std::invalid_argument("CCSD_PM_TASK_SYMMETRY requires full legacy-spatial CCSD directly in the localized basis and Laplace triples");
    const auto use_tiled_ccsd_intermediates =
        tiled_ccsd_intermediates_from_environment();
    const auto use_tiled_ccsd_doubles =
        tiled_ccsd_doubles_from_environment();
    const bool use_incremental_ccsd_diis_gram =
        incremental_ccsd_diis_gram_from_environment();
    const bool use_root_replicated_ccsd_diis =
        root_replicated_ccsd_diis_from_environment();
    const bool use_distributed_replicated_ccsd_energy =
        distributed_replicated_ccsd_energy_from_environment();
    const double ccsd_convergence_threshold =
        positive_double_from_environment("CCSD_CONVERGENCE_THRESHOLD",
                                         1.0e-8);
    const char* spatial_diis_env = std::getenv("CCSD_SPATIAL_DIIS_METRIC");
    const std::string spatial_diis_name =
        spatial_diis_env && *spatial_diis_env ? spatial_diis_env : "raw";
    const auto spatial_diis_error_metric =
        localized_cc::spatial_diis_metric(spatial_diis_name);
    const auto ccsd_diis_size = static_cast<std::size_t>(
        positive_int_from_environment("CCSD_DIIS_SIZE", 8));
    if (ccsd_diis_size < 2 || ccsd_diis_size > 32)
      throw std::invalid_argument("CCSD_DIIS_SIZE must be between 2 and 32");
    if (spatial_diis_error_metric != localized_cc::SpatialDiisMetric::raw &&
        ccsd_solver_backend != CcsdSolverBackend::legacy_spatial)
      throw std::invalid_argument("weighted spatial DIIS requires legacy-spatial");
    const auto ccsd_residual_metric =
        ccsd_residual_metric_from_environment();
    const bool use_exact_spin_manifold =
        boolean_from_environment("CCSD_EXACT_SPIN_MANIFOLD");
    const bool report_lmo_support =
        boolean_from_environment("CCSD_LMO_SUPPORT_DIAGNOSTICS");
    const char* ccsd_term_audit_csv_environment =
        std::getenv("CCSD_ENERGY_TERM_AUDIT_CSV");
    const std::optional<std::filesystem::path> ccsd_term_audit_csv =
        ccsd_term_audit_csv_environment != nullptr &&
                *ccsd_term_audit_csv_environment != '\0'
            ? std::optional<std::filesystem::path>{
                  ccsd_term_audit_csv_environment}
            : std::nullopt;
    const double ccsd_term_audit_tolerance =
        positive_double_from_environment(
            "CCSD_ENERGY_TERM_AUDIT_TOLERANCE", 1.0e-6);
    if (use_tiled_ccsd_doubles &&
        selected_ccsd_executor == SelectedCcsdExecutor::block_sparse)
      throw std::invalid_argument(
          "CCSD_DOUBLES_BACKEND=tiled-cpu currently requires the "
          "selected-kernel or dense-oracle CCSD executor");
    if (use_tiled_ccsd_intermediates &&
        selected_ccsd_executor == SelectedCcsdExecutor::block_sparse)
      throw std::invalid_argument(
          "CCSD_INTERMEDIATES_BACKEND=tiled-cpu currently requires the "
          "selected-kernel or dense-oracle CCSD executor");
    if (use_exact_spin_manifold &&
        selected_ccsd_executor == SelectedCcsdExecutor::block_sparse)
      throw std::invalid_argument(
          "CCSD_EXACT_SPIN_MANIFOLD currently requires the replicated "
          "selected-kernel or dense-oracle CCSD executor");
    if (selected_integral_backend ==
            SelectedIntegralBackend::ao_direct_selected &&
        selected_ccsd_executor == SelectedCcsdExecutor::dense_oracle)
      throw std::invalid_argument(
          "CCSD_SELECTED_INTEGRAL_BACKEND=ao-direct-selected requires "
          "CCSD_SELECTED_EXECUTOR=selected-kernel or sparse-localized");
    if (selected_integral_backend ==
            SelectedIntegralBackend::ao_direct_selected &&
        execution_stage != ExecutionStage::ccsd_only)
      throw std::invalid_argument(
          "the AO-direct selected-integral admission gate currently "
          "supports CCSD_EXECUTION_STAGE=ccsd-only; selected Laplace "
          "triples must pass its separate fail-closed integral gate before "
          "a full CCSD(T) run is enabled");
    const auto replicated_t2_max_mib =
        mib_from_environment("CCSD_REPLICATED_T2_MAX_MIB", 512, true);
    const auto wabef_tile_mib =
        mib_from_environment("CCSD_WABEF_TILE_MIB", 512, false);
    const auto wabef_invariant_cache_mib =
        mib_from_environment("CCSD_WABEF_INVARIANT_CACHE_MIB", 4096, false);
    const auto selected_eri_panel_cache_mib =
        mib_from_environment("CCSD_SELECTED_ERI_PANEL_CACHE_MIB", 512, true);
    const auto semidirect_slice_cap_mib =
        mib_from_environment("CCSD_SEMIDIRECT_SLICE_CAP_MIB", 512, false);
    const auto ccsd_max_iterations =
        positive_int_from_environment("CCSD_MAX_ITERATIONS", 200);
    const auto occupied_trust_localization_max_iterations =
        positive_int_from_environment(
            "CCSD_OCCUPIED_LOCALIZATION_MAX_ITERATIONS",
            localizer == Localizer::fm4 ? 1200 : 300);
    const auto virtual_trust_localization_max_iterations =
        positive_int_from_environment(
            "CCSD_VIRTUAL_LOCALIZATION_MAX_ITERATIONS", 1200);
    const auto occupied_pm_maximum_sweeps =
        positive_int_from_environment("CCSD_PM_OCCUPIED_MAX_SWEEPS", 100);
    const auto virtual_pm_maximum_sweeps =
        positive_int_from_environment("CCSD_PM_VIRTUAL_MAX_SWEEPS", 100);

    const auto geometry = read_geometry(argv[1]);
    std::vector<int> pm_water_cycle;
    if (use_adapted_pm) {
      std::vector<std::array<double,3>> positions;
      std::vector<int> numbers;
      for (const auto& atom:geometry.atoms) {
        positions.push_back(atom.position_bohr);
        numbers.push_back(atom.atomic_number);
      }
      pm_water_cycle=localized_cc::find_cyclic_water_cycle(positions,numbers);
    }
    const auto& atoms = geometry.atoms;
    const modernqc::molecule::Molecule molecule(atoms, 0, 1);
    modernqc::integrals::Libint2IntegralProvider integrals(molecule, basis_name,
                                                           true);
    std::size_t frozen_core_count = 0;
    if (core_mode == CoreMode::atomic)
      for (const auto& atom : atoms)
        if (atom.atomic_number == 8)
          ++frozen_core_count;
    const std::size_t occupied_count = molecule.occupied_orbitals_rhf();
    const std::size_t ao_count = integrals.basis_metadata().number_of_aos;
    if (occupied_count > ao_count || frozen_core_count > occupied_count)
      throw std::runtime_error("basis and occupied-orbital counts are inconsistent");
    const std::size_t virtual_count = ao_count - occupied_count;
    const long double spin_virtuals =
        2.0L * static_cast<long double>(virtual_count);
    const long double eliminated_wabef_gib =
        spin_virtuals * spin_virtuals * spin_virtuals * spin_virtuals *
        static_cast<long double>(sizeof(double)) /
        (1024.0L * 1024.0L * 1024.0L);
    const long double streamed_wabef_mib =
        spin_virtuals * spin_virtuals *
        static_cast<long double>(sizeof(double)) / (1024.0L * 1024.0L);
    if (mode == RunMode::plan) {
      const long double ao = static_cast<long double>(ao_count);
      const long double ao_eri_gib =
          ao * ao * ao * ao * static_cast<long double>(sizeof(double)) /
          (1024.0L * 1024.0L * 1024.0L);
      const long double working_spatial_orbitals =
          static_cast<long double>(occupied_count - frozen_core_count +
                                   virtual_count);
      const long double spatial_mo_eri_gib =
          working_spatial_orbitals * working_spatial_orbitals *
          working_spatial_orbitals * working_spatial_orbitals *
          static_cast<long double>(sizeof(double)) /
          (1024.0L * 1024.0L * 1024.0L);
      const long double spin_reference_mo_eri_gib =
          16.0L * spatial_mo_eri_gib;
      const auto symmetric_eri_gib = [](const long double orbitals) {
        const long double pairs = orbitals * (orbitals + 1.0L) / 2.0L;
        return pairs * (pairs + 1.0L) / 2.0L *
               static_cast<long double>(sizeof(double)) /
               (1024.0L * 1024.0L * 1024.0L);
      };
      const long double stored_spatial_mo_eri_gib =
          symmetric_eri_gib(working_spatial_orbitals);
      const long double full_mo_eri_gib = symmetric_eri_gib(ao);
      const long double semidirect_work_gib =
          3.0L * ao * ao * ao *
          static_cast<long double>(sizeof(double)) /
          (1024.0L * 1024.0L * 1024.0L);
      const long double semidirect_fixed_gib =
          full_mo_eri_gib + stored_spatial_mo_eri_gib +
          semidirect_work_gib;
      const localized_cc::OccupiedPairLayout spatial_t2_layout(
          occupied_count - frozen_core_count, virtual_count);
      if (occupied_count - frozen_core_count >
              std::numeric_limits<std::size_t>::max() / 2 ||
          virtual_count > std::numeric_limits<std::size_t>::max() / 2)
        throw std::overflow_error(
            "spin-orbital planning dimensions overflow size_t");
      const localized_cc::OccupiedPairLayout transitional_spin_t2_layout(
          2 * (occupied_count - frozen_core_count), 2 * virtual_count);
      const auto balanced_t2_bytes =
          ((spatial_t2_layout.element_count() + communicator.size - 1) /
           communicator.size) *
          sizeof(double);
      const auto diis_capacity = ccsd_diis_size;
      const long double transitional_diis_global_bytes =
          2.0L * static_cast<long double>(diis_capacity) *
          static_cast<long double>(transitional_spin_t2_layout.global_bytes());
      const long double transitional_diis_balanced_bytes =
          transitional_diis_global_bytes /
          static_cast<long double>(communicator.size);
      constexpr std::size_t solver_pair_arrays = 4;
      constexpr std::size_t intermediate_pair_arrays = 3;
      const long double solver_pair_global_bytes =
          static_cast<long double>(solver_pair_arrays) *
          static_cast<long double>(transitional_spin_t2_layout.global_bytes());
      const long double solver_pair_balanced_bytes =
          solver_pair_global_bytes /
          static_cast<long double>(communicator.size);
      const long double intermediate_pair_global_bytes =
          static_cast<long double>(intermediate_pair_arrays) *
          static_cast<long double>(transitional_spin_t2_layout.global_bytes());
      const long double intermediate_pair_balanced_bytes =
          intermediate_pair_global_bytes /
          static_cast<long double>(communicator.size);
      const long double wmnij_global_bytes =
          static_cast<long double>(
              transitional_spin_t2_layout.occupied()) *
          static_cast<long double>(
              transitional_spin_t2_layout.occupied()) *
          static_cast<long double>(
              transitional_spin_t2_layout.occupied()) *
          static_cast<long double>(
              transitional_spin_t2_layout.occupied()) *
          static_cast<long double>(sizeof(double));
      const long double wmnij_balanced_bytes =
          wmnij_global_bytes / static_cast<long double>(communicator.size);
      const long double bounded_pair_cache_bytes =
          (4.0L * static_cast<long double>(
                      transitional_spin_t2_layout.occupied()) +
           2.0L) *
          static_cast<long double>(transitional_spin_t2_layout.virtuals()) *
          static_cast<long double>(transitional_spin_t2_layout.virtuals()) *
          static_cast<long double>(sizeof(double)) +
          static_cast<long double>(
              transitional_spin_t2_layout.occupied()) *
          static_cast<long double>(
              transitional_spin_t2_layout.occupied()) *
          static_cast<long double>(sizeof(double));
      const long double spin_occupied =
          static_cast<long double>(transitional_spin_t2_layout.occupied());
      const long double spin_virtual =
          static_cast<long double>(transitional_spin_t2_layout.virtuals());
      const long double small_intermediate_bytes =
          (spin_virtual * spin_virtual + spin_occupied * spin_occupied +
           spin_occupied * spin_virtual) *
          static_cast<long double>(sizeof(double));
      const auto rank0_owned_virtual_pairs =
          (transitional_spin_t2_layout.virtuals() *
               transitional_spin_t2_layout.virtuals() +
           communicator.size - 1) /
          communicator.size;
      const long double rank0_wabef_accumulator_bytes =
          static_cast<long double>(rank0_owned_virtual_pairs) *
          spin_occupied * spin_occupied *
          static_cast<long double>(sizeof(double));
      const auto dense_localized_triples_memory =
          localized_cc::estimate_triples_memory_mib(
              transitional_spin_t2_layout.occupied(),
              transitional_spin_t2_layout.virtuals(), 6);
      if (communicator.rank == 0)
        std::cout << "MPI ranks                  " << communicator.size << "\n"
                  << "scientific energy backend "
                  << "C++ spin equations / zero-copy spatial ERIs"
                  << "\n"
                  << "geometry point group      " << geometry.point_group << "\n"
                << "atoms                     " << atoms.size() << "\n"
                << "water monomers            " << atoms.size() / 3 << "\n"
                << "spherical basis           yes\n"
                << "AO basis functions        " << ao_count << "\n"
                << "electrons                 " << molecule.electron_count() << "\n"
                << "occupied spatial          " << occupied_count << "\n"
                << "frozen core spatial       " << frozen_core_count << "\n"
                << "active occupied spatial   "
                << occupied_count - frozen_core_count << "\n"
                << "virtual spatial           " << virtual_count << "\n"
                << "occupied localization max "
                << (localizer == Localizer::pm
                        ? occupied_pm_maximum_sweeps
                        : occupied_trust_localization_max_iterations)
                << (localizer == Localizer::pm ? " macros\n"
                                                : " iterations\n")
                << "virtual localization max  "
                << (localizer == Localizer::pm
                        ? virtual_pm_maximum_sweeps
                        : virtual_trust_localization_max_iterations)
                << (localizer == Localizer::pm ? " macros\n"
                                                : " iterations\n")
                << "PM optimizer              "
                << (localizer == Localizer::pm ? "PySCF 2.11 CIAH"
                                                : "not applicable")
                << "\n"
                << "CCSD maximum iterations   " << ccsd_max_iterations
                << "\n"
                << "Global Arrays T2 backend "
                << (global_arrays_runtime.active() ? "enabled" : "disabled")
                << "\n"
                << std::fixed << std::setprecision(6)
                << "eliminated full AO ERI    "
                << static_cast<double>(ao_eri_gib) << " GiB (estimate)\n"
                << "spatial MO ERI logical    "
                << static_cast<double>(spatial_mo_eri_gib) << " GiB\n"
                << "spatial MO ERI stored     "
                << static_cast<double>(stored_spatial_mo_eri_gib)
                << " GiB/rank (exact 8-fold symmetry)\n"
                << "semidirect full MO ERI   "
                << static_cast<double>(full_mo_eri_gib) << " GiB/rank\n"
                << "semidirect work buffers  "
                << static_cast<double>(semidirect_work_gib) << " GiB/rank\n"
                << "semidirect fixed tensors "
                << static_cast<double>(semidirect_fixed_gib) << " GiB/rank"
                << " (excludes bounded first-index slice)\n"
                << "semicanonical work/rank  "
                << static_cast<double>(2.0L * spatial_mo_eri_gib)
                << " GiB (peak additional ERI buffers)\n"
                << "spin MO ERI reference     "
                << static_cast<double>(spin_reference_mo_eri_gib)
                << " GiB (transitional)\n"
                << "spatial T2 global         "
                << static_cast<double>(spatial_t2_layout.global_bytes()) /
                       (1024.0 * 1024.0)
                << " MiB\n"
                << "spatial T2 balanced/rank  "
                << static_cast<double>(balanced_t2_bytes) /
                       (1024.0 * 1024.0)
                << " MiB (estimate)\n"
                << "transitional spin T2     "
                << static_cast<double>(
                       transitional_spin_t2_layout.global_bytes()) /
                       (1024.0 * 1024.0)
                << " MiB (current iterate)\n"
                << "GA DIIS T2 max global    "
                << static_cast<double>(transitional_diis_global_bytes) /
                       (1024.0 * 1024.0)
                << " MiB (vector+error history)\n"
                << "GA DIIS T2 balanced/rank "
                << static_cast<double>(transitional_diis_balanced_bytes) /
                       (1024.0 * 1024.0)
                << " MiB (estimate)\n"
                << "GA solver pairs global   "
                << static_cast<double>(solver_pair_global_bytes) /
                       (1024.0 * 1024.0)
                << " MiB (current+raw+candidate+R2)\n"
                << "GA solver pairs/rank     "
                << static_cast<double>(solver_pair_balanced_bytes) /
                       (1024.0 * 1024.0)
                << " MiB (balanced estimate)\n"
                << "GA intermediates global  "
                << static_cast<double>(intermediate_pair_global_bytes) /
                       (1024.0 * 1024.0)
                << " MiB (tau+tilde-tau+W_mbej)\n"
                << "GA intermediates/rank    "
                << static_cast<double>(intermediate_pair_balanced_bytes) /
                       (1024.0 * 1024.0)
                << " MiB (balanced estimate)\n"
                << "GA W_mnij global         "
                << static_cast<double>(wmnij_global_bytes) /
                       (1024.0 * 1024.0)
                << " MiB\n"
                << "GA W_mnij/rank           "
                << static_cast<double>(wmnij_balanced_bytes) /
                       (1024.0 * 1024.0)
                << " MiB (balanced estimate)\n"
                << "bounded pair cache/rank  "
                << static_cast<double>(bounded_pair_cache_bytes) /
                       (1024.0 * 1024.0)
                << " MiB (maximum explicit cache)\n"
                << "small work replicated/rank "
                << static_cast<double>(small_intermediate_bytes) /
                       (1024.0 * 1024.0)
                << " MiB (F intermediates only)\n"
                << "W_abef accumulator rank0 "
                << static_cast<double>(rank0_wabef_accumulator_bytes) /
                       (1024.0 * 1024.0)
                << " MiB (distributed by external pair)\n"
                << "eliminated W_abef/rank   "
                << static_cast<double>(eliminated_wabef_gib)
                << " GiB (former replicated tensor)\n"
                << "streamed W_abef block    "
                << static_cast<double>(streamed_wabef_mib)
                << " MiB/task\n"
                << "dense localized T3 tensor "
                << dense_localized_triples_memory.at(
                       "triples_response")
                << " MiB (one spin-orbital tensor)\n"
                << "dense localized triples   "
                << dense_localized_triples_memory.at("total")
                << " MiB/rank0 (retained exact reference only)\n"
                << "production pair ownership "
                   "GA solver and intermediate pair tensors\n"
                << "basis-projection source  "
                << (projection_requested ? projection_source_path.string()
                                         : "none")
                << "\n";
      return 0;
    }
    auto fingerprint = calculation_fingerprint(geometry, basis_name, core_mode);
    if (use_adapted_pm) fingerprint += ";pm_symmetry_adapted=v1";
    // Preserve W4 checkpoint compatibility; never reuse ordinary-PM W3 states
    // as symmetry-adapted states or import their already-partitioned triples.
    if (use_adapted_pm && pm_water_cycle.size()==3)
      fingerprint += ";pm_cyclic_order=3";
    if (selected_maximum_monomers !=
        std::numeric_limits<std::size_t>::max())
      fingerprint += "|selected_maximum_monomers=" +
                     std::to_string(selected_maximum_monomers);
    if (selected_ccsd_executor == SelectedCcsdExecutor::block_sparse)
      fingerprint += "|selected_executor=block_sparse";
    else if (selected_ccsd_executor == SelectedCcsdExecutor::selected_kernel)
      fingerprint += "|selected_executor=selected_kernel";
    if (selected_integral_backend ==
        SelectedIntegralBackend::ao_direct_selected)
      fingerprint += "|selected_integrals=ao_direct_selected";
    if (use_exact_spin_manifold)
      fingerprint += "|exact_spin_manifold=1";
    localized_cc::CalculationCheckpoint checkpoint;
    checkpoint.fingerprint = fingerprint;
    const bool checkpoint_exists = std::filesystem::exists(checkpoint_path);
    if (restart_mode == RestartMode::required && !checkpoint_exists)
      throw std::runtime_error("required checkpoint does not exist: " +
                               checkpoint_path.string());
    if (restart_mode == RestartMode::none && checkpoint_exists)
      throw std::runtime_error(
          "RESTART=none refuses to overwrite an existing checkpoint; use a "
          "new path or remove the old checkpoint explicitly");
    if (restart_mode == RestartMode::none &&
        std::filesystem::exists(ccsd_stage_path))
      throw std::runtime_error(
          "RESTART=none refuses to overwrite an existing CCSD-stage "
          "checkpoint; use a new path or remove the old checkpoint "
          "explicitly");
    if (checkpoint_exists && restart_mode != RestartMode::none)
      checkpoint = localized_cc::read_checkpoint(checkpoint_path, fingerprint);
    if (boolean_from_environment("CCSD_REQUIRE_CONVERGED_RESTART") ||
        boolean_from_environment("CCSD_W6_INVERSION_AUDIT_ONLY")) {
      if (mode!=RunMode::localized || !checkpoint.rhf || !checkpoint.localization ||
          !checkpoint.localized_ccsd || !checkpoint.localized_ccsd->converged ||
          boolean_from_environment("CCSD_REFINE_SAVED_CCSD"))
        throw std::runtime_error("triples-only restart requires saved RHF, PM and converged localized CCSD; refinement disabled");
      // Reject inadequate convergence before any Hamiltonian or triples work.
      const auto& saved=*checkpoint.localized_ccsd;
      localized_cc::validate_converged_ccsd_checkpoint(saved,true,
          saved.t1.dimensions()[0],saved.t1.dimensions()[1],
          ccsd_convergence_threshold,ccsd_convergence_threshold,ccsd_residual_metric);
    }
    const auto persist_checkpoint = [&]() {
      localized_cc::parallel::barrier();
      if (communicator.rank == 0)
        localized_cc::write_checkpoint_transactional(checkpoint_path,
                                                     checkpoint);
      localized_cc::parallel::barrier();
    };
    const auto persist_ccsd_stage_checkpoint = [&]() {
      localized_cc::parallel::barrier();
      if (communicator.rank == 0) {
        auto dense_triples = std::move(checkpoint.localized_triples);
        auto streamed_triples =
            std::move(checkpoint.localized_streamed_triples);
        auto laplace_triples =
            std::move(checkpoint.localized_laplace_triples);
        checkpoint.localized_triples.reset();
        checkpoint.localized_streamed_triples.reset();
        checkpoint.localized_laplace_triples.reset();
        try {
          localized_cc::write_checkpoint_transactional(ccsd_stage_path,
                                                       checkpoint);
        } catch (...) {
          checkpoint.localized_triples = std::move(dense_triples);
          checkpoint.localized_streamed_triples =
              std::move(streamed_triples);
          checkpoint.localized_laplace_triples =
              std::move(laplace_triples);
          throw;
        }
        checkpoint.localized_triples = std::move(dense_triples);
        checkpoint.localized_streamed_triples = std::move(streamed_triples);
        checkpoint.localized_laplace_triples = std::move(laplace_triples);
      }
      localized_cc::parallel::barrier();
    };
    const auto report_progress = [&](const std::string& message) {
      if (communicator.rank == 0) {
        std::cout << "progress                   " << message << '\n';
        std::cout.flush();
      }
    };
    if (use_adapted_pm)
      report_progress("cyclic geometry verified before RHF; waters="+
          std::to_string(pm_water_cycle.size())+"; full atomic map; tolerance_bohr=1e-9");
    {
#if defined(MODERNQC_HAS_MPI)
      constexpr bool molecular_mpi_enabled = true;
#else
      constexpr bool molecular_mpi_enabled = false;
#endif
#if defined(MODERNQC_HAS_OPENMP)
      constexpr bool molecular_openmp_enabled = true;
#else
      constexpr bool molecular_openmp_enabled = false;
#endif
      if (communicator.size > 1 && !molecular_mpi_enabled) {
        throw std::runtime_error(
            "multi-rank molecular execution requires the ModernQC RHF "
            "sources to be compiled with MPI support");
      }
      std::ostringstream features;
      features << "molecular backend features; MPI="
               << (molecular_mpi_enabled ? "enabled" : "disabled")
               << "; OpenMP="
               << (molecular_openmp_enabled ? "enabled" : "disabled");
      report_progress(features.str());
    }
    const auto configure_ccsd_runtime =
        [&](localized_cc::CcsdOptions& options, const std::string& label) {
          options.energy_tolerance = ccsd_convergence_threshold;
          options.residual_tolerance = ccsd_convergence_threshold;
          options.residual_metric = ccsd_residual_metric;
          options.diis_size = ccsd_diis_size;
          options.spatial_diis_metric = spatial_diis_error_metric;
          report_progress("CCSD DIIS metric=" + spatial_diis_name +
                          "; history_capacity=" + std::to_string(ccsd_diis_size) +
                          "; convergence uses unchanged unweighted residuals");
          options.amplitude_backend = ccsd_backend;
          options.replicated_t2_max_bytes =
              replicated_t2_max_mib * 1024U * 1024U;
          options.wabef_tile_max_bytes = wabef_tile_mib * 1024U * 1024U;
          options.wabef_backend = ccsd_wabef_backend;
          options.wabef_packing_backend = ccsd_wabef_packing_backend;
          options.use_wabef_pair_symmetry = use_wabef_pair_symmetry;
          options.wabef_invariant_cache_max_bytes =
              wabef_invariant_cache_mib * 1024U * 1024U;
          options.selected_eri_panel_cache_max_bytes =
              selected_eri_panel_cache_mib * 1024U * 1024U;
          options.use_tiled_intermediates = use_tiled_ccsd_intermediates;
          options.use_tiled_doubles = use_tiled_ccsd_doubles;
          options.use_tiled_spatial_pairs = native_spatial_pairs;
          options.use_compact_spatial_doubles = compact_spatial_doubles;
          options.audit_compact_spatial_doubles = audit_compact_spatial_doubles;
          options.use_pm_residual_orbits = pm_residual_orbits;
          options.use_direct_spatial_singles = native_spatial_pairs;
          options.use_direct_spatial_doubles = native_spatial_pairs;
          if (native_spatial_pairs)
            report_progress("CCSD native spatial pairs enabled; iterative spin T1/T2 expansion=none; spatial BLAS panels; admission pending");
          if (compact_spatial_doubles)
            report_progress("CCSD compact spatial doubles enabled; Fock corrections cached per evaluation; mixed-pair spin sums use DGEMM; full equations; molecular admission pending");
          if (native_spatial_pairs)
            report_progress("native spatial timing: W_abef total is measured; build/BLAS subphase fields are uninstrumented (zero is not a timing measurement); spatial workspace field excludes transient pair matrices");
          options.use_incremental_diis_gram =
              use_incremental_ccsd_diis_gram;
          options.use_root_replicated_diis =
              use_root_replicated_ccsd_diis;
          options.use_distributed_replicated_energy =
              use_distributed_replicated_ccsd_energy;
          options.iteration_observer =
              [&, label](const localized_cc::CcsdIterationReport& iteration) {
                std::ostringstream line;
                line.setf(std::ios::scientific);
                line << std::setprecision(6) << label << " CCSD "
                     << (iteration.converged ? "convergence check after iteration "
                                             : "iteration ")
                     << iteration.iteration << "; E="
                     << iteration.correlation_energy << " Eh; dE="
                     << iteration.delta_energy << " Eh; R1="
                     << iteration.r1_norm << "; R2=" << iteration.r2_norm
                     << "; R1_measure="
                     << iteration.r1_convergence_measure
                     << "; R2_measure="
                     << iteration.r2_convergence_measure
                     << "; backend="
                     << localized_cc::ccsd_amplitude_backend_name(
                            iteration.amplitude_backend)
                     << "; W_abef="
                     << localized_cc::ccsd_wabef_backend_name(
                            iteration.wabef_backend)
                     << "; wall=" << std::fixed << std::setprecision(3)
                     << iteration.timing.iteration_seconds
                     << " s; intermediates="
                     << iteration.timing.intermediates_seconds
                     << " s; intermediate_mixed_tasks="
                     << iteration.timing.intermediate_mixed_pair_tasks
                     << "; intermediate_BLAS="
                     << iteration.timing.intermediate_blas_contractions
                     << "; singles=" << iteration.timing.singles_seconds
                     << " s; doubles=" << iteration.timing.doubles_seconds
                     << " s; W_abef_time=" << iteration.timing.wabef_seconds
                     << " s; W_abef_build="
                     << iteration.timing.wabef_build_seconds
                     << " s; W_abef_BLAS="
                     << iteration.timing.wabef_contraction_seconds
                     << " s; W_abef_accumulate="
                     << iteration.timing.wabef_accumulation_seconds
                     << " s; W_abef_external_tiles="
                     << iteration.timing.wabef_external_tiles
                     << "; W_abef_internal_panels="
                     << iteration.timing.wabef_internal_panels
                     << "; W_abef_output_tasks="
                     << iteration.timing.wabef_output_tasks
                     << "; W_abef_BLAS_contractions="
                     << iteration.timing.wabef_blas_contractions
                     << "; W_abef_factorized_cached="
                     << iteration.timing.wabef_factorized_cached_evaluations
                     << "; residual=" << iteration.timing.residual_seconds
                     << " s; DIIS=" << iteration.timing.diis_seconds
                     << " s; DIIS_products="
                     << iteration.timing.diis_error_dot_products
                     << "; DIIS_history_vectors="
                     << iteration.timing
                            .diis_history_vector_accumulations
                     << "; energy=" << iteration.timing.energy_seconds
                     << " s";
                report_progress(line.str());
              };
        };
    std::optional<localized_cc::CalculationCheckpoint> projection_source;
    std::optional<modernqc::linalg::Matrix> target_source_overlap;
    std::string projection_source_basis;
    if (projection_requested) {
      if (projection_source_path.lexically_normal() ==
          checkpoint_path.lexically_normal())
        throw std::invalid_argument(
            "projection source and target checkpoint paths must differ");
      projection_source =
          localized_cc::read_checkpoint(projection_source_path, "");
      projection_source_basis =
          fingerprint_basis(projection_source->fingerprint);
      if (fingerprint_without_basis(projection_source->fingerprint) !=
          fingerprint_without_basis(fingerprint))
        throw std::runtime_error(
            "basis projection requires identical geometry, charge, "
            "multiplicity, spherical convention, core, and method");
      if (projection_source_basis == basis_name)
        throw std::runtime_error(
            "basis projection requires different source and target bases");
      if (!projection_source->rhf)
        throw std::runtime_error(
            "projection-source checkpoint has no converged RHF state");
      validate_projection_source_rhf(*projection_source->rhf, occupied_count);
      if (metric_orthonormality_error(
              copy_matrix(projection_source->rhf->coefficients),
              copy_matrix(projection_source->rhf->overlap)) > 1.0e-8)
        throw std::runtime_error(
            "projection-source canonical MOs violate S-orthonormality "
            "tolerance");
      const std::size_t source_ao_count =
          projection_source->rhf->overlap.dimensions()[0];
      if (source_ao_count >= ao_count)
        throw std::runtime_error(
            "basis progression requires fewer source AOs than target AOs");
      modernqc::integrals::Libint2IntegralProvider source_integrals(
          molecule, projection_source_basis, true);
      if (source_integrals.basis_metadata().number_of_aos != source_ao_count)
        throw std::runtime_error(
            "projection-source basis name and RHF AO count disagree");
      target_source_overlap =
          integrals.compute_cross_overlap(source_integrals);
      if (!checkpoint.rhf && !checkpoint.rhf_restart_density) {
        const auto source_coefficients =
            copy_matrix(projection_source->rhf->coefficients);
        const auto source_occupied =
            columns(source_coefficients, 0, occupied_count);
        const auto target_overlap = integrals.compute_one_electron().overlap;
        const auto projected = localized_cc::project_rhf_occupied_guess(
            copy_matrix(projection_source->rhf->overlap), source_occupied,
            target_overlap, *target_source_overlap);
        checkpoint.rhf_restart_density =
            copy_matrix(projected.spin_summed_density);
        checkpoint.basis_projection =
            localized_cc::BasisProjectionCheckpointState{
                projection_source_path.string(),
                projection_source->fingerprint,
                projection_source_basis,
                basis_name,
                projected.diagnostics.source_ao_count,
                projected.diagnostics.target_ao_count,
                projected.diagnostics.target_metric_rank,
                projected.diagnostics.minimum_projection_gram_eigenvalue,
                projected.diagnostics.projected_orthonormality_error,
                projected.diagnostics.spin_summed_electron_count,
                false,
                false,
                false};
        persist_checkpoint();
      } else if (!checkpoint.basis_projection) {
        throw std::runtime_error(
            "target checkpoint already contains RHF state but no matching "
            "basis-projection provenance; use a new target checkpoint");
      } else if (checkpoint.basis_projection->source_fingerprint !=
                     projection_source->fingerprint ||
                 checkpoint.basis_projection->target_basis != basis_name) {
        throw std::runtime_error(
            "target checkpoint basis-projection provenance differs from the "
            "requested source checkpoint");
      }
    }
    localized_cc::parallel::barrier();
    const auto start = std::chrono::steady_clock::now();
    modernqc::scf::RhfOptions rhf_options;
    rhf_options.energy_tolerance = 1.0e-11;
    rhf_options.density_rms_tolerance = 1.0e-9;
    rhf_options.commutator_rms_tolerance = 1.0e-8;
    // Use the same distributed direct-Fock execution path as localized LMP2.
    // Every unique AO shell quartet has one MPI owner, each rank evaluates its
    // share with OpenMP, and run_rhf reduces the partial Fock matrices once per
    // iteration.
    //
    // The Cauchy-Schwarz integral screen defaults to zero, which keeps the
    // strict canonical gate exact.  Setting CCSD_AO_SCHWARZ_THRESHOLD enables
    // it; the bound is rigorous, so the omitted quartets are provably below the
    // threshold.  Density screening and the Fock contribution threshold stay
    // off regardless: those are separate approximations without the same
    // guarantee, and enabling them alongside would confound the accounting.
    rhf_options.eri_schwarz_threshold = ao_schwarz_threshold_from_environment();
    rhf_options.density_screen = false;
    rhf_options.fock_contribution_threshold = 0.0;
    rhf_options.mpi_rank = communicator.rank;
    rhf_options.mpi_ranks = communicator.size;
    rhf_options.fock_threads = openmp_thread_count();
    const bool reused_rhf = checkpoint.rhf.has_value();
    std::optional<modernqc::linalg::Matrix> rhf_restart_density;
    if (!reused_rhf && checkpoint.rhf_restart_density) {
      if (checkpoint.rhf_restart_density->dimensions() !=
          std::array<std::size_t, 2>{ao_count, ao_count})
        throw std::runtime_error(
            "RHF restart-density checkpoint dimensions do not match basis");
      rhf_restart_density = copy_matrix(*checkpoint.rhf_restart_density);
    }
    if (reused_rhf) {
      report_progress("RHF checkpoint restore started");
    } else {
      std::ostringstream message;
      // Report the options actually in force rather than the values they used
      // to be fixed at; a screening threshold that is configurable must not be
      // announced as a literal.
      message << "RHF calculation started; MPI ranks="
              << rhf_options.mpi_ranks << "; Fock threads/rank="
              << rhf_options.fock_threads << "; Schwarz threshold="
              << rhf_options.eri_schwarz_threshold << "; density screen="
              << (rhf_options.density_screen ? "enabled" : "disabled");
      report_progress(message.str());
    }
    auto rhf_iteration_start = std::chrono::steady_clock::now();
    const modernqc::scf::RhfObserver rhf_observer =
        [&](const modernqc::scf::RhfIteration& record) {
          const auto now = std::chrono::steady_clock::now();
          const double iteration_seconds =
              std::chrono::duration<double>(now - rhf_iteration_start)
                  .count();
          rhf_iteration_start = now;
          std::ostringstream message;
          message.setf(std::ios::scientific);
          message << std::setprecision(6) << "RHF iteration "
                  << record.iteration << "; E=" << record.total_energy
                  << " Eh; dE=" << record.energy_change
                  << " Eh; density=" << record.density_rms
                  << "; commutator=" << record.commutator_rms
                  << "; wall=" << std::fixed << std::setprecision(3)
                  << iteration_seconds << " s";
          report_progress(message.str());
        };
    auto rhf = reused_rhf ? restore_rhf(*checkpoint.rhf, ao_count)
                          : modernqc::scf::run_rhf(
                                molecule, integrals, rhf_options,
                                rhf_restart_density, rhf_observer,
                                [&](const modernqc::scf::RhfIteration& record,
                                    const modernqc::linalg::Matrix& density) {
                                  if (record.iteration % 5 == 0) {
                                    checkpoint.rhf_restart_density =
                                        copy_matrix(density);
                                    persist_checkpoint();
                                  }
                                });
    if (!reused_rhf) {
      checkpoint.rhf = checkpoint_rhf(rhf);
      checkpoint.rhf_restart_density.reset();
      // Any downstream state belonged to no validated RHF parent.
      checkpoint.localization.reset();
      checkpoint.occupied_localization_optimization.reset();
      checkpoint.virtual_localization_optimization.reset();
      checkpoint.canonical_ccsd.reset();
      checkpoint.canonical_triples.reset();
      checkpoint.localized_ccsd.reset();
      checkpoint.localized_sparse_ccsd.reset();
      checkpoint.localized_triples.reset();
      checkpoint.localized_streamed_triples.reset();
      checkpoint.localized_laplace_triples.reset();
      persist_checkpoint();
    }
    localized_cc::parallel::barrier();
    const auto after_rhf = std::chrono::steady_clock::now();
    if (reused_rhf) {
      report_progress("RHF complete; restored from checkpoint");
    } else {
      std::ostringstream message;
      message << std::setprecision(17) << "RHF complete; iterations="
              << rhf.iteration_count << "; direct-Fock wall="
              << rhf.direct_fock_wall_seconds_total
              << " s; last-rank range="
              << rhf.minimum_direct_fock_wall_seconds_last << "--"
              << rhf.maximum_direct_fock_wall_seconds_last << " s";
      report_progress(message.str());
    }
    const auto rhf_runtime_report = [&]() {
      std::ostringstream report;
      report << "RHF execution topology     distributed direct Fock\n"
             << "RHF MPI ranks              " << rhf_options.mpi_ranks
             << "\n"
             << "RHF Fock threads/rank      " << rhf_options.fock_threads
             << "\n"
             << std::scientific << std::setprecision(12)
             << "RHF ERI Schwarz threshold  "
             << rhf_options.eri_schwarz_threshold << "\n"
             << "RHF density screen         "
             << (rhf_options.density_screen ? "enabled" : "disabled")
             << "\n";
      if (!reused_rhf) {
        report << std::fixed << std::setprecision(6)
               << "RHF shell quartets/fock    "
               << rhf.unique_shell_quartets_per_fock << "\n"
               << "RHF owned quartets min/max "
               << rhf.minimum_owned_shell_quartets_per_fock << " / "
               << rhf.maximum_owned_shell_quartets_per_fock << "\n"
               << "RHF direct Fock total      "
               << rhf.direct_fock_wall_seconds_total << " s\n"
               << "RHF last Fock min/max      "
               << rhf.minimum_direct_fock_wall_seconds_last << " / "
               << rhf.maximum_direct_fock_wall_seconds_last << " s\n";
      }
      return report.str();
    }();

    if (rhf.coefficients.columns() != ao_count)
      throw std::runtime_error("RHF coefficient count differs from AO count");
    const auto canonical_occupied = columns(
        rhf.coefficients, frozen_core_count,
        occupied_count - frozen_core_count);
    const auto canonical_virtuals =
        columns(rhf.coefficients, occupied_count, virtual_count);
    if (run_canonical && projection_source && target_source_overlap) {
      const auto source_coefficients =
          copy_matrix(projection_source->rhf->coefficients);
      const std::size_t source_ao_count = source_coefficients.rows();
      const auto source_active_occupied = columns(
          source_coefficients, frozen_core_count,
          occupied_count - frozen_core_count);
      const auto source_virtuals = columns(
          source_coefficients, occupied_count,
          source_ao_count - occupied_count);
      const auto canonical_occupied_projection =
          localized_cc::project_orbital_rotation(
              canonical_occupied, source_active_occupied,
              *target_source_overlap);
      const auto canonical_virtual_projection =
          localized_cc::project_orbital_rotation(
              canonical_virtuals, source_virtuals,
              *target_source_overlap);
      if (!checkpoint.canonical_ccsd &&
          projection_source->canonical_ccsd &&
          projection_source->canonical_ccsd->converged) {
        const auto projected = localized_cc::project_ccsd_amplitudes(
            projection_source->canonical_ccsd->t1,
            projection_source->canonical_ccsd->t2,
            canonical_occupied_projection.source_to_target,
            canonical_virtual_projection.source_to_target);
        checkpoint.canonical_ccsd = localized_cc::CcsdCheckpointState{
            false, false, 0, 0.0, 0.0, 0.0,
            projected.t1, projected.t2, {}};
        checkpoint.canonical_triples.reset();
        if (!checkpoint.basis_projection)
          throw std::runtime_error(
              "basis-projection provenance was lost before CCSD projection");
        checkpoint.basis_projection->canonical_amplitudes_projected = true;
        persist_checkpoint();
      }
    }
    std::optional<double> canonical_ccsd_reference_energy;
    std::optional<double> canonical_triples_reference_energy;
    if (run_canonical) {
      localized_cc::parallel::barrier();
      const auto after_ao_eri = std::chrono::steady_clock::now();
      auto canonical_transform =
          localized_cc::build_semidirect_spatial_hamiltonian(
              rhf, integrals, canonical_occupied, canonical_virtuals,
              frozen_core_count, memory_mib * 1024 * 1024, 1.0e-10, 1.0e-8,
              semidirect_slice_cap_mib * 1024 * 1024);
      const auto &canonical_hamiltonian = canonical_transform.hamiltonian;
      localized_cc::parallel::barrier();
      const auto after_canonical_hamiltonian = std::chrono::steady_clock::now();
      localized_cc::CcsdOptions canonical_ccsd_options;
      canonical_ccsd_options.checkpoint_interval = 5;
      canonical_ccsd_options.max_iterations =
          static_cast<std::size_t>(ccsd_max_iterations);
      configure_ccsd_runtime(canonical_ccsd_options, "canonical");
      if (checkpoint.canonical_ccsd) {
        const auto &saved = *checkpoint.canonical_ccsd;
        if (saved.localized)
          throw std::runtime_error("canonical checkpoint is marked localized");
        canonical_ccsd_options.initial_t1 = saved.t1;
        canonical_ccsd_options.initial_t2 = saved.t2;
        canonical_ccsd_options.completed_iterations =
            saved.completed_iterations;
        canonical_ccsd_options.initial_history = saved.history;
      }
      canonical_ccsd_options.observer =
          [&](const localized_cc::CcsdProgress &progress) {
            checkpoint.canonical_ccsd =
                localized_cc::CcsdCheckpointState{false,
                                                  progress.converged,
                                                  progress.completed_iterations,
                                                  progress.correlation_energy,
                                                  progress.r1_norm,
                                                  progress.r2_norm,
                                                  progress.t1,
                                                  progress.t2,
                                                  progress.history};
            persist_checkpoint();
          };
      const auto canonical_ccsd =
          run_ccsd(canonical_hamiltonian, std::move(canonical_ccsd_options));
      const auto after_canonical_ccsd = std::chrono::steady_clock::now();
      localized_cc::CanonicalTriplesOptions triples_options;
      triples_options.fock_offdiagonal_tolerance = 1.0e-8;
      triples_options.occupied_task_grain = task_grain;
      triples_options.backend = triples_backend;
      triples_options.compute_bracket = false;
      triples_options.virtual_tile_size = triples_tile_size;
      triples_options.checkpoint_interval_tasks = 8;
      if (checkpoint.canonical_triples) {
        triples_options.initial_bracket_task_energies =
            checkpoint.canonical_triples->bracket_task_energies;
        triples_options.initial_parenthetical_task_energies =
            checkpoint.canonical_triples->parenthetical_task_energies;
        triples_options.completed_tasks =
            checkpoint.canonical_triples->completed_tasks;
      }
      triples_options.observer =
          [&](const localized_cc::CanonicalTriplesProgress &progress) {
            checkpoint.canonical_triples =
                localized_cc::CanonicalTriplesCheckpointState{
                    progress.bracket_task_energies,
                    progress.parenthetical_task_energies,
                    progress.completed_tasks};
            persist_checkpoint();
          };
      const auto canonical_triples =
          localized_cc::solve_canonical_triples_streamed(
              canonical_hamiltonian, canonical_ccsd.t1, canonical_ccsd.t2,
              triples_options);
      canonical_ccsd_reference_energy = canonical_ccsd.correlation_energy;
      canonical_triples_reference_energy =
          canonical_triples.parenthetical_energy;
      const auto after_canonical_triples = std::chrono::steady_clock::now();

      if (communicator.rank == 0) {
        std::cout << "execution mode             canonical only\n"
                  << "MPI ranks                  " << communicator.size << "\n"
                  << "OpenMP threads/rank        " << openmp_thread_count()
                  << "\n"
                  << std::scientific << std::setprecision(6)
                  << "CCSD convergence threshold "
                  << ccsd_convergence_threshold << "\n"
                  << "CCSD residual metric       "
                  << localized_cc::ccsd_residual_metric_name(
                         ccsd_residual_metric)
                  << "\n"
                  << "CCSD backend policy        "
                  << localized_cc::ccsd_amplitude_backend_name(ccsd_backend)
                  << "\n"
                  << "CCSD amplitude backend     "
                  << localized_cc::ccsd_amplitude_backend_name(
                         canonical_ccsd.amplitude_backend)
                  << "\n"
                  << "CCSD W_abef backend        "
                  << localized_cc::ccsd_wabef_backend_name(
                         canonical_ccsd.wabef_backend)
                  << "\n"
                  << "CCSD intermediates backend "
                  << (use_tiled_ccsd_intermediates
                          ? "tiled CPU BLAS W_mbej"
                          : "scalar equation oracle")
                  << "\n"
                  << "CCSD doubles backend       "
                  << (use_tiled_ccsd_doubles ? "tiled CPU BLAS"
                                              : "scalar equation oracle")
                  << "\n"
                  << "scientific energy backend "
                  << "C++ spin equations / zero-copy spatial ERIs"
                  << "\n"
                  << "Global Arrays T2 backend "
                  << (global_arrays_runtime.active()
                          ? (canonical_ccsd.diis_memory.global_arrays
                                 ? "enabled; DIIS history distributed"
                                 : "enabled; pair-store active")
                          : "disabled")
                  << "\n"
                  << "production pair ownership "
                  << (canonical_ccsd.distributed_memory.global_arrays
                          ? "Global Arrays solver+intermediates\n"
                          : "replicated reference solver\n")
                  << "RHF MOs                   "
                  << (reused_rhf ? "reused from checkpoint" : "computed")
                  << "\n"
                  << rhf_runtime_report;
        if (checkpoint.basis_projection) {
          const auto &projection = *checkpoint.basis_projection;
          std::cout << "basis projection          " << projection.source_basis
                    << " -> " << projection.target_basis << "\n"
                    << "projection source AOs     "
                    << projection.source_ao_count << "\n"
                    << "projection target AOs     "
                    << projection.target_ao_count << "\n"
                    << "projection metric rank    "
                    << projection.target_metric_rank << "\n"
                    << std::scientific << "projection min Gram eig   "
                    << projection.minimum_projection_gram_eigenvalue << "\n"
                    << "projection S-orth error   "
                    << projection.occupied_orthonormality_error << "\n"
                    << std::fixed << "projection electron count "
                    << projection.spin_summed_electron_count << "\n"
                    << "projected canonical T1/T2 "
                    << (projection.canonical_amplitudes_projected ? "yes"
                                                                  : "no")
                    << "\n"
                    << "projected localization    "
                    << (projection.localization_seed_projected ? "yes" : "no")
                    << "\n"
                    << "projected localized T1/T2 "
                    << (projection.localized_amplitudes_projected ? "yes"
                                                                  : "no")
                    << "\n";
        }
        std::cout << "geometry point group      " << geometry.point_group
                  << "\n"
                  << "spherical AO functions    " << ao_count << "\n"
                  << "active occupied spatial   "
                  << occupied_count - frozen_core_count << "\n"
                  << "virtual spatial           " << virtual_count << "\n"
                  << std::fixed << std::setprecision(12)
                  << "RHF                       " << rhf.total_energy << " Eh\n"
                  << "RHF iterations            " << rhf.iteration_count << "\n"
                  << "canonical CCSD corr       "
                  << canonical_ccsd.correlation_energy << " Eh\n"
                  << "canonical CCSD iterations "
                  << canonical_ccsd.history.size() << "\n"
                  << "canonical (T)             "
                  << canonical_triples.parenthetical_energy << " Eh\n"
                  << "canonical CCSD(T)         "
                  << rhf.total_energy + canonical_ccsd.correlation_energy +
                         canonical_triples.parenthetical_energy
                  << " Eh\n"
                  << std::scientific << "canonical residuals R1/R2 "
                  << canonical_ccsd.r1_norm << " / " << canonical_ccsd.r2_norm
                  << "\n"
                  << "canonical convergence res "
                  << localized_cc::ccsd_residual_measure(
                         canonical_ccsd.r1_norm, canonical_ccsd.t1.size(),
                         ccsd_residual_metric)
                  << " / "
                  << localized_cc::ccsd_residual_measure(
                         canonical_ccsd.r2_norm, canonical_ccsd.t2.size(),
                         ccsd_residual_metric)
                  << "\n"
                  << "DIIS T2 history backend   "
                  << (canonical_ccsd.diis_memory.global_arrays
                          ? "Global Arrays distributed"
                          : "replicated C++")
                  << "\n";
        if (canonical_ccsd.diis_memory.global_arrays)
          std::cout
              << std::fixed << "DIIS history entries      "
              << canonical_ccsd.diis_memory.history_entries << "\n"
              << "DIIS T2 global            "
              << static_cast<double>(
                     canonical_ccsd.diis_memory.doubles_history_global_bytes) /
                     (1024.0 * 1024.0)
              << " MiB\n"
              << "DIIS T2 local rank0       "
              << static_cast<double>(
                     canonical_ccsd.diis_memory.doubles_history_local_bytes) /
                     (1024.0 * 1024.0)
              << " MiB\n"
              << "DIIS T2 replicated/rank reference "
              << static_cast<double>(
                     canonical_ccsd.diis_memory
                         .doubles_history_replicated_bytes_per_rank) /
                     (1024.0 * 1024.0)
              << " MiB\n"
              << "DIIS T2 rank0 reduction   "
              << static_cast<double>(
                     canonical_ccsd.diis_memory
                         .doubles_history_replicated_bytes_per_rank -
                     canonical_ccsd.diis_memory.doubles_history_local_bytes) /
                     (1024.0 * 1024.0)
              << " MiB\n";
        if (canonical_ccsd.distributed_memory.global_arrays)
          std::cout
              << std::fixed << "solver pair arrays        "
              << canonical_ccsd.distributed_memory.solver_pair_arrays << "\n"
              << "solver pairs global       "
              << static_cast<double>(canonical_ccsd.distributed_memory
                                         .solver_pair_global_bytes) /
                     (1024.0 * 1024.0)
              << " MiB\n"
              << "solver pairs local rank0  "
              << static_cast<double>(canonical_ccsd.distributed_memory
                                         .solver_pair_local_bytes) /
                     (1024.0 * 1024.0)
              << " MiB\n"
              << "solver pairs replicated/rank reference "
              << static_cast<double>(
                     canonical_ccsd.distributed_memory
                         .solver_pair_replicated_bytes_per_rank) /
                     (1024.0 * 1024.0)
              << " MiB\n"
              << "intermediate pair arrays  "
              << canonical_ccsd.distributed_memory.intermediate_pair_arrays
              << "\n"
              << "intermediate pairs global "
              << static_cast<double>(canonical_ccsd.distributed_memory
                                         .intermediate_pair_global_bytes) /
                     (1024.0 * 1024.0)
              << " MiB\n"
              << "intermediate pairs local rank0 "
              << static_cast<double>(canonical_ccsd.distributed_memory
                                         .intermediate_pair_local_bytes) /
                     (1024.0 * 1024.0)
              << " MiB\n"
              << "W_mnij pair arrays       "
              << canonical_ccsd.distributed_memory
                     .occupied_intermediate_pair_arrays
              << "\n"
              << "W_mnij global            "
              << static_cast<double>(
                     canonical_ccsd.distributed_memory
                         .occupied_intermediate_pair_global_bytes) /
                     (1024.0 * 1024.0)
              << " MiB\n"
              << "W_mnij local rank0       "
              << static_cast<double>(
                     canonical_ccsd.distributed_memory
                         .occupied_intermediate_pair_local_bytes) /
                     (1024.0 * 1024.0)
              << " MiB\n"
              << "bounded pair cache/rank  "
              << static_cast<double>(canonical_ccsd.distributed_memory
                                         .bounded_pair_cache_bytes_per_rank) /
                     (1024.0 * 1024.0)
              << " MiB\n"
              << "small work replicated/rank "
              << static_cast<double>(
                     canonical_ccsd.distributed_memory
                         .replicated_small_intermediate_bytes_per_rank) /
                     (1024.0 * 1024.0)
              << " MiB\n"
              << "W_abef accumulator rank0 "
              << static_cast<double>(
                     canonical_ccsd.distributed_memory
                         .streamed_wabef_accumulator_bytes_per_rank) /
                     (1024.0 * 1024.0)
              << " MiB\n";
        std::cout
            << "triples backend            "
            << triples_backend_name(canonical_triples.backend) << "\n"
            << "triples virtual tile      " << triples_tile_size << "\n"
            << std::scientific << "canonical F offdiag max   "
            << canonical_triples.maximum_fock_offdiagonal << " Eh\n"
            << std::fixed << "canonical triples tasks   "
            << canonical_triples.tasks << "\n"
            << "workspace/worker          "
            << canonical_triples.workspace_per_worker_mib << " MiB\n"
            << "tile workspace/task       "
            << canonical_triples.tile_workspace_per_task_mib << " MiB\n"
            << "AO ERI resident           0.000000000000 MiB"
            << " (direct shell quartets)\n"
            << "semidirect owned p rank0 "
            << canonical_transform.diagnostics.local_first_indices << "\n"
            << "semidirect first slice   "
            << static_cast<double>(
                   canonical_transform.diagnostics.first_index_slice_bytes) /
                   (1024.0 * 1024.0)
            << " MiB/rank0\n"
            << "semidirect work buffers  "
            << static_cast<double>(
                   canonical_transform.diagnostics.transform_work_bytes) /
                   (1024.0 * 1024.0)
            << " MiB/rank\n"
            << "semidirect peak estimate "
            << static_cast<double>(
                   canonical_transform.diagnostics.estimated_peak_bytes) /
                   (1024.0 * 1024.0)
            << " MiB/rank0\n"
            << "spatial MO ERI resident   "
            << static_cast<double>(canonical_hamiltonian.eri_chemist.size()) *
                   sizeof(double) / (1024.0 * 1024.0)
            << " MiB\n"
            << "spin MO ERI eliminated   "
            << static_cast<double>(
                   canonical_hamiltonian.eri_chemist.logical_size()) *
                   16.0 * sizeof(double) / (1024.0 * 1024.0)
            << " MiB (zero-copy view)\n"
            << "eliminated W_abef/rank   "
            << static_cast<double>(eliminated_wabef_gib)
            << " GiB (former replicated tensor)\n"
            << "streamed W_abef block    "
            << static_cast<double>(streamed_wabef_mib) << " MiB/task\n"
            << "time RHF                   "
            << elapsed_seconds(start, after_rhf) << " s\n"
            << "time AO ERIs               "
            << elapsed_seconds(after_rhf, after_ao_eri) << " s\n"
            << "time MO Hamiltonian        "
            << elapsed_seconds(after_ao_eri, after_canonical_hamiltonian)
            << " s\n"
            << "time canonical CCSD        "
            << elapsed_seconds(after_canonical_hamiltonian,
                               after_canonical_ccsd)
            << " s\n"
            << "time canonical triples     "
            << elapsed_seconds(after_canonical_ccsd, after_canonical_triples)
            << " s\n";
      }
    }

    if (run_localized) {
      const auto localization_start = std::chrono::steady_clock::now();
      report_progress("orbital localization started");
      const auto centers = atom_centers(integrals.basis_metadata(), atoms.size());
      const std::string expected_localization_method =
          localization_checkpoint_method(localizer);
      const bool reused_localization =
          checkpoint.localization &&
          checkpoint.localization->method == expected_localization_method;
      std::optional<localized_cc::OrbitalRotationProjection>
          projected_localized_occupied_seed;
      std::optional<localized_cc::OrbitalRotationProjection>
          projected_localized_virtual_seed;
      if (!reused_localization && projection_source && target_source_overlap &&
          projection_source->localization &&
          projection_source->localization->method ==
              expected_localization_method) {
        const auto source_overlap =
            copy_matrix(projection_source->rhf->overlap);
        const auto source_localized_occupied = copy_matrix(
            projection_source->localization->occupied_coefficients);
        const auto source_localized_virtual = copy_matrix(
            projection_source->localization->virtual_coefficients);
        if (metric_orthonormality_error(source_localized_occupied,
                                        source_overlap) > 1.0e-8 ||
            metric_orthonormality_error(source_localized_virtual,
                                        source_overlap) > 1.0e-8)
          throw std::runtime_error(
              "projection-source localized MOs violate S-orthonormality "
              "tolerance");
        projected_localized_occupied_seed =
            localized_cc::project_orbital_rotation(
                canonical_occupied, source_localized_occupied,
                *target_source_overlap);
        projected_localized_virtual_seed =
            localized_cc::project_orbital_rotation(
                canonical_virtuals, source_localized_virtual,
                *target_source_overlap);
      }
      modernqc::linalg::Matrix localized_occupied_coefficients(
          canonical_occupied.rows(), canonical_occupied.columns());
      modernqc::linalg::Matrix localized_virtual_coefficients(
          canonical_virtuals.rows(), canonical_virtuals.columns());
      modernqc::linalg::Matrix occupied_rotation(
          canonical_occupied.columns(), canonical_occupied.columns());
      modernqc::linalg::Matrix virtual_rotation(
          canonical_virtuals.columns(), canonical_virtuals.columns());
      double occupied_objective = 0.0;
      double virtual_objective = 0.0;
      if (reused_localization) {
        const auto& saved = *checkpoint.localization;
        if (saved.occupied_coefficients.dimensions() !=
                std::array<std::size_t, 2>{canonical_occupied.rows(),
                                           canonical_occupied.columns()} ||
            saved.virtual_coefficients.dimensions() !=
                std::array<std::size_t, 2>{canonical_virtuals.rows(),
                                           canonical_virtuals.columns()} ||
            saved.occupied_rotation.dimensions() !=
                std::array<std::size_t, 2>{canonical_occupied.columns(),
                                           canonical_occupied.columns()} ||
            saved.virtual_rotation.dimensions() !=
                std::array<std::size_t, 2>{canonical_virtuals.columns(),
                                           canonical_virtuals.columns()})
          throw std::runtime_error(
              "localized checkpoint dimensions do not match RHF spaces");
        localized_occupied_coefficients =
            copy_matrix(saved.occupied_coefficients);
        localized_virtual_coefficients =
            copy_matrix(saved.virtual_coefficients);
        occupied_rotation = copy_matrix(saved.occupied_rotation);
        virtual_rotation = copy_matrix(saved.virtual_rotation);
      } else {
        std::optional<modernqc::localization::CartesianMomentIntegrals>
            moments;
        if (localizer != Localizer::pm) {
          if (communicator.rank == 0) {
            report_progress("FM AO moment construction started");
            moments =
                integrals.compute_cartesian_moments_through_fourth();
          } else {
            moments.emplace();
          }
          const auto broadcast_collection = [&](auto& collection) {
            for (auto& matrix : collection) {
              if (communicator.rank != 0)
                matrix = modernqc::linalg::Matrix{ao_count, ao_count};
              localized_cc::parallel::broadcast_in_place(matrix.values());
            }
          };
          broadcast_collection(moments->first);
          broadcast_collection(moments->second);
          broadcast_collection(moments->third);
          broadcast_collection(moments->fourth);
          report_progress("FM AO moment construction complete");
        }

        std::optional<modernqc::linalg::Matrix>
            meta_lowdin_orthogonal_ao;
        if (localizer == Localizer::pm) {
          report_progress(
              "rank-zero ANO meta-Lowdin AO construction started");
          std::vector<int> construction_status(1, 0);
          std::string construction_error;
          if (communicator.rank == 0) {
            try {
              const auto reference =
                  integrals.compute_ano_reference_overlap();
              meta_lowdin_orthogonal_ao =
                  modernqc::localization::build_meta_lowdin_orthogonal_ao(
                      molecule, integrals.basis_metadata(), rhf.overlap,
                      reference.source_basis,
                      reference.target_source_overlap);
            } catch (const std::exception& error) {
              construction_status.front() = 1;
              construction_error = error.what();
            }
          }
          localized_cc::parallel::broadcast_in_place(construction_status);
          if (construction_status.front() != 0)
            throw std::runtime_error(
                communicator.rank == 0
                    ? construction_error
                    : "rank zero failed to construct the ANO meta-Lowdin "
                      "AO basis");
          if (communicator.rank != 0)
            meta_lowdin_orthogonal_ao.emplace(ao_count, ao_count);
          localized_cc::parallel::broadcast_in_place(
              meta_lowdin_orthogonal_ao->values());
          report_progress(
              "ANO meta-Lowdin AO construction complete; broadcast to MPI "
              "ranks");
        }

        struct SpaceLocalization {
          modernqc::linalg::Matrix coefficients;
          modernqc::linalg::Matrix rotation;
          double objective{};
          int iterations{};
        };
        const auto checked_int = [](const std::size_t value,
                                    const char* description) {
          if (value >
              static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error(std::string{description} +
                                      " exceeds the integer range");
          return static_cast<int>(value);
        };
        const auto run_space =
            [&](const std::string& space,
                const modernqc::linalg::Matrix& canonical,
                const std::optional<localized_cc::OrbitalRotationProjection>&
                    projected_seed,
                std::optional<
                    localized_cc::LocalizationOptimizationCheckpointState>&
                    saved_state) -> SpaceLocalization {
          const bool pipek_mezey = localizer == Localizer::pm;
          const std::string checkpoint_method =
              pipek_mezey ? expected_localization_method
                           : localizer_name(localizer);
          if (saved_state) {
            if (saved_state->method != checkpoint_method ||
                saved_state->space != space)
              saved_state.reset();
            else if (saved_state->rotation.dimensions() !=
                     std::array<std::size_t, 2>{canonical.columns(),
                                                canonical.columns()})
              throw std::runtime_error(
                  space + " localization checkpoint dimensions do not "
                          "match the RHF orbital space");
          }
          if (saved_state && saved_state->converged &&
              saved_state->method == checkpoint_method) {
            report_progress(space +
                            " localization reused from intermediate "
                            "checkpoint");
            const auto rotation = copy_matrix(saved_state->rotation);
            return SpaceLocalization{
                modernqc::linalg::multiply(canonical, rotation), rotation,
                saved_state->objective_history.back(),
                checked_int(saved_state->completed_iterations,
                            "localization iteration count")};
          }

          if (pipek_mezey) {
            modernqc::localization::PipekMezeyOptions options;
            options.maximum_sweeps =
                space == "occupied" ? occupied_pm_maximum_sweeps
                                    : virtual_pm_maximum_sweeps;
            options.objective_tolerance = 1.0e-6;
            options.optimizer =
                modernqc::localization::PipekMezeyOptimizer::pyscf_ciah;
            options.gradient_tolerance = 0.0;
            options.ambiguity_population_margin = 0.05;
            options.population_method = modernqc::localization::
                PipekMezeyPopulationMethod::meta_lowdin;
            options.meta_lowdin_orthogonal_ao =
                *meta_lowdin_orthogonal_ao;
            const bool distributed_localization =
                communicator.size <= atoms.size();
            const bool participates =
                distributed_localization || communicator.rank == 0;
            std::size_t resumed_iterations = 0;
            std::vector<double> resumed_objective_history;
            if (saved_state && saved_state->method == checkpoint_method) {
              options.initial_rotation = copy_matrix(saved_state->rotation);
              resumed_iterations = saved_state->completed_iterations;
              resumed_objective_history = saved_state->objective_history;
              report_progress(
                  space + " PM/CIAH restarted from macro " +
                  std::to_string(saved_state->completed_iterations));
            } else if (projected_seed) {
              options.initial_rotation = projected_seed->initial_rotation;
              report_progress(space +
                              " PM/CIAH using projected initial rotation");
            } else {
              options.atomic_initial_guess = true;
            }
            if (options.initial_rotation.empty()) {
              report_progress(
                  space + " PM/CIAH started; orbitals=" +
                  std::to_string(canonical.columns()) + ", max macros=" +
                  std::to_string(options.maximum_sweeps) +
                  ", MPI ranks=" +
                  std::to_string(distributed_localization
                                     ? communicator.size
                                     : 1) +
                  (distributed_localization
                       ? ", atom ownership=cyclic"
                       : ", rank-zero fallback because MPI ranks exceed "
                         "atoms"));
            }
            if (distributed_localization) {
              options.parallel =
                  modernqc::localization::LocalizationParallelOptions{
                      communicator.rank, communicator.size,
                      [](std::vector<double>& values) {
                        localized_cc::parallel::sum_in_place(values);
                      },
                      [](std::vector<double>& values) {
                        localized_cc::parallel::broadcast_in_place(values);
                      }};
            }
            options.progress =
                [&](const modernqc::localization::PipekMezeySweep& record,
                    const modernqc::linalg::Matrix& rotation,
                    const std::vector<double>& objective_history) {
                  std::vector<double> combined_history =
                      resumed_objective_history;
                  const auto first_new = objective_history.begin() +
                      static_cast<std::ptrdiff_t>(
                          !combined_history.empty() &&
                                  !objective_history.empty()
                              ? 1
                              : 0);
                  combined_history.insert(combined_history.end(), first_new,
                                          objective_history.end());
                  const auto completed =
                      resumed_iterations + static_cast<std::size_t>(
                                               record.completed_sweeps);
                  saved_state =
                      localized_cc::LocalizationOptimizationCheckpointState{
                          checkpoint_method, space, record.converged,
                          copy_matrix(rotation), 1.0,
                          completed,
                          static_cast<std::size_t>(record.keyframes), 0,
                          std::move(combined_history)};
                  std::ostringstream message;
                  message << space << " PM/CIAH ";
                  if (record.completed_sweeps == 0)
                    message << "state";
                  else
                    message << "macro " << completed;
                  message << "; objective=" << std::scientific
                          << std::setprecision(10) << record.objective
                          << ", delta=" << record.sweep_gain
                          << ", gradient=" << record.gradient_norm
                          << ", keyframes=" << record.keyframes
                          << ", hessian_actions="
                          << record.hessian_actions;
                  report_progress(message.str());
                  if (distributed_localization)
                    persist_checkpoint();
                };
            std::vector<std::size_t> orbital_indices(canonical.columns());
            std::iota(orbital_indices.begin(), orbital_indices.end(), 0);
            SpaceLocalization localized;
            if (participates) {
              const auto result =
                  modernqc::localization::localize_pipek_mezey(
                      molecule, integrals.basis_metadata(), rhf.overlap,
                      canonical, orbital_indices, options);
              localized = SpaceLocalization{
                  result.coefficients, result.rotation,
                  result.objective_history.back(),
                  checked_int(resumed_iterations +
                                  static_cast<std::size_t>(
                                      result.completed_sweeps),
                              "PM/CIAH macro count")};
            }
            if (!distributed_localization) {
              if (communicator.rank != 0) {
                localized.coefficients = modernqc::linalg::Matrix{
                    canonical.rows(), canonical.columns()};
                localized.rotation = modernqc::linalg::Matrix{
                    canonical.columns(), canonical.columns()};
              }
              localized_cc::parallel::broadcast_in_place(
                  localized.coefficients.values());
              localized_cc::parallel::broadcast_in_place(
                  localized.rotation.values());
              std::vector<double> objective{localized.objective};
              std::vector<int> iterations{localized.iterations};
              localized_cc::parallel::broadcast_in_place(objective);
              localized_cc::parallel::broadcast_in_place(iterations);
              localized.objective = objective.front();
              localized.iterations = iterations.front();
            }
            report_progress(
                space + " PM/CIAH complete after " +
                std::to_string(localized.iterations) + " macros");
            return localized;
          }

          modernqc::localization::TrustRegionLocalizationOptions options;
          options.maximum_iterations =
              space == "occupied"
                  ? occupied_trust_localization_max_iterations
                  : virtual_trust_localization_max_iterations;
          options.parallel =
              modernqc::localization::LocalizationParallelOptions{
                  communicator.rank, communicator.size,
                  [](std::vector<double>& values) {
                    localized_cc::parallel::sum_in_place(values);
                  }};
          if (saved_state) {
            options.restart_state =
                modernqc::localization::LocalizationRestartState{
                    copy_matrix(saved_state->rotation),
                    saved_state->trust_radius,
                    checked_int(saved_state->completed_iterations,
                                "localization iteration count"),
                    checked_int(saved_state->accepted_steps,
                                "localization accepted-step count"),
                    checked_int(saved_state->rejected_steps,
                                "localization rejected-step count"),
                    saved_state->objective_history};
            report_progress(
                space + " localization restarted at iteration " +
                std::to_string(saved_state->completed_iterations));
          } else {
            if (projected_seed)
              options.initial_rotation = projected_seed->initial_rotation;
            report_progress(
                space + " localization started; orbitals=" +
                std::to_string(canonical.columns()) + ", MPI ranks=" +
                std::to_string(communicator.size) + ", max iterations=" +
                std::to_string(options.maximum_iterations));
          }
          bool first_progress_record = true;
          options.progress =
              [&](const modernqc::localization::LocalizationIteration& record,
                  const modernqc::linalg::Matrix& rotation,
                  const std::vector<double>& objective_history) {
                saved_state =
                    localized_cc::LocalizationOptimizationCheckpointState{
                        checkpoint_method, space, record.converged,
                        copy_matrix(rotation), record.trust_radius,
                        static_cast<std::size_t>(record.iteration),
                        static_cast<std::size_t>(record.accepted_steps),
                        static_cast<std::size_t>(record.rejected_steps),
                        objective_history};
                std::ostringstream message;
                message << space << " localization iteration "
                        << record.iteration;
                if (first_progress_record)
                  message << " state";
                else
                  message << (record.step_accepted ? " accepted"
                                                        : " rejected");
                message
                        << "; objective=" << std::scientific
                        << std::setprecision(10) << record.objective
                        << ", gradient=" << record.gradient_norm
                        << ", radius=" << record.trust_radius;
                report_progress(message.str());
                persist_checkpoint();
                first_progress_record = false;
              };

          const int power = localizer == Localizer::fm2 ? 2 : 4;
          const modernqc::localization::LocalizationResult result =
              modernqc::localization::localize_fourth_moment(
                  canonical, rhf.overlap, *moments, power, options);
          if (!result.converged)
            throw std::runtime_error(
                space + " orbital localization did not converge: " +
                result.convergence_reason);
          report_progress(
              space + " localization complete after " +
              std::to_string(result.iterations) + " iterations");
          return SpaceLocalization{result.coefficients, result.rotation,
                                   result.objective_final,
                                   result.iterations};
        };

        const auto occupied = run_space(
            "occupied", canonical_occupied,
            projected_localized_occupied_seed,
            checkpoint.occupied_localization_optimization);
        localized_occupied_coefficients = occupied.coefficients;
        occupied_rotation = occupied.rotation;
        occupied_objective = occupied.objective;
        const auto virtuals = run_space(
            "virtual", canonical_virtuals,
            projected_localized_virtual_seed,
            checkpoint.virtual_localization_optimization);
        localized_virtual_coefficients = virtuals.coefficients;
        virtual_rotation = virtuals.rotation;
        virtual_objective = virtuals.objective;
        if (use_adapted_pm) {
          report_progress("W"+std::to_string(pm_water_cycle.size())+
              " symmetry-adapted PM localization started; rank-zero constrained optimization");
          std::vector<int> status(1,0);
          std::string failure;
          if (communicator.rank==0) {
            try {
              const auto action=localized_cc::cyclic_ao_symmetry_action(
                  integrals,molecule,rhf.overlap,pm_water_cycle);
              std::vector<std::size_t> ao_atoms(ao_count);
              for(const auto& shell:integrals.basis_metadata().shells)
                for(std::size_t k=0;k<shell.function_count;++k)
                  ao_atoms[shell.first_ao+k]=shell.atom_index;
              auto adapt=[&](const auto& canonical,auto& coefficients,auto& rotation,double& objective) {
                const auto assignments=assign_water_monomers(coefficients,rhf.overlap,centers,atoms.size(),assignment_threshold);
                const auto result=localized_cc::adapt_cyclic_pm(canonical,coefficients,
                    assignments.labels,rhf.overlap,action,*meta_lowdin_orthogonal_ao,
                    ao_atoms,pm_water_cycle.size(),report_progress);
                coefficients=result.coefficients;rotation=result.rotation;objective=result.objective;
              };
              adapt(canonical_occupied,localized_occupied_coefficients,occupied_rotation,occupied_objective);
              adapt(canonical_virtuals,localized_virtual_coefficients,virtual_rotation,virtual_objective);
            } catch(const std::exception& error) {status[0]=1;failure=error.what();}
          }
          localized_cc::parallel::broadcast_in_place(status);
          if(status[0]) throw std::runtime_error(communicator.rank==0?failure:"rank-zero symmetry-adapted PM failed");
          report_progress("W"+std::to_string(pm_water_cycle.size())+
              " symmetry-adapted PM complete; full occupied and virtual spaces retained");
        }
      }
      localized_cc::parallel::broadcast_in_place(
          localized_occupied_coefficients.values());
      localized_cc::parallel::broadcast_in_place(
          localized_virtual_coefficients.values());
      localized_cc::parallel::broadcast_in_place(occupied_rotation.values());
      localized_cc::parallel::broadcast_in_place(virtual_rotation.values());

      MonomerAssignments occupied_assignments;
      MonomerAssignments virtual_assignments;
      if (communicator.rank == 0) {
        occupied_assignments = assign_water_monomers(
            localized_occupied_coefficients, rhf.overlap, centers,
            atoms.size(), assignment_threshold);
        virtual_assignments = assign_water_monomers(
            localized_virtual_coefficients, rhf.overlap, centers,
            atoms.size(), assignment_threshold);
      } else {
        occupied_assignments.labels.resize(canonical_occupied.columns());
        occupied_assignments.confidence.resize(canonical_occupied.columns());
        virtual_assignments.labels.resize(canonical_virtuals.columns());
        virtual_assignments.confidence.resize(canonical_virtuals.columns());
      }
      localized_cc::parallel::broadcast_in_place(occupied_assignments.labels);
      localized_cc::parallel::broadcast_in_place(occupied_assignments.confidence);
      localized_cc::parallel::broadcast_in_place(virtual_assignments.labels);
      localized_cc::parallel::broadcast_in_place(virtual_assignments.confidence);
      occupied_assignments.confident = static_cast<std::size_t>(std::count_if(
          occupied_assignments.confidence.begin(),
          occupied_assignments.confidence.end(),
          [&](const double value) { return value >= assignment_threshold; }));
      virtual_assignments.confident = static_cast<std::size_t>(std::count_if(
          virtual_assignments.confidence.begin(),
          virtual_assignments.confidence.end(),
          [&](const double value) { return value >= assignment_threshold; }));
      const auto require_confident_assignment =
          [&](const MonomerAssignments& assignments,
              const char* space) -> double {
            const double minimum = *std::min_element(
                assignments.confidence.begin(),
                assignments.confidence.end());
            if (assignments.confident == assignments.labels.size())
              return minimum;
            std::ostringstream message;
            message << space
                    << " localization failed the symmetric-Lowdin monomer "
                       "assignment gate: "
                    << assignments.labels.size() - assignments.confident
                    << " orbitals are below " << assignment_threshold
                    << "; minimum assigned-monomer population=" << minimum;
            throw std::runtime_error(message.str());
          };
      if (localizer == Localizer::pm) {
        const double occupied_minimum =
            require_confident_assignment(occupied_assignments, "occupied");
        const double virtual_minimum =
            require_confident_assignment(virtual_assignments, "virtual");
        std::ostringstream assignment_progress;
        assignment_progress << std::setprecision(17)
                            << "PM symmetric-Lowdin assignment gate passed; "
                            << "occupied minimum=" << occupied_minimum
                            << ", virtual minimum=" << virtual_minimum;
        report_progress(assignment_progress.str());
      }
      if (report_lmo_support && communicator.rank == 0) {
        report_lmo_shell_support(
            "occupied", localized_occupied_coefficients,
            integrals.basis_metadata());
        report_lmo_shell_support(
            "virtual", localized_virtual_coefficients,
            integrals.basis_metadata());
        std::cout.flush();
      }
      const auto occupied_spin_assignments =
          localized_cc::expand_spatial_labels(occupied_assignments.labels);
      const auto virtual_spin_assignments =
          localized_cc::expand_spatial_labels(virtual_assignments.labels);
      localized_cc::SelectedCcsdManifoldOptions selected_manifold_options;
      selected_manifold_options.maximum_monomers =
          selected_maximum_monomers;
      const auto selected_manifold =
          localized_cc::plan_selected_ccsd_manifold(
              occupied_spin_assignments, virtual_spin_assignments,
              selected_manifold_options);
      const bool selected_model_is_truncated =
          selected_manifold.selected_singles_elements !=
              selected_manifold.full_singles_elements ||
          selected_manifold.selected_doubles_elements !=
              selected_manifold.full_doubles_elements;
      if (communicator.rank == 0) {
        std::cout
            << "progress                   localized CCSD manifold planned; "
            << "maximum_monomers="
            << (selected_maximum_monomers ==
                        std::numeric_limits<std::size_t>::max()
                    ? "full"
                    : std::to_string(selected_maximum_monomers))
            << ", selected_singles="
            << selected_manifold.selected_singles_elements << "/"
            << selected_manifold.full_singles_elements
            << ", selected_doubles="
            << selected_manifold.selected_doubles_elements << "/"
            << selected_manifold.full_doubles_elements << "\n";
        std::cout.flush();
      }
      const bool assignment_metadata_changed =
          reused_localization &&
          (checkpoint.localization->assignment_threshold !=
               assignment_threshold ||
           checkpoint.localization->occupied_monomers !=
               occupied_assignments.labels ||
           checkpoint.localization->virtual_monomers !=
               virtual_assignments.labels ||
           checkpoint.localization->occupied_confidence !=
               occupied_assignments.confidence ||
           checkpoint.localization->virtual_confidence !=
               virtual_assignments.confidence);
      if (!reused_localization) {
        checkpoint.localization = localized_cc::LocalizationCheckpointState{
            expected_localization_method,
            copy_matrix(localized_occupied_coefficients),
            copy_matrix(localized_virtual_coefficients),
            copy_matrix(occupied_rotation),
            copy_matrix(virtual_rotation),
            occupied_assignments.labels,
            virtual_assignments.labels,
            occupied_assignments.confidence,
            virtual_assignments.confidence,
            assignment_threshold};
        checkpoint.occupied_localization_optimization.reset();
        checkpoint.virtual_localization_optimization.reset();
        checkpoint.localized_ccsd.reset();
        checkpoint.localized_sparse_ccsd.reset();
        checkpoint.localized_triples.reset();
        checkpoint.localized_streamed_triples.reset();
        checkpoint.localized_laplace_triples.reset();
        if (checkpoint.basis_projection &&
            projected_localized_occupied_seed &&
            projected_localized_virtual_seed)
          checkpoint.basis_projection->localization_seed_projected = true;
        persist_checkpoint();
      } else if (assignment_metadata_changed) {
        checkpoint.localization->occupied_monomers =
            occupied_assignments.labels;
        checkpoint.localization->virtual_monomers = virtual_assignments.labels;
        checkpoint.localization->occupied_confidence =
            occupied_assignments.confidence;
        checkpoint.localization->virtual_confidence =
            virtual_assignments.confidence;
        checkpoint.localization->assignment_threshold = assignment_threshold;
        checkpoint.localized_sparse_ccsd.reset();
        checkpoint.localized_laplace_triples.reset();
        persist_checkpoint();
      }
      const auto after_localization = std::chrono::steady_clock::now();
      report_progress("orbital localization complete");

      if(lmp2_adaptive) {
        localized_cc::run_lmp2_adaptive_pilot(molecule,integrals,
            localized_occupied_coefficients,localized_virtual_coefficients,rhf.fock,
            occupied_assignments.labels,virtual_assignments.labels,checkpoint_path,fingerprint,
            mib_from_environment("CCSD_LMP2_DOMAIN_WORKSPACE_MIB",512,false)*1024*1024,
            openmp_thread_count(),boolean_from_environment("CCSD_LMP2_ADAPTIVE_FULL"),report_progress);
        return 0;
      }
      if(lmp2_preintegral) {
        localized_cc::run_lmp2_preintegral_pilot(molecule,integrals,
            localized_occupied_coefficients,localized_virtual_coefficients,rhf.fock,
            occupied_assignments.labels,virtual_assignments.labels,
            checkpoint_path,fingerprint,nonnegative_double_from_environment(
                "CCSD_LMP2_PREINTEGRAL_BUDGET",1e-7),
            mib_from_environment("CCSD_LMP2_DOMAIN_WORKSPACE_MIB",512,false)*1024*1024,
            openmp_thread_count(),report_progress);
        report_progress("preintegral LMP2 pilot complete; full Hamiltonian and CCSD/(T) skipped");
        return 0;
      }

      std::optional<localized_cc::CcsdResult>
          canonical_solver_localized_result;
      std::optional<localized_cc::SpatialRhfHamiltonian>
          canonical_solver_hamiltonian_for_rotation;
      std::size_t canonical_solver_start_iteration = 0;
      double canonical_solver_basis_seconds = 0.0;

      // Full-space CCSD is invariant to independent occupied/virtual unitary
      // rotations.  This opt-in fast path solves in the diagonal-Fock RHF
      // basis and rotates the converged amplitudes exactly into the PM basis
      // before the localized Hamiltonian, nM partition, or (T) is evaluated.
      // No canonical and localized four-index Hamiltonians coexist here.
      if (ccsd_solver_basis == CcsdSolverBasis::canonical &&
          !(checkpoint.localized_ccsd &&
            checkpoint.localized_ccsd->converged)) {
        if (selected_model_is_truncated)
          throw std::invalid_argument(
              "CCSD_SOLVER_BASIS=canonical requires the full, untruncated "
              "CCSD excitation manifold");
        if (selected_ccsd_executor == SelectedCcsdExecutor::block_sparse)
          throw std::invalid_argument(
              "CCSD_SOLVER_BASIS=canonical is incompatible with the "
              "block-sparse selected CCSD executor");
        if (ccsd_solver_backend == CcsdSolverBackend::legacy_spatial &&
            ccsd_backend == localized_cc::CcsdAmplitudeBackend::global_arrays)
          throw std::invalid_argument(
              "CCSD_SOLVER_BACKEND=legacy-spatial currently requires "
              "CCSD_AMPLITUDE_BACKEND=replicated or auto");

        report_progress(
            "canonical-basis full CCSD acceleration started; final "
            "amplitudes will be rotated to the localized basis");
        const auto canonical_transform_begin =
            std::chrono::steady_clock::now();
        auto canonical_transform =
            localized_cc::build_semidirect_spatial_hamiltonian(
                rhf, integrals, canonical_occupied, canonical_virtuals,
                frozen_core_count, memory_mib * 1024 * 1024, 1.0e-10,
                1.0e-8, semidirect_slice_cap_mib * 1024 * 1024);
        const auto canonical_transform_end =
            std::chrono::steady_clock::now();
        if (communicator.rank == 0) {
          const auto& diagnostics = canonical_transform.diagnostics;
          std::cout
              << "progress                   canonical AO-to-MO complete; "
              << diagnostics.cooperative_first_index_groups << " groups, "
              << diagnostics.ranks_in_first_index_group
              << " ranks in rank0 group, "
              << diagnostics.first_indices_in_group
              << " first-index slices in rank0 group; seconds="
              << elapsed_seconds(canonical_transform_begin,
                                 canonical_transform_end)
              << "\n";
          std::cout.flush();
        }

        localized_cc::CcsdOptions canonical_solver_options;
        canonical_solver_options.checkpoint_interval = 5;
        canonical_solver_options.max_iterations =
            static_cast<std::size_t>(ccsd_max_iterations);
        configure_ccsd_runtime(canonical_solver_options,
                               "canonical solver-basis");
        if (use_exact_spin_manifold &&
            ccsd_solver_backend == CcsdSolverBackend::spin_orbital)
          canonical_solver_options.projection_mask =
              std::make_shared<localized_cc::CcsdProjectionMask>(
                  localized_cc::make_closed_shell_spin_projection_mask(
                      canonical_transform.hamiltonian.nocc,
                      canonical_transform.hamiltonian.nvirt));

        if (checkpoint.canonical_ccsd) {
          const auto& saved = *checkpoint.canonical_ccsd;
          if (saved.localized)
            throw std::runtime_error(
                "canonical solver-basis checkpoint is marked localized");
          canonical_solver_options.initial_t1 = saved.t1;
          canonical_solver_options.initial_t2 = saved.t2;
          canonical_solver_options.completed_iterations =
              saved.completed_iterations;
          canonical_solver_options.initial_history = saved.history;
          canonical_solver_start_iteration = saved.completed_iterations;
        } else if (checkpoint.localized_ccsd &&
                   !checkpoint.localized_ccsd->converged) {
          const auto& saved = *checkpoint.localized_ccsd;
          if (!saved.localized)
            throw std::runtime_error(
                "localized solver checkpoint is marked canonical");
          const auto canonical_guess =
              localized_cc::project_ccsd_amplitudes(
                  saved.t1, saved.t2, occupied_rotation, virtual_rotation);
          canonical_solver_options.initial_t1 = canonical_guess.t1;
          canonical_solver_options.initial_t2 = canonical_guess.t2;
          canonical_solver_options.completed_iterations =
              saved.completed_iterations;
          canonical_solver_options.initial_history = saved.history;
          canonical_solver_start_iteration = saved.completed_iterations;
        }
        if (ccsd_solver_backend == CcsdSolverBackend::spin_orbital) {
          canonical_solver_options.observer =
              [&](const localized_cc::CcsdProgress& progress) {
                checkpoint.canonical_ccsd = localized_cc::CcsdCheckpointState{
                    false, progress.converged,
                    progress.completed_iterations,
                    progress.correlation_energy, progress.r1_norm,
                    progress.r2_norm, progress.t1, progress.t2,
                    progress.history};
                persist_checkpoint();
              };
        } else {
          canonical_solver_options.spatial_observer =
              [&](const localized_cc::SpatialRccsdProgress& progress) {
                if (communicator.rank == 0) {
                  const auto spin_t1 =
                      localized_cc::expand_spatial_rccsd_t1_to_spin_orbital(
                          progress.amplitudes.t1);
                  const auto spin_t2 =
                      localized_cc::expand_spatial_rccsd_t2_to_spin_orbital(
                          progress.amplitudes.t2);
                  checkpoint.canonical_ccsd =
                      localized_cc::CcsdCheckpointState{
                          false, progress.converged,
                          progress.completed_iterations,
                          progress.correlation_energy, progress.r1_norm,
                          progress.r2_norm, spin_t1, spin_t2,
                          progress.history};
                }
                persist_checkpoint();
              };
        }

        auto canonical_solution =
            ccsd_solver_backend == CcsdSolverBackend::legacy_spatial
                ? run_legacy_spatial_ccsd(canonical_transform.hamiltonian,
                                          std::move(canonical_solver_options))
                : run_ccsd(canonical_transform.hamiltonian,
                           std::move(canonical_solver_options));
        checkpoint.canonical_ccsd = localized_cc::CcsdCheckpointState{
            false,
            canonical_solution.converged,
            canonical_solution.history.back().iteration,
            canonical_solution.correlation_energy,
            canonical_solution.r1_norm,
            canonical_solution.r2_norm,
            canonical_solution.t1,
            canonical_solution.t2,
            canonical_solution.history};
        checkpoint.canonical_triples.reset();

        const auto amplitude_rotation_begin =
            std::chrono::steady_clock::now();
        const auto localized_guess =
            localized_cc::project_ccsd_amplitudes(
                canonical_solution.t1, canonical_solution.t2,
                modernqc::linalg::transpose(occupied_rotation),
                modernqc::linalg::transpose(virtual_rotation));
        const auto amplitude_rotation_end =
            std::chrono::steady_clock::now();
        canonical_solver_basis_seconds = elapsed_seconds(
            canonical_transform_begin, amplitude_rotation_end);
        checkpoint.localized_ccsd = localized_cc::CcsdCheckpointState{
            true,
            canonical_solution.converged,
            canonical_solution.history.back().iteration,
            canonical_solution.correlation_energy,
            canonical_solution.r1_norm,
            canonical_solution.r2_norm,
            localized_guess.t1,
            localized_guess.t2,
            canonical_solution.history};
        canonical_solution.t1 = localized_guess.t1;
        canonical_solution.t2 = localized_guess.t2;
        canonical_solver_localized_result = std::move(canonical_solution);
        if (localized_hamiltonian_backend ==
            LocalizedHamiltonianBackend::canonical_rotation)
          canonical_solver_hamiltonian_for_rotation =
              std::move(canonical_transform.hamiltonian);
        checkpoint.localized_sparse_ccsd.reset();
        checkpoint.localized_triples.reset();
        checkpoint.localized_streamed_triples.reset();
        checkpoint.localized_laplace_triples.reset();
        persist_checkpoint();
        report_progress(
            "canonical CCSD amplitudes rotated to localized basis; "
            "seconds=" +
            std::to_string(elapsed_seconds(amplitude_rotation_begin,
                                           amplitude_rotation_end)));
      }

      const bool use_block_sparse_ccsd =
          selected_ccsd_executor == SelectedCcsdExecutor::block_sparse;
      const bool use_selected_kernel =
          selected_ccsd_executor == SelectedCcsdExecutor::selected_kernel;
      const bool use_selected_block_boundary =
          use_block_sparse_ccsd ||
          (use_selected_kernel && selected_model_is_truncated);
      const bool use_ao_direct_selected_integrals =
          selected_integral_backend ==
          SelectedIntegralBackend::ao_direct_selected;
      if (localized_hamiltonian_backend ==
              LocalizedHamiltonianBackend::canonical_rotation &&
          (selected_model_is_truncated || use_ao_direct_selected_integrals))
        throw std::invalid_argument(
            "canonical-to-PM Hamiltonian rotation currently requires the "
            "full dense-oracle excitation and integral model");
      std::optional<localized_cc::SemidirectHamiltonianResult>
          localized_transform;
      std::optional<localized_cc::SpatialHamiltonianRotationDiagnostics>
          localized_hamiltonian_rotation;
      std::optional<localized_cc::SelectedSpatialHamiltonian>
          selected_localized_hamiltonian;
      localized_cc::SpatialRhfHamiltonian localized_hamiltonian;
      localized_cc::SelectedSemidirectTransformDiagnostics
          selected_transform_diagnostics;
      const auto localized_hamiltonian_begin =
          std::chrono::steady_clock::now();
      if (use_ao_direct_selected_integrals) {
        report_progress(
            "selected AO-to-localized-MO transform started; full active "
            "localized four-index tensor disabled");
        selected_localized_hamiltonian.emplace(
            localized_cc::build_selected_semidirect_spatial_hamiltonian(
                rhf, integrals, localized_occupied_coefficients,
                localized_virtual_coefficients, selected_manifold,
                memory_mib * 1024 * 1024,
                selected_transform_diagnostics));
        localized_hamiltonian.fock =
            selected_localized_hamiltonian->fock;
        localized_hamiltonian.hcore =
            selected_localized_hamiltonian->hcore;
        localized_hamiltonian.nocc =
            selected_localized_hamiltonian->nocc;
        localized_hamiltonian.nvirt =
            selected_localized_hamiltonian->nvirt;
        localized_hamiltonian.frozen_core = frozen_core_count;
        localized_hamiltonian.reference_energy = rhf.total_energy;
        localized_hamiltonian.nuclear_energy =
            rhf.nuclear_repulsion_energy;
      } else if (localized_hamiltonian_backend ==
                 LocalizedHamiltonianBackend::canonical_rotation) {
        if (!canonical_solver_hamiltonian_for_rotation)
          throw std::runtime_error(
              "canonical-to-PM Hamiltonian rotation requires a canonical "
              "CCSD solve in the current invocation; use a new checkpoint");
        report_progress(
            "exact canonical-to-PM four-index Hamiltonian rotation started; "
            "second AO shell-quartet pass disabled");
        auto rotated = localized_cc::rotate_spatial_hamiltonian(
            std::move(*canonical_solver_hamiltonian_for_rotation),
            occupied_rotation, virtual_rotation,
            memory_mib * 1024U * 1024U);
        localized_hamiltonian_rotation = rotated.diagnostics;
        localized_hamiltonian = std::move(rotated.hamiltonian);
      } else {
        const bool save_hamiltonian=boolean_from_environment("CCSD_HAMILTONIAN_CHECKPOINT");
        const std::filesystem::path sidecar=checkpoint_path.string()+".localized-hamiltonian";
        const std::string contract=fingerprint+"|localized-semidirect-v1|orth=1e-10|rhf=1e-8";
        const bool restore=save_hamiltonian && std::filesystem::exists(sidecar);
        if(restore) {
          report_progress("localized Hamiltonian checkpoint restore started; AO-to-MO skipped");
          std::vector<double> failures(1,0.);
          std::string failure;
          try {
            localized_hamiltonian=localized_cc::read_hamiltonian_checkpoint(sidecar,
                contract,copy_matrix(localized_occupied_coefficients),
                copy_matrix(localized_virtual_coefficients),copy_matrix(rhf.fock),
                memory_mib*1024*1024);
          } catch(const std::exception& e) {failures[0]=1.;failure=e.what();}
          localized_cc::parallel::sum_in_place(failures);
          if(failures[0]!=0.) throw std::runtime_error(failure.empty()?"Hamiltonian checkpoint restore failed on another rank":failure);
          report_progress("localized Hamiltonian checkpoint restored; orbital identity and checksum verified");
          // Explicit legacy migration only. Saved AO hcore and PM coefficients
          // independently check a rounding-only repair; old sidecar stays intact.
          if (boolean_from_environment("CCSD_REBUILD_LEGACY_ONE_BODY")) {
            modernqc::linalg::Matrix coefficients(ao_count,
                localized_hamiltonian.nocc+localized_hamiltonian.nvirt);
            for (std::size_t mu=0;mu<ao_count;++mu) {
              for (std::size_t p=0;p<localized_hamiltonian.nocc;++p)
                coefficients(mu,p)=localized_occupied_coefficients(mu,p);
              for (std::size_t p=0;p<localized_hamiltonian.nvirt;++p)
                coefficients(mu,localized_hamiltonian.nocc+p)=localized_virtual_coefficients(mu,p);
            }
            failures[0]=0.; failure.clear();
            localized_cc::detail::FockRebuildDiagnostics audit;
            try {
              audit=localized_cc::detail::rebuild_legacy_one_body(
                  localized_hamiltonian,rhf.core_hamiltonian,coefficients);
            } catch(const std::exception& e) {failures[0]=1.;failure=e.what();}
            localized_cc::parallel::sum_in_place(failures);
            if(failures[0]!=0.) throw std::runtime_error(failure.empty()?"legacy one-body audit failed on another rank":failure);
            std::ostringstream message;
            message<<std::scientific<<std::setprecision(12)
                <<"legacy one-body rebuild verified; original_Fock_asymmetry="<<audit.original_asymmetry
                <<"; hcore_max_change="<<audit.hcore_maximum_change
                <<"; Fock_max_change="<<audit.fock_maximum_change
                <<"; arithmetic_bound_max="<<audit.maximum_arithmetic_bound
                <<"; units=Eh; eigensolver_symmetry_tolerance=1e-12 unchanged; disk_sidecar_unchanged";
            report_progress(message.str());
          }
        } else {
        report_progress("cooperative localized AO-to-MO transform started");
        localized_transform.emplace(
            localized_cc::build_semidirect_spatial_hamiltonian(
                rhf, integrals, localized_occupied_coefficients,
                localized_virtual_coefficients, frozen_core_count,
                memory_mib * 1024 * 1024, 1.0e-10, 1.0e-8,
                semidirect_slice_cap_mib * 1024 * 1024));
        localized_hamiltonian =
            std::move(localized_transform->hamiltonian);
        if(save_hamiltonian) {
          std::vector<int> status(1,0);std::string failure;
          if(communicator.rank==0) try {
            localized_cc::write_hamiltonian_checkpoint(sidecar,contract,
                copy_matrix(localized_occupied_coefficients),copy_matrix(localized_virtual_coefficients),
                copy_matrix(rhf.fock),localized_hamiltonian);
          } catch(const std::exception& e) {status[0]=1;failure=e.what();}
          localized_cc::parallel::broadcast_in_place(status);
          if(status[0]) throw std::runtime_error(communicator.rank==0?failure:"rank-zero Hamiltonian checkpoint write failed");
          report_progress("localized Hamiltonian checkpoint saved; path="+sidecar.string());
        }
        }
      }
      const auto after_localized_hamiltonian =
          std::chrono::steady_clock::now();
      if (communicator.rank == 0) {
        if (use_ao_direct_selected_integrals) {
          const auto& diagnostics = selected_transform_diagnostics;
          std::cout
              << "progress                   selected AO-to-localized-MO "
                 "complete; "
              << diagnostics.cooperative_first_index_groups << " groups, "
              << diagnostics.ranks_in_first_index_group
              << " ranks in rank0 group, "
              << diagnostics.first_indices_in_group
              << " first-index slices in rank0 group; selected_bytes="
              << diagnostics.selected_integral_bytes
              << "; requested_spatial_blocks="
              << diagnostics.requested_spatial_integral_blocks
              << "; symmetry_unique_spatial_blocks="
              << diagnostics.symmetry_unique_spatial_integral_blocks
              << "; tuple_lookup_bytes="
              << diagnostics.selected_integral_tuple_lookup_bytes
              << "; avoided_full_active_mo_eri_bytes="
              << diagnostics.avoided_full_active_mo_eri_bytes
              << "; estimated_peak_bytes="
              << diagnostics.estimated_peak_bytes
              << "; integral_storage=replicated admission boundary\n";
        } else if (localized_hamiltonian_rotation) {
          std::cout
              << "progress                   canonical-to-PM four-index "
                 "Hamiltonian rotation complete; seconds="
              << localized_hamiltonian_rotation->wall_seconds
              << "; logical_eri_bytes="
              << localized_hamiltonian_rotation->logical_eri_bytes
              << "; stored_eri_bytes="
              << localized_hamiltonian_rotation->stored_eri_bytes
              << "; estimated_peak_additional_bytes="
              << localized_hamiltonian_rotation
                     ->estimated_peak_additional_bytes
              << "\n";
        } else if (localized_transform) {
          const auto& diagnostics = localized_transform->diagnostics;
          std::cout
              << "progress                   localized AO-to-MO complete; "
              << diagnostics.cooperative_first_index_groups << " groups, "
              << diagnostics.ranks_in_first_index_group
              << " ranks in rank0 group, "
              << diagnostics.first_indices_in_group
              << " first-index slices in rank0 group; slice_bytes="
              << diagnostics.first_index_slice_bytes
              << "; slice_cap_mib=" << semidirect_slice_cap_mib << "\n";
        }
        std::cout.flush();
      }
      if (lmp2_domain_audit) {
        report_progress("LMP2 domain pilot started; full PM Fock coupling; "
                        "rank-zero bounded reference; no CCSD or (T) solve");
        std::vector<int> status(1,0);
        std::string failure;
        if (communicator.rank == 0) try {
          const auto o=localized_hamiltonian.nocc,v=localized_hamiltonian.nvirt;
          localized_cc::Matrix foo({o,o}),fvv({v,v});
          for(std::size_t i=0;i<o;++i) for(std::size_t j=0;j<o;++j)
            foo(i,j)=localized_hamiltonian.fock(i,j);
          for(std::size_t a=0;a<v;++a) for(std::size_t b=0;b<v;++b)
            fvv(a,b)=localized_hamiltonian.fock(o+a,o+b);
          localized_cc::Lmp2DomainOptions opt;
          opt.omission_budget_hartree=nonnegative_double_from_environment(
              "CCSD_LMP2_DOMAIN_BUDGET",1e-7);
          opt.maximum_workspace_bytes=mib_from_environment(
              "CCSD_LMP2_DOMAIN_WORKSPACE_MIB",512,false)*1024*1024;
          opt.observer=[&](std::size_t iteration,double residual,double error) {
            std::ostringstream s;s<<std::scientific<<std::setprecision(12)
              <<"LMP2 domain PCG iteration="<<iteration<<" residual_l2="<<residual
              <<" fixed_MP2_solve_energy_bound_Eh="<<error;
            report_progress(s.str());
          };
          const auto pilot=localized_cc::select_lmp2_domains(foo,fvv,
              [&](std::size_t i,std::size_t j,std::size_t a,std::size_t b) {
                return localized_hamiltonian.eri_chemist(i,o+a,j,o+b);
              },occupied_assignments.labels,virtual_assignments.labels,opt);
          const auto audit_path=checkpoint_path.string()+".lmp2-domains";
          localized_cc::write_lmp2_domain_audit(audit_path,pilot,opt,
              occupied_assignments.labels,virtual_assignments.labels,fingerprint);
          for(std::size_t n=0;n<4;++n)
            std::cout<<std::setprecision(17)<<"LMP2 full M"<<n+1<<" "<<pilot.full_nm[n]<<" Eh\n";
          std::cout<<"LMP2 retained elements "<<pilot.retained_elements<<"/"<<pilot.amplitudes.size()
              <<"\nLMP2 fixed-amplitude omission bound "<<pilot.omission_bound<<" Eh\n"
              <<"LMP2 domain audit directory "<<audit_path<<"\n"
              <<"LMP2 domain pilot complete; candidate only; CCSD(T) accuracy admission pending\n"<<std::flush;
        } catch(const std::exception& e) {status[0]=1;failure=e.what();}
        localized_cc::parallel::broadcast_in_place(status);
        if(status[0]) throw std::runtime_error(communicator.rank==0?failure:
            "rank-zero LMP2 domain pilot failed; no CCSD selection applied");
        return 0;
      }
      localized_cc::CcsdOptions localized_ccsd_options;
      localized_ccsd_options.checkpoint_interval = 5;
      localized_ccsd_options.max_iterations =
          static_cast<std::size_t>(ccsd_max_iterations);
      configure_ccsd_runtime(localized_ccsd_options, "localized");
      std::shared_ptr<const localized_cc::PmSymmetry> pm_task_symmetry;
      if (use_pm_task_symmetry) {
        if (localizer != Localizer::pm || use_selected_kernel || use_block_sparse_ccsd ||
            use_ao_direct_selected_integrals || use_exact_spin_manifold)
          throw std::invalid_argument("PM symmetry admission requires unprojected PM full-space dense-oracle execution");
        report_progress("W"+std::to_string(pm_water_cycle.size())+
            " PM symmetry mapping and exhaustive Hamiltonian covariance audit started");
        auto verified=localized_cc::cyclic_adapted_permutation(
            localized_hamiltonian.nocc,localized_hamiltonian.nvirt,pm_water_cycle.size());
        localized_cc::validate_pm_symmetry_labels(verified,occupied_assignments.labels,virtual_assignments.labels);
        localized_cc::validate_pm_symmetry(localized_hamiltonian,verified);
        pm_task_symmetry=std::make_shared<localized_cc::PmSymmetry>(std::move(verified));
        localized_ccsd_options.pm_symmetry=pm_task_symmetry;
        std::ostringstream audit;
        audit << std::setprecision(17) << "W" << pm_water_cycle.size()
              << " PM symmetry verified; operations=" << pm_water_cycle.size()
              << "; hamiltonian_covariance_tolerance_hartree=1e-10"
              << "; reuse=W_abef external pairs and triples moment rows; full nM retained";
        report_progress(audit.str());
      }
      report_progress(pm_residual_orbits
          ? "PM residual execution: representative external tasks; full internal sums"
          : "PM residual execution: all external tasks");
      if ((selected_maximum_monomers !=
               std::numeric_limits<std::size_t>::max() ||
           use_selected_kernel || use_exact_spin_manifold) &&
          !use_block_sparse_ccsd) {
        auto projection_mask =
            localized_cc::make_selected_ccsd_projection_mask(
                occupied_spin_assignments, virtual_spin_assignments,
                selected_manifold_options);
        if (use_exact_spin_manifold) {
          const auto spin_mask =
              localized_cc::make_closed_shell_spin_projection_mask(
                  localized_hamiltonian.nocc,
                  localized_hamiltonian.nvirt);
          projection_mask = localized_cc::intersect_ccsd_projection_masks(
              projection_mask, spin_mask);
          if (communicator.rank == 0) {
            std::cout
                << "progress                   exact closed-shell spin "
                   "manifold planned; selected_singles="
                << projection_mask.selected_singles << "/"
                << projection_mask.singles.size()
                << "; selected_doubles="
                << projection_mask.selected_doubles << "/"
                << projection_mask.doubles.size() << "\n";
            std::cout.flush();
          }
        }
        localized_ccsd_options.projection_mask =
            std::make_shared<localized_cc::CcsdProjectionMask>(
                std::move(projection_mask));
        // The projected equation oracle is deliberately replicated. The
        // block-sparse GA executor remains gated by its own equivalence test.
        localized_ccsd_options.amplitude_backend =
            localized_cc::CcsdAmplitudeBackend::replicated;
      }
      if (use_block_sparse_ccsd && projection_source)
        throw std::runtime_error(
            "basis-projected sparse CCSD restart is not yet admitted");
      if (!use_block_sparse_ccsd && !checkpoint.localized_ccsd &&
          projection_source &&
          target_source_overlap && projection_source->localization &&
          projection_source->localized_ccsd &&
          projection_source->localized_ccsd->converged &&
          projection_source->localization->method ==
              expected_localization_method) {
        const auto occupied_projection =
            localized_cc::project_orbital_rotation(
                localized_occupied_coefficients,
                copy_matrix(
                    projection_source->localization->occupied_coefficients),
                *target_source_overlap);
        const auto virtual_projection =
            localized_cc::project_orbital_rotation(
                localized_virtual_coefficients,
                copy_matrix(
                    projection_source->localization->virtual_coefficients),
                *target_source_overlap);
        const auto projected = localized_cc::project_ccsd_amplitudes(
            projection_source->localized_ccsd->t1,
            projection_source->localized_ccsd->t2,
            occupied_projection.source_to_target,
            virtual_projection.source_to_target);
        checkpoint.localized_ccsd = localized_cc::CcsdCheckpointState{
            true, false, 0, 0.0, 0.0, 0.0,
            projected.t1, projected.t2, {}};
        checkpoint.localized_triples.reset();
        checkpoint.localized_streamed_triples.reset();
        checkpoint.localized_laplace_triples.reset();
        if (!checkpoint.basis_projection)
          throw std::runtime_error(
              "basis-projection provenance was lost before localized CCSD "
              "projection");
        checkpoint.basis_projection->localized_amplitudes_projected = true;
        persist_checkpoint();
      }
      if (boolean_from_environment("CCSD_REFINE_SAVED_CCSD") &&
          !use_block_sparse_ccsd && !canonical_solver_localized_result &&
          checkpoint.localized_ccsd && checkpoint.localized_ccsd->converged) {
        try {
          localized_cc::validate_converged_ccsd_checkpoint(
              *checkpoint.localized_ccsd, true, 2 * localized_hamiltonian.nocc,
              2 * localized_hamiltonian.nvirt, ccsd_convergence_threshold,
              ccsd_convergence_threshold, ccsd_residual_metric);
        } catch (const localized_cc::CcsdCheckpointConvergenceError&) {
          // Only convergence failure is recoverable. Representation, dimensions,
          // finite values, history and antisymmetry must already have passed.
          checkpoint.localized_ccsd->converged = false;
          checkpoint.localized_triples.reset();
          checkpoint.localized_streamed_triples.reset();
          checkpoint.localized_laplace_triples.reset();
          persist_checkpoint();
          report_progress("saved localized CCSD requires refinement at the requested "
                          "tolerance; reusing T1/T2 and rebuilding DIIS; old triples invalidated");
        }
      }
      const auto localized_ccsd_start_iteration =
          canonical_solver_localized_result
              ? canonical_solver_start_iteration
          : use_block_sparse_ccsd
              ? (checkpoint.localized_sparse_ccsd
                     ? checkpoint.localized_sparse_ccsd->completed_iterations
                     : 0U)
              : (checkpoint.localized_ccsd
                     ? checkpoint.localized_ccsd->completed_iterations
                     : 0U);
      if (!use_block_sparse_ccsd && checkpoint.localized_ccsd) {
        const auto& saved = *checkpoint.localized_ccsd;
        if (!saved.localized)
          throw std::runtime_error("localized checkpoint is marked canonical");
        localized_ccsd_options.initial_t1 = saved.t1;
        localized_ccsd_options.initial_t2 = saved.t2;
        localized_ccsd_options.completed_iterations = saved.completed_iterations;
        localized_ccsd_options.initial_history = saved.history;
      }
      if (!use_block_sparse_ccsd && !use_ao_direct_selected_integrals &&
          ccsd_solver_backend == CcsdSolverBackend::legacy_spatial) {
        localized_ccsd_options.spatial_observer =
            [&](const localized_cc::SpatialRccsdProgress& progress) {
              // Keep iterative amplitudes spatial. Expand only on the writer
              // rank at checkpoint boundaries, as in the canonical driver.
              // The converged boundary is saved explicitly after the solve.
              if (progress.converged) return;
              if (communicator.rank == 0) {
                const auto spin_t1 =
                    localized_cc::expand_spatial_rccsd_t1_to_spin_orbital(
                        progress.amplitudes.t1);
                const auto spin_t2 =
                    localized_cc::expand_spatial_rccsd_t2_to_spin_orbital(
                        progress.amplitudes.t2);
                checkpoint.localized_ccsd = localized_cc::CcsdCheckpointState{
                    true, false, progress.completed_iterations,
                    progress.correlation_energy, progress.r1_norm,
                    progress.r2_norm, spin_t1, spin_t2, progress.history};
              }
              persist_checkpoint();
            };
      } else if (!use_block_sparse_ccsd) {
        localized_ccsd_options.observer =
          [&](const localized_cc::CcsdProgress& progress) {
            checkpoint.localized_ccsd = localized_cc::CcsdCheckpointState{
                true,
                progress.converged,
                progress.completed_iterations,
                progress.correlation_energy,
                progress.r1_norm,
                progress.r2_norm,
                progress.t1,
                progress.t2,
                progress.history};
            // Iterative states remain restartable. The converged state is
            // persisted explicitly below as the durable boundary before (T).
            if (!progress.converged)
              persist_checkpoint();
          };
      }
      const bool reused_localized_ccsd = use_block_sparse_ccsd
          ? checkpoint.localized_sparse_ccsd &&
                checkpoint.localized_sparse_ccsd->converged
          : !canonical_solver_localized_result && checkpoint.localized_ccsd &&
                checkpoint.localized_ccsd->converged;
      localized_cc::CcsdResult localized_ccsd;
      std::optional<localized_cc::SparseCcsdResult> sparse_ccsd;
      std::optional<localized_cc::SelectedDoublesLayout> sparse_t2_layout;
      std::optional<std::vector<std::vector<double>>>
          selected_kernel_t2_blocks;
      std::optional<localized_cc::SelectedDoublesView> sparse_t2_view;
      if (use_block_sparse_ccsd) {
        checkpoint.localized_streamed_triples.reset();
        checkpoint.localized_laplace_triples.reset();
        localized_cc::SparseCcsdOptions sparse_options;
        sparse_options.equations = localized_ccsd_options;
        sparse_options.equations.amplitude_backend =
            localized_cc::CcsdAmplitudeBackend::global_arrays;
        sparse_options.equations.wabef_backend =
            localized_cc::CcsdWabefBackend::cpu;
        sparse_options.maximum_phase_cache_bytes =
            memory_mib * 1024U * 1024U;
        sparse_options.maximum_monomers = selected_maximum_monomers;
        sparse_options.restart = checkpoint.localized_sparse_ccsd;
        sparse_options.observer =
            [&](const localized_cc::SparseCcsdCheckpointState& progress) {
              checkpoint.localized_sparse_ccsd = progress;
              if (!progress.converged)
                persist_checkpoint();
            };
        sparse_options.phase_observer =
            [&](const localized_cc::SparseCcsdPhaseStatistics& phase) {
              if (phase.rank != 0)
                return;
              std::ostringstream message;
              message << "block-sparse CCSD phase " << phase.phase
                      << "; iteration=" << phase.iteration
                      << "; rank=" << phase.rank << '/' << phase.ranks
                      << "; owned_output_blocks="
                      << phase.owned_output_blocks
                      << "; declared_source_blocks="
                      << phase.declared_source_blocks
                      << "; locally_owned_source_blocks="
                      << phase.locally_owned_source_blocks
                      << "; unique_blocks_fetched="
                      << phase.unique_blocks_fetched
                      << "; remote_unique_blocks_fetched="
                      << phase.remote_unique_blocks_fetched
                      << "; fetched_bytes=" << phase.fetched_bytes
                      << "; remote_fetched_bytes="
                      << phase.remote_fetched_bytes
                      << "; cache_hits=" << phase.cache_hits
                      << "; dgemm_contractions="
                      << phase.dgemm_contractions
                      << "; ga_gets_inside_kernel="
                      << phase.ga_gets_inside_kernel
                      << "; replicate_calls=" << phase.replicate_calls
                      << "; fetch_seconds=" << phase.fetch_seconds;
              report_progress(message.str());
            };
        report_progress(reused_localized_ccsd
                            ? "converged block-sparse localized CCSD "
                              "checkpoint restore started"
                            : "block-sparse localized CCSD iterations "
                              "started");
        if (use_ao_direct_selected_integrals) {
          if (!selected_localized_hamiltonian)
            throw std::logic_error(
                "selected integral Hamiltonian was not constructed");
          sparse_ccsd = localized_cc::solve_sparse_ccsd(
              selected_localized_hamiltonian->fock,
              selected_localized_hamiltonian->nocc,
              selected_localized_hamiltonian->nvirt,
              selected_localized_hamiltonian->integrals,
              selected_manifold, sparse_options);
        } else {
          sparse_ccsd = localized_cc::solve_sparse_ccsd(
              localized_hamiltonian, selected_manifold, sparse_options);
        }
        if (!sparse_ccsd->converged)
          throw std::runtime_error(
              "block-sparse localized CCSD did not converge within the "
              "iteration limit");
        checkpoint.localized_sparse_ccsd =
            localized_cc::make_sparse_ccsd_checkpoint(
                *sparse_ccsd, occupied_spin_assignments,
                virtual_spin_assignments, selected_maximum_monomers);
        checkpoint.localized_ccsd.reset();
        persist_checkpoint();
        sparse_t2_layout.emplace(selected_manifold);
        sparse_t2_view.emplace(*sparse_t2_layout, sparse_ccsd->t2_blocks);
        localized_ccsd.correlation_energy =
            sparse_ccsd->correlation_energy;
        localized_ccsd.t1 = sparse_ccsd->t1;
        localized_ccsd.r1_norm = sparse_ccsd->r1_norm;
        localized_ccsd.r2_norm = sparse_ccsd->r2_norm;
        localized_ccsd.history = sparse_ccsd->history;
        localized_ccsd.converged = sparse_ccsd->converged;
        localized_ccsd.cumulative_timing =
            sparse_ccsd->cumulative_timing;
        localized_ccsd.amplitude_backend =
            localized_cc::CcsdAmplitudeBackend::global_arrays;
        localized_ccsd.wabef_backend = localized_cc::CcsdWabefBackend::cpu;
        report_progress(reused_localized_ccsd
                            ? "converged block-sparse localized CCSD "
                              "checkpoint restored; CCSD iterations this "
                              "run=0"
                            : "block-sparse localized CCSD converged");
      } else if (canonical_solver_localized_result) {
        localized_ccsd =
            std::move(*canonical_solver_localized_result);
        report_progress(
            "canonical-basis CCSD solve and localized-amplitude rotation "
            "complete");
      } else if (reused_localized_ccsd) {
        report_progress("converged localized CCSD checkpoint restore started");
        const auto& saved = *checkpoint.localized_ccsd;
        localized_ccsd = localized_cc::restore_converged_ccsd(
            saved, true, 2 * localized_hamiltonian.nocc,
            2 * localized_hamiltonian.nvirt, ccsd_convergence_threshold,
            ccsd_convergence_threshold, ccsd_residual_metric,
            restored_ccsd_backend(localized_ccsd_options, saved,
                                  global_arrays_runtime.active()),
            localized_ccsd_options.wabef_backend);
        report_progress(
            "converged localized CCSD checkpoint restored; CCSD iterations "
            "this run=0");
      } else {
        checkpoint.localized_streamed_triples.reset();
        checkpoint.localized_laplace_triples.reset();
        if (use_selected_kernel)
          checkpoint.localized_sparse_ccsd.reset();
        report_progress(
            "localized CCSD hybrid execution planned; dominant output "
            "elements have cyclic MPI ownership and rank-local OpenMP "
            "contractions; MPI ranks=" +
            std::to_string(communicator.size) + ", OpenMP threads/rank=" +
            std::to_string(openmp_thread_count()) +
            ", W_abef tile cap MiB=" + std::to_string(wabef_tile_mib));
        report_progress("localized CCSD iterations started");
        if (use_ao_direct_selected_integrals) {
          if (!selected_localized_hamiltonian)
            throw std::logic_error(
                "selected integral Hamiltonian was not constructed");
          localized_ccsd = run_ccsd(*selected_localized_hamiltonian,
                                    std::move(localized_ccsd_options));
        } else if (ccsd_solver_backend == CcsdSolverBackend::legacy_spatial) {
          // Closed-shell spatial solver on the localized Hamiltonian.
          //
          // This is not a canonical-only path by construction: the spatial
          // residual carries the (1 - delta) Fock terms in both F_ae and F_mi,
          // so a non-diagonal occupied or virtual Fock block is handled
          // exactly.  It was previously reachable only through the canonical
          // route because that was the only place it had been wired, not
          // because the equations required it.
          //
          // The saving is the spin summation: the spatial manifold holds
          // nocc^2 nvirt^2 amplitudes against the spin-orbital (2 nocc)^2
          // (2 nvirt)^2, a factor of sixteen.
          //
          // The preconditioner is the caveat.  It is f_ii - f_aa, which in a
          // localized basis is no longer an orbital-energy difference, so
          // convergence is slower than the canonical case and the solver will
          // refuse outright if any denominator falls below its floor.  Compare
          // the converged energy against the spin-orbital backend before
          // trusting a localized spatial run.
          report_progress(
              "localized CCSD using the closed-shell spatial solver; "
              "amplitudes expand to the spin-orbital convention at the "
              "triples boundary");
          localized_ccsd = run_legacy_spatial_ccsd(
              localized_hamiltonian, std::move(localized_ccsd_options));
        } else {
          localized_ccsd = run_ccsd(localized_hamiltonian,
                                    std::move(localized_ccsd_options));
        }
        checkpoint.localized_ccsd = localized_cc::CcsdCheckpointState{
            true,
            localized_ccsd.converged,
            localized_ccsd.history.back().iteration,
            localized_ccsd.correlation_energy,
            localized_ccsd.r1_norm,
            localized_ccsd.r2_norm,
            localized_ccsd.t1,
            localized_ccsd.t2,
            localized_ccsd.history};
        persist_checkpoint();
      }
      if (use_selected_kernel && execution_stage == ExecutionStage::full) {
        if (use_selected_block_boundary) {
          sparse_t2_layout.emplace(selected_manifold);
          selected_kernel_t2_blocks.emplace(
              sparse_t2_layout->pack(localized_ccsd.t2));
          sparse_t2_view.emplace(*sparse_t2_layout,
                                 *selected_kernel_t2_blocks);
          report_progress(
              "optimized selected CCSD amplitudes packed into the "
              "block-native triples boundary");
        } else {
          report_progress(
              "exhaustive selected CCSD reuses native dense T2 at the "
              "triples boundary; redundant monomer-block packing disabled");
        }
      }
      if (pm_residual_compare) {
        // One fixed-amplitude equation audit, not another CCSD solve. In
        // particular a checkpoint restart must not silently skip this gate.
        report_progress("PM fixed-amplitude residual comparison started");
        const auto o=localized_hamiltonian.nocc,v=localized_hamiltonian.nvirt;
        localized_cc::SpatialRccsdAmplitudes spatial{
            localized_cc::Matrix({o,v}),localized_cc::Tensor4({o,o,v,v})};
        for (std::size_t i=0;i<o;++i) for (std::size_t a=0;a<v;++a) {
          spatial.t1(i,a)=localized_ccsd.t1(2*i,2*a);
          for (std::size_t j=0;j<o;++j) for (std::size_t b=0;b<v;++b)
            spatial.t2(i,j,a,b)=localized_ccsd.t2(2*i,2*j+1,2*a,2*b+1);
        }
        localized_cc::CcsdOptions audit_options;
        configure_ccsd_runtime(audit_options,"localized");
        audit_options.pm_symmetry=pm_task_symmetry;
        audit_options.use_pm_residual_orbits=false;
        const auto start=std::chrono::steady_clock::now();
        const auto baseline=localized_cc::evaluate_spatial_rccsd_residual(localized_hamiltonian,spatial,audit_options);
        const auto middle=std::chrono::steady_clock::now();
        audit_options.use_pm_residual_orbits=true;
        const auto candidate=localized_cc::evaluate_spatial_rccsd_residual(localized_hamiltonian,spatial,audit_options);
        const auto end=std::chrono::steady_clock::now();
        const auto check=[&](const auto& a,const auto& b,const char* label) {
          double maximum=0.0,norm=0.0,scale=0.0;
          for (std::size_t k=0;k<a.size();++k) {
            if (!std::isfinite(a.data()[k]) || !std::isfinite(b.data()[k]))
              throw std::runtime_error("non-finite PM residual audit");
            const double d=std::abs(a.data()[k]-b.data()[k]);
            maximum=std::max(maximum,d); norm=std::hypot(norm,d);
            scale=std::max({scale,std::abs(a.data()[k]),std::abs(b.data()[k])});
          }
          std::ostringstream line;
          line << std::scientific << std::setprecision(12) << "PM residual audit " << label
               << " infinity=" << maximum << " two=" << norm << " gate=" << 1e-12*(1+scale);
          report_progress(line.str());
          if (maximum>1e-12*(1+scale)) throw std::runtime_error("PM residual equation agreement failed");
        };
        check(baseline.r1,candidate.r1,"R1"); check(baseline.r2,candidate.r2,"R2");
        if (communicator.rank==0) {
          std::cout << "PM residual baseline seconds " << std::chrono::duration<double>(middle-start).count() << '\n'
                    << "PM residual representative seconds " << std::chrono::duration<double>(end-middle).count() << '\n'
                    << "PM intermediate baseline seconds " << baseline.timing.intermediates_seconds << '\n'
                    << "PM intermediate representative seconds " << candidate.timing.intermediates_seconds << '\n';
        }
        report_progress("PM fixed-amplitude residual comparison passed; abs=1e-12 rel=1e-12; no amplitudes updated");
      }
      persist_ccsd_stage_checkpoint();
      report_progress("standalone converged localized CCSD checkpoint saved "
                      "before perturbative triples; path=" +
                      ccsd_stage_path.string());
      const auto after_localized_ccsd = std::chrono::steady_clock::now();
      report_progress(reused_localized_ccsd
                          ? "localized CCSD reuse complete"
                          : "localized CCSD converged");
      const auto localized_ccsd_completed_iterations =
          localized_ccsd.history.empty()
              ? localized_ccsd_start_iteration
              : localized_ccsd.history.back().iteration;
      if (localized_ccsd_completed_iterations <
          localized_ccsd_start_iteration)
        throw std::runtime_error(
            "localized CCSD completed-iteration counter moved backwards");
      const auto localized_ccsd_iterations_this_run =
          reused_localized_ccsd
              ? 0U
              : localized_ccsd_completed_iterations -
                    localized_ccsd_start_iteration;
      const auto localized_r1_convergence_measure =
          localized_cc::ccsd_residual_measure(
              localized_ccsd.r1_norm, localized_ccsd.t1.size(),
              ccsd_residual_metric);
      const auto localized_r2_element_count =
          sparse_t2_layout ? sparse_t2_layout->selected_element_count()
                           : localized_ccsd.t2.size();
      const auto localized_r2_convergence_measure =
          localized_cc::ccsd_residual_measure(
              localized_ccsd.r2_norm, localized_r2_element_count,
              ccsd_residual_metric);
      if (ccsd_term_audit_csv) {
        if (use_block_sparse_ccsd)
          throw std::invalid_argument(
              "CCSD energy-term audit currently requires dense replicated "
              "CCSD amplitudes");
        const auto audit_begin = std::chrono::steady_clock::now();
        report_progress(
            "post-convergence CCSD E_ijab sparsity audit started");
        std::vector<int> audit_failed(1, 0);
        if (communicator.rank == 0) {
          try {
            const auto audit = localized_cc::audit_ccsd_energy_terms(
                localized_hamiltonian, localized_ccsd.t1,
                localized_ccsd.t2, occupied_spin_assignments,
                virtual_spin_assignments, ccsd_term_audit_tolerance);
            localized_cc::write_ccsd_energy_term_audit_csv(
                *ccsd_term_audit_csv, audit);
            const auto saved_flags = std::cout.flags();
            const auto saved_precision = std::cout.precision();
            for (const auto& row : audit.rows) {
              const double percentage =
                  audit.total_doubles_terms == 0
                      ? 0.0
                      : 100.0 * static_cast<double>(row.omitted_terms) /
                            static_cast<double>(audit.total_doubles_terms);
              std::cout
                  << "CCSD E_ijab audit        criterion=" << row.criterion
                  << "; tolerance=" << std::scientific
                  << std::setprecision(12) << row.tolerance_hartree
                  << " Eh; omitted=" << row.omitted_terms << "/"
                  << audit.total_doubles_terms << "; percent=" << percentage
                  << "; maximum_term="
                  << row.maximum_omitted_term_hartree
                  << " Eh; signed_change="
                  << row.omitted_signed_energy_hartree
                  << " Eh; absolute_bound="
                  << row.omitted_absolute_bound_hartree << " Eh\n";
            }
            std::cout << "CCSD E_ijab exact zeros "
                      << audit.exact_zero_terms << "/"
                      << audit.total_doubles_terms << "\n"
                      << "CCSD E_ijab singles      "
                      << audit.singles_energy_hartree << " Eh\n"
                      << "CCSD E_ijab doubles      "
                      << audit.doubles_energy_hartree << " Eh\n"
                      << "CCSD E_ijab reconstructed "
                      << audit.singles_energy_hartree +
                             audit.doubles_energy_hartree
                      << " Eh\n"
                      << "CCSD E_ijab audit CSV   "
                      << ccsd_term_audit_csv->string() << "\n";
            std::cout.flags(saved_flags);
            std::cout.precision(saved_precision);
            std::cout.flush();
          } catch (const std::exception& error) {
            audit_failed[0] = 1;
            std::cerr << "CCSD E_ijab audit failed: " << error.what()
                      << '\n';
          }
        }
        localized_cc::parallel::broadcast_in_place(audit_failed);
        if (audit_failed[0] != 0)
          throw std::runtime_error(
              "rank-zero CCSD E_ijab sparsity audit failed");
        report_progress(
            "post-convergence CCSD E_ijab sparsity audit complete; "
            "wall_seconds=" +
            std::to_string(elapsed_seconds(
                audit_begin, std::chrono::steady_clock::now())));
      }
      if (execution_stage == ExecutionStage::ccsd_only) {
        const auto ccsd_partition_begin = std::chrono::steady_clock::now();
        std::optional<localized_cc::EnergyPartition> ccsd_only_partition;
        if (communicator.rank == 0) {
          ccsd_only_partition = use_ao_direct_selected_integrals
              ? localized_cc::partition_ccsd(
                    selected_localized_hamiltonian->fock,
                    selected_localized_hamiltonian->integrals,
                    localized_ccsd.t1, localized_ccsd.t2,
                    selected_localized_hamiltonian->nocc,
                    occupied_spin_assignments, virtual_spin_assignments,
                    partition_screening)
              : localized_cc::partition_ccsd(
                    localized_hamiltonian, localized_ccsd.t1,
                    localized_ccsd.t2, occupied_spin_assignments,
                    virtual_spin_assignments, partition_screening);
          if (std::abs(ccsd_only_partition->sum_error) > 1.0e-9)
            throw std::runtime_error(
                "localized CCSD-only partition violates its additive sum "
                "rule");
        }
        if (communicator.rank == 0) {
          const auto monomer_count = atoms.size() / 3;
          const auto maximum_reported_order =
              std::min<std::size_t>(4, monomer_count);
          const auto retained_order = selected_model_is_truncated
              ? std::min(selected_maximum_monomers,
                         maximum_reported_order)
              : maximum_reported_order;
          double retained_span_energy = 0.0;
          double higher_span_disconnected_energy = 0.0;
          for (std::size_t order = 1; order <= maximum_reported_order;
               ++order) {
            const auto contribution =
                many_body_energy(*ccsd_only_partition, order);
            if (order <= retained_order)
              retained_span_energy += contribution;
            else
              higher_span_disconnected_energy += contribution;
          }
          std::cout
              << "execution stage             localized CCSD only\n"
              << "MPI ranks                  " << communicator.size << "\n"
              << "OpenMP threads/rank        " << openmp_thread_count() << "\n"
              << std::scientific << std::setprecision(6)
              << "CCSD convergence threshold "
              << ccsd_convergence_threshold << "\n"
              << "CCSD residual metric       "
              << localized_cc::ccsd_residual_metric_name(
                     ccsd_residual_metric)
              << "\n"
              << "CCSD solver basis          "
              << ccsd_solver_basis_name(ccsd_solver_basis) << "\n"
              << "CCSD solver backend        "
              << ccsd_solver_backend_name(ccsd_solver_backend) << "\n"
              << "CCSD amplitude backend     "
              << localized_cc::ccsd_amplitude_backend_name(
                     localized_ccsd.amplitude_backend)
              << "\n"
              << "CCSD integral backend      "
              << (use_ao_direct_selected_integrals
                      ? "AO-direct selected spatial blocks; replicated "
                        "admission boundary"
                      : "dense spatial Hamiltonian oracle")
              << "\n"
              << "localized Hamiltonian backend "
              << localized_hamiltonian_backend_name(
                     localized_hamiltonian_backend)
              << "\n"
              << "localized Hamiltonian rotation peak "
              << (localized_hamiltonian_rotation
                      ? localized_hamiltonian_rotation
                            ->estimated_peak_additional_bytes
                      : 0U)
              << " bytes\n"
              << "CCSD selected ERI bytes    "
              << (use_ao_direct_selected_integrals
                      ? selected_transform_diagnostics
                            .selected_integral_bytes
                      : 0U)
              << "\n"
              << "CCSD requested spatial ERI blocks "
              << (use_ao_direct_selected_integrals
                      ? selected_transform_diagnostics
                            .requested_spatial_integral_blocks
                      : 0U)
              << "\n"
              << "CCSD symmetry-unique ERI blocks "
              << (use_ao_direct_selected_integrals
                      ? selected_transform_diagnostics
                            .symmetry_unique_spatial_integral_blocks
                      : 0U)
              << "\n"
              << "CCSD selected tuple lookup bytes "
              << (use_ao_direct_selected_integrals
                      ? selected_transform_diagnostics
                            .selected_integral_tuple_lookup_bytes
                      : 0U)
              << "\n"
              << "CCSD avoided full ERI bytes "
              << (use_ao_direct_selected_integrals
                      ? selected_transform_diagnostics
                            .avoided_full_active_mo_eri_bytes
                      : 0U)
              << "\n"
              << "CCSD W_abef backend        "
              << localized_cc::ccsd_wabef_backend_name(
                     localized_ccsd.wabef_backend)
              << "\n"
              << "CCSD W_abef packing        "
              << localized_cc::ccsd_wabef_packing_backend_name(
                     localized_ccsd.wabef_packing_backend)
              << "\n"
              << "CCSD W_abef pair symmetry  "
              << (use_wabef_pair_symmetry ? "unique a<b; exchange restored" : "off")
              << "\n"
              << "CCSD W_abef invariant cache "
              << localized_ccsd.wabef_invariant_cache_bytes << " bytes\n"
              << "CCSD W_abef cache build    "
              << localized_ccsd.wabef_invariant_cache_build_seconds << " s\n"
              << "CCSD spatial residual workspace "
              // This existing field counts the solve-lifetime spin workspace,
              // not all native per-evaluation pair matrices.
              << localized_ccsd.spatial_residual_workspace_bytes
              << " bytes\n"
              << "CCSD spatial workspace build "
              << localized_ccsd.spatial_residual_workspace_build_seconds
              << " s\n"
              << "CCSD selected panel cache "
              << (localized_ccsd.selected_eri_panel_cache_enabled
                      ? "enabled"
                      : "disabled")
              << "\n"
              << "CCSD selected panel required bytes "
              << localized_ccsd.selected_eri_panel_cache_required_bytes
              << "\n"
              << "CCSD selected panel bytes "
              << localized_ccsd.selected_eri_panel_cache_bytes << "\n"
              << "CCSD selected panel populated elements "
              << localized_ccsd.selected_eri_panel_cache_populated_elements
              << "\n"
              << "CCSD selected panel build "
              << localized_ccsd.selected_eri_panel_cache_build_seconds
              << " s\n"
              << "CCSD intermediates backend "
              << (use_tiled_ccsd_intermediates
                      ? selected_model_is_truncated
                            ? "projection-aware tiled CPU BLAS W_mbej"
                            : "tiled CPU BLAS W_mbej"
                      : "scalar equation oracle")
              << "\n"
              << "CCSD doubles backend       "
              << (use_tiled_ccsd_doubles ? "tiled CPU BLAS"
                                          : "scalar equation oracle")
              << "\n"
              << "CCSD DIIS Gram backend     "
              << (use_incremental_ccsd_diis_gram
                      ? "incremental cached Gram"
                      : "recomputed equation oracle")
              << "\n"
              << "CCSD replicated DIIS       "
              << (use_root_replicated_ccsd_diis
                      ? "rank-zero owner plus amplitude broadcast"
                      : "redundant all-rank oracle")
              << "\n"
              << "CCSD replicated energy     "
              << (use_distributed_replicated_ccsd_energy
                      ? "distributed MPI/OpenMP scalar reduction"
                      : "redundant all-rank serial oracle")
              << "\n"
              << "CCSD execution topology    hybrid MPI/OpenMP\n"
              << "CCSD contraction ownership cyclic MPI output elements; "
                 "rank-local OpenMP\n"
              << "CCSD state source          "
              << (reused_localized_ccsd
                      ? "converged checkpoint; solver skipped\n"
                      : "computed and saved before triples\n")
              << "CCSD completed iterations  "
              << localized_ccsd_completed_iterations << "\n"
              << "CCSD iterations this run   "
              << localized_ccsd_iterations_this_run << "\n"
              << std::fixed << std::setprecision(12)
              << "RHF                       " << rhf.total_energy << " Eh\n"
              << "localized CCSD corr       "
              << localized_ccsd.correlation_energy << " Eh\n"
              << "selected CCSD retained corr "
              << retained_span_energy << " Eh\n"
              << "selected CCSD higher-span disconnected "
              << higher_span_disconnected_energy << " Eh\n"
              << std::scientific
              << "localized residuals       " << localized_ccsd.r1_norm
              << " / " << localized_ccsd.r2_norm << "\n"
              << "localized convergence res "
              << localized_r1_convergence_measure << " / "
              << localized_r2_convergence_measure << "\n"
              << std::fixed
              << "CCSD kernel intermediates this run "
              << localized_ccsd.cumulative_timing.intermediates_seconds
              << " s\n"
              << "CCSD intermediate mixed-pair tiles "
              << localized_ccsd.cumulative_timing
                     .intermediate_mixed_pair_tasks
              << "\n"
              << "CCSD intermediate BLAS contractions "
              << localized_ccsd.cumulative_timing
                     .intermediate_blas_contractions
              << "\n"
              << "CCSD kernel singles this run "
              << localized_ccsd.cumulative_timing.singles_seconds << " s\n"
              << "CCSD kernel doubles this run "
              << localized_ccsd.cumulative_timing.doubles_seconds << " s\n"
              << "CCSD doubles occupied-pair tiles "
              << localized_ccsd.cumulative_timing
                     .doubles_occupied_pair_tasks
              << "\n"
              << "CCSD doubles BLAS contractions "
              << localized_ccsd.cumulative_timing
                     .doubles_blas_contractions
              << "\n"
              << "CCSD kernel W_abef this run "
              << localized_ccsd.cumulative_timing.wabef_seconds << " s\n"
              << "CCSD kernel W_abef build this run "
              << localized_ccsd.cumulative_timing.wabef_build_seconds
              << " s\n"
              << "CCSD kernel W_abef BLAS this run "
              << localized_ccsd.cumulative_timing.wabef_contraction_seconds
              << " s\n"
              << "CCSD W_abef external tiles "
              << localized_ccsd.cumulative_timing.wabef_external_tiles
              << "\n"
              << "CCSD W_abef internal panels "
              << localized_ccsd.cumulative_timing.wabef_internal_panels
              << "\n"
              << "CCSD W_abef output tasks "
              << localized_ccsd.cumulative_timing.wabef_output_tasks << "\n"
              << "CCSD W_abef BLAS contractions "
              << localized_ccsd.cumulative_timing.wabef_blas_contractions
              << "\n"
              << "CCSD W_abef factorized cached evaluations "
              << localized_ccsd.cumulative_timing
                     .wabef_factorized_cached_evaluations
              << "\n"
              << "CCSD kernel residual this run "
              << localized_ccsd.cumulative_timing.residual_seconds << " s\n"
              << "CCSD kernel DIIS this run "
              << localized_ccsd.cumulative_timing.diis_seconds << " s\n"
              << "CCSD DIIS error dot products "
              << localized_ccsd.cumulative_timing.diis_error_dot_products
              << "\n"
              << "CCSD DIIS history-vector accumulations "
              << localized_ccsd.cumulative_timing
                     .diis_history_vector_accumulations
              << "\n"
              << "CCSD kernel energy this run "
              << localized_ccsd.cumulative_timing.energy_seconds << " s\n"
              << "CCSD kernel total this run "
              << localized_ccsd.cumulative_timing.iteration_seconds << " s\n"
              << "time localization          "
              << elapsed_seconds(localization_start, after_localization)
              << " s\n"
              << "time localized Hamiltonian "
              << elapsed_seconds(localized_hamiltonian_begin,
                                 after_localized_hamiltonian)
              << " s\n"
              << "time localized CCSD        "
              << canonical_solver_basis_seconds +
                     elapsed_seconds(after_localized_hamiltonian,
                                     after_localized_ccsd)
              << " s\n"
              << "CCSD checkpoint            " << ccsd_stage_path.string()
              << "\n"
              << "triples execution           skipped by explicit execution "
                 "stage\n";
          for (std::size_t order = 1;
               order <= std::min<std::size_t>(4, monomer_count); ++order)
            std::cout << "CCSD M" << order << "                  "
                      << many_body_energy(*ccsd_only_partition, order)
                      << " Eh\n";
          std::cout << std::scientific
                    << "CCSD partition error      "
                    << ccsd_only_partition->sum_error << " Eh\n"
                    << std::fixed
                    << "time CCSD partition       "
                    << elapsed_seconds(ccsd_partition_begin,
                                       std::chrono::steady_clock::now())
                    << " s\n"
                    << "wall time                   "
                    << elapsed_seconds(start, std::chrono::steady_clock::now())
                    << " s\n";
          std::cout.flush();
        }
        localized_cc::parallel::barrier();
        return 0;
      }
      if (boolean_from_environment("CCSD_W6_INVERSION_AUDIT_ONLY")) {
        if(atoms.size()!=18 || localizer!=Localizer::pm || localized_ccsd_iterations_this_run!=0)
          throw std::runtime_error("W6 inversion audit requires saved full PM CCSD; no CCSD recomputation allowed");
        report_progress("W6 inversion audit: actual geometry, saved PM coefficients, Hamiltonian and CCSD amplitudes; no triples launched");
        const auto action=localized_cc::inversion_ao_action(integrals,molecule,rhf.overlap);
        report_progress("W6 inversion geometry/AO metric gate passed; coordinates unchanged");
        const auto symmetry=localized_cc::match_pm_inversion(localized_occupied_coefficients,
            localized_virtual_coefficients,rhf.overlap,action,report_progress);
        localized_cc::validate_pm_symmetry_labels(symmetry,occupied_assignments.labels,virtual_assignments.labels);
        report_progress("W6 inversion saved PM signed-permutation and nM label gates passed");
        localized_cc::validate_pm_symmetry(localized_hamiltonian,symmetry);
        localized_cc::validate_pm_symmetry_amplitudes(symmetry,localized_ccsd.t1,localized_ccsd.t2);
        if(communicator.rank==0) {
          const auto& map=symmetry.operations[1];
          std::cout<<"CCSD iterations this run   0\n";
          for(std::size_t p=0;p<map.image.size();++p)
            std::cout<<"PM inversion map "<<p<<' '<<map.image[p]<<' '<<map.sign[p]<<'\n';
          std::cout<<"W6_INVERSION_AUDIT_PASSED; disk-backed point-group reuse NOT enabled\n"<<std::flush;
        }
        localized_cc::parallel::barrier();
        return 0;
      }
      std::size_t localized_triples_restarted_tasks = 0;
      std::optional<localized_cc::SemicanonicalTriplesResult>
          semicanonical_triples;
      std::optional<localized_cc::StreamedLaplaceTriplesResult>
          laplace_triples;
      if (localized_triples_representation ==
          LocalizedTriplesRepresentation::semicanonical) {
        if (use_block_sparse_ccsd)
          throw std::runtime_error(
              "block-sparse CCSD requires orbital-invariant Laplace "
              "triples; semicanonical triples are not a block-native "
              "boundary");
        localized_triples_restarted_tasks =
            checkpoint.localized_streamed_triples
                ? static_cast<std::size_t>(std::count(
                      checkpoint.localized_streamed_triples->completed_tasks
                          .begin(),
                      checkpoint.localized_streamed_triples->completed_tasks
                          .end(),
                      static_cast<std::uint8_t>(1U)))
                : 0U;
        localized_cc::CanonicalTriplesOptions localized_triples_options;
        localized_triples_options.fock_offdiagonal_tolerance = 1.0e-8;
        localized_triples_options.occupied_task_grain = task_grain;
        localized_triples_options.backend = triples_backend;
        localized_triples_options.compute_bracket = false;
        localized_triples_options.virtual_tile_size = triples_tile_size;
        localized_triples_options.checkpoint_interval_tasks = 8;
        if (checkpoint.localized_streamed_triples) {
          localized_triples_options.initial_bracket_task_energies =
              checkpoint.localized_streamed_triples->bracket_task_energies;
          localized_triples_options.initial_parenthetical_task_energies =
              checkpoint.localized_streamed_triples
                  ->parenthetical_task_energies;
          localized_triples_options.completed_tasks =
              checkpoint.localized_streamed_triples->completed_tasks;
        }
        localized_triples_options.observer =
            [&](const localized_cc::CanonicalTriplesProgress& progress) {
              checkpoint.localized_streamed_triples =
                  localized_cc::CanonicalTriplesCheckpointState{
                      progress.bracket_task_energies,
                      progress.parenthetical_task_energies,
                      progress.completed_tasks};
              persist_checkpoint();
            };
        report_progress("semicanonical streamed triples started");
        semicanonical_triples =
            localized_cc::solve_localized_triples_semicanonical(
                localized_hamiltonian, localized_ccsd.t1, localized_ccsd.t2,
                localized_triples_options, memory_mib * 1024 * 1024);
        report_progress("semicanonical streamed triples complete");
      } else {
        localized_cc::StreamedLaplaceTriplesOptions laplace_options;
        laplace_options.symmetry_propagation=pm_propagation_orbits;
        report_progress(pm_propagation_orbits
            ? "PM propagation execution: representative targets; full sources and nM orbit multiplicities"
            : "PM propagation execution: all targets");
        laplace_options.shared_pair_packing =
            boolean_from_environment("CCSD_TRIPLES_SHARED_PAIR_PACKING");
        laplace_options.spin_aware_construction =
            boolean_from_environment("CCSD_TRIPLES_SPIN_AWARE");
        report_progress(laplace_options.spin_aware_construction
            ? "Laplace triples construction: spin-aware external tiles; all four sectors; full internal sums and nM"
            : "Laplace triples construction: general spin-orbital external tiles");
        report_progress(laplace_options.shared_pair_packing
            ? "Laplace triples moment packing: shared pairs; coarse OpenMP; CPU only"
            : "Laplace triples moment packing: independent permutation pairs");
        laplace_options.sector_streaming =
            boolean_from_environment("CCSD_LAPLACE_TRIPLES_SECTOR_STREAMING");
        laplace_options.disk_parallel_panels =
            boolean_from_environment("CCSD_TRIPLES_DISK_PARALLEL_PANELS");
        report_progress(laplace_options.disk_parallel_panels
            ? "Laplace disk propagation: concurrent rank-local panel preparation; deterministic wave/root broadcasts"
            : "Laplace disk propagation: sequential root panel preparation");
        if(const char* scratch=std::getenv("CCSD_TRIPLES_SCRATCH_DIR"); scratch && *scratch) {
          laplace_options.scratch_directory=scratch;
          laplace_options.maximum_scratch_bytes=mib_from_environment("CCSD_TRIPLES_SCRATCH_MIB_PER_RANK",0,true)*1024U*1024U;
          laplace_options.propagation_row_batch=static_cast<std::size_t>(
              positive_int_from_environment("CCSD_TRIPLES_PANEL_ROWS",8));
          report_progress("Laplace triples disk rows enabled; explicit scratch quota; source panel rows="+
              std::to_string(laplace_options.propagation_row_batch)+"; no full resident moment/response tensors");
        }
        report_progress(laplace_options.sector_streaming
            ? "Laplace triples storage: sector-streamed moments with bounded fused response; full M1-M6; point checkpoints deferred until all sectors"
            : "Laplace triples storage: all-sector moments; point-major propagation");
        laplace_options.pm_symmetry = pm_task_symmetry;
        laplace_options.block_screening_budget_hartree = triples_screen_budget;
        laplace_options.screening_observer = report_progress;
        laplace_options.progress_observer = report_progress;
        laplace_options.two_spin_sectors=boolean_from_environment("CCSD_TRIPLES_TWO_SPIN_SECTORS");
        if(const auto* path=std::getenv("CCSD_TRIPLES_MOMENT_CHECKPOINT_DIR"))
          laplace_options.moment_checkpoint_directory=path;
        if(const auto* path=std::getenv("CCSD_TRIPLES_SECTOR_CHECKPOINT")) {
          laplace_options.sector_checkpoint_path=path;
          const auto* identity=std::getenv("CCSD_TRIPLES_SECTOR_IDENTITY");
          if(!identity || !*identity) throw std::runtime_error("triples sector checkpoint requires source identity");
          laplace_options.sector_checkpoint_identity=identity;
        }
        if (compare_triples_screen) {
          // Old point checkpoints contain no screen-mask provenance. Keep
          // RHF/PM/CCSD, but never mix screened and unscreened point energies.
          checkpoint.localized_laplace_triples.reset();
          laplace_options.pm_symmetry.reset();
          report_progress("triples block-screen comparison: fresh quadrature points; moment-row symmetry reuse disabled in both arms; RHF/PM/CCSD reusable");
        }
        laplace_options.quadrature_points = laplace_triples_points;
        laplace_options.quadrature_scheme = laplace_triples_scheme;
        laplace_options.requested_energy_error_hartree =
            laplace_triples_error_tolerance;
        laplace_options.virtual_tile_size = triples_tile_size;
        laplace_options.moment_backend = triples_backend;
        laplace_options.maximum_additional_memory_bytes =
            memory_mib * 1024 * 1024;
        if (checkpoint.localized_laplace_triples) {
          if (checkpoint.localized_laplace_triples->quadrature_points !=
              laplace_triples_points)
            throw std::runtime_error(
                "saved Laplace triples quadrature order differs from the "
                "current request");
          if (checkpoint.localized_laplace_triples->quadrature_scheme !=
              localized_cc::laplace_triples_quadrature_scheme_name(
                  laplace_triples_scheme))
            throw std::runtime_error(
                "saved Laplace triples quadrature scheme differs from the "
                "current request");
          if (laplace_options.two_spin_sectors) {
            // The combined point state has no per-sector provenance. The
            // two-sector executor must use its independently fingerprinted
            // sector checkpoint, or recompute triples when none is available.
            // Do not reinterpret combined energies as single-sector values.
            report_progress("two-sector restart: combined point state not reused; "
                            "using sector checkpoints when available, otherwise "
                            "recomputing triples; saved CCSD retained");
          } else {
            laplace_options.initial_point_many_body_energies =
                checkpoint.localized_laplace_triples
                    ->point_many_body_energies;
            laplace_options.completed_points =
                checkpoint.localized_laplace_triples->completed_points;
          }
        }
        laplace_options.checkpoint_observer =
            [&](const localized_cc::StreamedLaplaceTriplesProgress& progress) {
              checkpoint.localized_laplace_triples =
                  localized_cc::LaplaceTriplesCheckpointState{
                      laplace_triples_points,
                      progress.point_many_body_energies,
                      progress.completed_points,
                      localized_cc::laplace_triples_quadrature_scheme_name(
                          laplace_triples_scheme)};
              persist_checkpoint();
            };
        if (compare_triples_screen) {
          laplace_options.checkpoint_observer = {};
          report_progress("triples point checkpointing disabled for block-screen comparison; converged CCSD checkpoint is preserved");
        }
        laplace_options.observer =
            [&](const localized_cc::StreamedLaplaceTriplesPoint& point) {
              std::ostringstream message;
              message << std::setprecision(17)
                      << "orbital-invariant Laplace triples point "
                      << point.point << "/" << point.points
                      << " complete; node=" << point.node
                      << ", weight=" << point.weight
                      << ", seconds=" << point.seconds;
              for (std::size_t order = 0;
                   order < point.many_body_energies.size(); ++order)
                message << ", M" << order + 1 << "="
                        << point.many_body_energies[order];
              report_progress(message.str());
            };
        report_progress(
            "orbital-invariant streamed Laplace triples started");
        report_progress(
            "Laplace triples hybrid execution planned; occupied-triples "
            "rows have cyclic MPI ownership and virtual propagator rows "
            "use rank-local OpenMP; MPI ranks=" +
            std::to_string(communicator.size) + ", OpenMP threads/rank=" +
            std::to_string(openmp_thread_count()));
        laplace_triples = use_selected_block_boundary
            ? localized_cc::solve_localized_triples_laplace_streamed(
                  localized_hamiltonian, localized_ccsd.t1, *sparse_t2_view,
                  occupied_assignments.labels, virtual_assignments.labels,
                  laplace_options)
            : localized_cc::solve_localized_triples_laplace_streamed(
                  localized_hamiltonian, localized_ccsd.t1,
                  localized_ccsd.t2, occupied_assignments.labels,
                  virtual_assignments.labels, laplace_options);
        localized_triples_restarted_tasks =
            laplace_triples->restarted_points;
        report_progress(
            "orbital-invariant streamed Laplace triples complete");
      }
      const double localized_triples_energy =
          laplace_triples
              ? laplace_triples->parenthetical_energy
              : semicanonical_triples->triples.parenthetical_energy;
      std::optional<localized_cc::EnergyPartition> ccsd_partition;
      std::optional<localized_cc::EnergyPartition> triples_partition;
      double ccsd_partition_seconds = 0.0;
      if (communicator.rank == 0) {
        const auto partition_begin = std::chrono::steady_clock::now();
        ccsd_partition = use_selected_block_boundary
            ? localized_cc::partition_ccsd(
                  localized_hamiltonian, localized_ccsd.t1, *sparse_t2_view,
                  occupied_spin_assignments, virtual_spin_assignments,
                  partition_screening)
            : localized_cc::partition_ccsd(
                  localized_hamiltonian, localized_ccsd.t1,
                  localized_ccsd.t2, occupied_spin_assignments,
                  virtual_spin_assignments, partition_screening);
        ccsd_partition_seconds = elapsed_seconds(
            partition_begin, std::chrono::steady_clock::now());
        if (laplace_triples)
          triples_partition =
              localized_cc::partition_triples_many_body_sums(
                  laplace_triples->parenthetical_energy,
                  laplace_triples->many_body_energies,
                  laplace_triples->many_body_terms, partition_screening);
        else if (atoms.size() == 3)
          triples_partition =
              localized_cc::partition_triples_single_monomer(
                  localized_triples_energy, occupied_spin_assignments,
                  virtual_spin_assignments, partition_screening);
        if (std::abs(ccsd_partition->sum_error) > 1.0e-9)
          throw std::runtime_error(
              "localized correlation partition violates its additive sum rule");
        if (triples_partition && std::abs(triples_partition->sum_error) > 1.0e-9)
          throw std::runtime_error(
              "localized triples partition violates its additive sum rule");
        if (partition_prefix != "none" && triples_partition) {
          const std::filesystem::path energy_path =
              partition_prefix + ".energy.csv";
          const std::filesystem::path orbital_path =
              partition_prefix + ".orbitals.csv";
          if (std::filesystem::exists(energy_path) ||
              std::filesystem::exists(orbital_path))
            throw std::runtime_error(
                "refusing to overwrite existing Phase 12 CSV output");
          localized_cc::write_partition_csv(energy_path, *ccsd_partition,
                                             *triples_partition);
          localized_cc::write_orbital_assignments_csv(
              orbital_path, localizer_name(localizer), assignment_threshold,
              occupied_assignments.labels, occupied_assignments.confidence,
              virtual_assignments.labels, virtual_assignments.confidence);
        }
      }
      localized_cc::parallel::barrier();
      const auto after_localized_triples = std::chrono::steady_clock::now();
      if (communicator.rank == 0) {
        if (!run_canonical) {
          std::cout
              << "execution mode             localized only\n"
              << "canonical validation       external reference\n"
              << "MPI ranks                  " << communicator.size << "\n"
              << "OpenMP threads/rank        " << openmp_thread_count() << "\n"
              << std::scientific << std::setprecision(6)
              << "CCSD convergence threshold "
              << ccsd_convergence_threshold << "\n"
              << "CCSD residual metric       "
              << localized_cc::ccsd_residual_metric_name(
                     ccsd_residual_metric)
              << "\n"
              << "CCSD backend policy        "
              << localized_cc::ccsd_amplitude_backend_name(ccsd_backend)
              << "\n"
              << "CCSD amplitude backend     "
              << localized_cc::ccsd_amplitude_backend_name(
                     localized_ccsd.amplitude_backend)
              << "\n"
              << "CCSD W_abef backend        "
              << localized_cc::ccsd_wabef_backend_name(
                     localized_ccsd.wabef_backend)
              << "\n"
              << "CCSD W_abef packing        "
              << localized_cc::ccsd_wabef_packing_backend_name(
                     localized_ccsd.wabef_packing_backend)
              << "\n"
              << "CCSD W_abef pair symmetry  "
              << (use_wabef_pair_symmetry ? "unique a<b; exchange restored" : "off")
              << "\n"
              << "CCSD W_abef invariant cache "
              << localized_ccsd.wabef_invariant_cache_bytes << " bytes\n"
              << "CCSD W_abef cache build    "
              << localized_ccsd.wabef_invariant_cache_build_seconds << " s\n"
              << "CCSD spatial residual workspace "
              << localized_ccsd.spatial_residual_workspace_bytes
              << " bytes\n"
              << "CCSD spatial workspace build "
              << localized_ccsd.spatial_residual_workspace_build_seconds
              << " s\n"
              << "CCSD selected panel cache "
              << (localized_ccsd.selected_eri_panel_cache_enabled
                      ? "enabled"
                      : "disabled")
              << "\n"
              << "CCSD selected panel required bytes "
              << localized_ccsd.selected_eri_panel_cache_required_bytes
              << "\n"
              << "CCSD selected panel bytes "
              << localized_ccsd.selected_eri_panel_cache_bytes << "\n"
              << "CCSD selected panel populated elements "
              << localized_ccsd.selected_eri_panel_cache_populated_elements
              << "\n"
              << "CCSD selected panel build "
              << localized_ccsd.selected_eri_panel_cache_build_seconds
              << " s\n"
              << "CCSD intermediates backend "
              << (use_tiled_ccsd_intermediates
                      ? "tiled CPU BLAS W_mbej"
                      : "scalar equation oracle")
              << "\n"
              << "CCSD doubles backend       "
              << (use_tiled_ccsd_doubles ? "tiled CPU BLAS"
                                          : "scalar equation oracle")
              << "\n"
              << "CCSD execution topology    hybrid MPI/OpenMP\n"
              << "CCSD contraction ownership cyclic MPI output elements; "
                 "rank-local OpenMP\n"
              << "CCSD DIIS solver           "
              << "scale-invariant Pulay; rank-pruned history\n"
              << "replicated T2 threshold   " << replicated_t2_max_mib
              << " MiB\n"
              << "W_abef tile cap           " << wabef_tile_mib << " MiB\n"
              << "W_abef invariant cache cap " << wabef_invariant_cache_mib
              << " MiB/rank\n"
              << "selected ERI panel cache cap "
              << selected_eri_panel_cache_mib << " MiB/rank\n"
              << "scientific energy backend "
              << "C++ spin equations / zero-copy spatial ERIs\n"
              << "Global Arrays T2 backend "
              << (use_block_sparse_ccsd
                      ? "enabled; selected blocks and DIIS distributed"
                      : global_arrays_runtime.active()
                      ? (localized_ccsd.diis_memory.global_arrays
                             ? "enabled; DIIS history distributed"
                             : "enabled; pair-store active")
                      : "disabled")
              << "\n"
              << "production pair ownership "
              << (use_block_sparse_ccsd
                      ? "Global Arrays selected T2/tau/W blocks\n"
                      : localized_ccsd.distributed_memory.global_arrays
                      ? "Global Arrays solver+intermediates\n"
                      : "replicated reference solver\n")
              << "RHF MOs                   "
              << (reused_rhf ? "reused from checkpoint" : "computed")
              << "\n"
              << rhf_runtime_report
              << "geometry point group      " << geometry.point_group << "\n"
              << "spherical AO functions    " << ao_count << "\n"
              << "active occupied spatial   "
              << occupied_count - frozen_core_count << "\n"
              << "virtual spatial           " << virtual_count << "\n"
              << std::fixed << std::setprecision(12)
              << "RHF                       " << rhf.total_energy << " Eh\n"
              << "RHF iterations            " << rhf.iteration_count << "\n";
        }
        std::cout << "localizer                  " << localizer_name(localizer)
                  << (reused_localization ? " (reused)\n" : " (computed)\n")
                  << "localization optimizer      "
                  << (use_adapted_pm ? "CIAH seed + cyclic symmetry-constrained PM"
                      : localizer == Localizer::pm ? "PySCF 2.11 CIAH"
                                                  : "trust region")
                  << "\n"
                  << "PM population definition    "
                  << (localizer == Localizer::pm
                          ? "ANO-preorthogonalized meta-Lowdin"
                          : "not applicable")
                  << "\n"
                  << "assignment definition       "
                  << (localizer == Localizer::pm
                          ? "symmetric-Lowdin atom then monomer"
                          : "method-specific")
                  << "\n"
                  << "occupied localization max "
                  << (localizer == Localizer::pm
                          ? occupied_pm_maximum_sweeps
                          : occupied_trust_localization_max_iterations)
                  << (localizer == Localizer::pm ? " macros\n"
                                                  : " iterations\n")
                  << "virtual localization max  "
                  << (localizer == Localizer::pm
                          ? virtual_pm_maximum_sweeps
                          : virtual_trust_localization_max_iterations)
                  << (localizer == Localizer::pm ? " macros\n"
                                                  : " iterations\n")
                  << "CCSD maximum iterations    "
                  << ccsd_max_iterations << "\n"
                  << "CCSD state source          "
                  << (reused_localized_ccsd
                          ? "converged checkpoint; solver skipped\n"
                          : "computed and saved before triples\n")
                  << "CCSD completed iterations  "
                  << localized_ccsd_completed_iterations << "\n"
                  << "CCSD iterations this run   "
                  << localized_ccsd_iterations_this_run
                  << "\n"
                  << "assignment threshold       " << assignment_threshold
                  << "\n"
                  << "occupied confident labels "
                  << occupied_assignments.confident << " / "
                  << occupied_assignments.labels.size() << "\n"
                  << "virtual confident labels  "
                  << virtual_assignments.confident << " / "
                  << virtual_assignments.labels.size() << "\n";
        if (!reused_localization)
          std::cout << std::fixed << std::setprecision(12)
                    << "occupied objective        " << occupied_objective
                    << "\nvirtual objective         " << virtual_objective
                    << "\n";
        std::cout << "CCSD manifold              "
                  << (selected_model_is_truncated
                          ? "selected structural approximation"
                          : "full exact external-index span")
                  << "\n"
                  << "CCSD equation executor     "
                  << (use_block_sparse_ccsd
                          ? "sparse localized selected equations; "
                            "fail-closed spatial ERI blocks"
                          : use_selected_kernel
                          ? "optimized localized selected kernel"
                          : selected_maximum_monomers !=
                              std::numeric_limits<std::size_t>::max()
                          ? "dense projected-equation oracle"
                          : "dense full-space equations")
                  << "\n"
                  << "CCSD triples boundary      "
                  << (use_block_sparse_ccsd
                          ? "selected-block T2 view; no dense unpack"
                          : use_selected_block_boundary
                          ? "selected-block T2 view; packed once after CCSD"
                          : use_selected_kernel
                          ? "native dense amplitudes; exhaustive selected "
                            "manifold"
                          : "native dense amplitudes")
                  << "\n"
                  << "CCSD selected singles      "
                  << selected_manifold.selected_singles_elements << " / "
                  << selected_manifold.full_singles_elements << "\n"
                  << "CCSD selected doubles      "
                  << selected_manifold.selected_doubles_elements << " / "
                  << selected_manifold.full_doubles_elements << "\n"
                  << "CCSD required Fock blocks  "
                  << selected_manifold.required_fock_blocks.size() << "\n"
                  << "CCSD required ERI blocks   "
                  << selected_manifold.required_integral_blocks.size() << "\n"
                  << "CCSD dependency closure    "
                  << (selected_manifold.dependency_status.ready_for_sparse_ccsd()
                          ? "complete"
                          : "incomplete")
                  << "\n"
                  << "CCSD sparse global bytes   "
                  << (sparse_ccsd
                          ? sparse_ccsd->memory.global_arrays_allocated_bytes
                          : 0U)
                  << "\n"
                  << "CCSD declared spin ERIs    "
                  << (sparse_ccsd
                          ? sparse_ccsd->memory
                                .declared_spin_integral_elements
                          : 0U)
                  << " elements\n"
                  << "CCSD selected spatial ERIs "
                  << (sparse_ccsd
                          ? sparse_ccsd->memory
                                .selected_spatial_integral_elements
                          : 0U)
                  << " elements; "
                  << (sparse_ccsd
                          ? sparse_ccsd->memory
                                .selected_spatial_integral_bytes
                          : 0U)
                  << " bytes/rank (dense-oracle producer)\n"
                  << "CCSD sparse cache peak     "
                  << (sparse_ccsd
                          ? sparse_ccsd->memory.largest_phase_cache_bytes
                          : 0U)
                  << " bytes/rank\n"
                  << std::fixed << std::setprecision(12)
                  << "localized CCSD corr       "
                  << localized_ccsd.correlation_energy << " Eh\n"
                  << "localized (T)             "
                  << localized_triples_energy << " Eh\n"
                  << "localized CCSD(T)         "
                  << rhf.total_energy + localized_ccsd.correlation_energy +
                         localized_triples_energy
                  << " Eh\n";
        if (canonical_ccsd_reference_energy &&
            canonical_triples_reference_energy)
          std::cout << std::scientific
                    << "CCSD invariance           "
                    << std::abs(*canonical_ccsd_reference_energy -
                                localized_ccsd.correlation_energy)
                    << " Eh\n"
                    << "(T) invariance            "
                    << std::abs(*canonical_triples_reference_energy -
                                localized_triples_energy)
                    << " Eh\n";
        std::cout << std::scientific
                  << "localized residuals       " << localized_ccsd.r1_norm
                  << " / " << localized_ccsd.r2_norm << "\n"
                  << "localized convergence res "
                  << localized_r1_convergence_measure << " / "
                  << localized_r2_convergence_measure << "\n";
        if (laplace_triples) {
          std::cout
              << "localized F offdiag max   "
              << laplace_triples->localized_fock_offdiagonal << " Eh\n"
              << "(T) representation        orbital-invariant Laplace "
                 "resolvent\n"
              << "triples backend            "
              << triples_backend_name(triples_backend) << "\n"
              << "triples execution topology hybrid MPI/OpenMP\n"
              << "triples row ownership      cyclic occupied triples; "
                 "rank-local virtual rows\n"
              << "triples virtual tile      " << triples_tile_size << "\n"
              << "Laplace triples points    "
              << laplace_triples->quadrature_points << "\n"
              << "Laplace triples scheme    "
              << localized_cc::laplace_triples_quadrature_scheme_name(
                     laplace_triples_scheme)
              << "\n"
              << "Laplace reciprocal error "
              << laplace_triples->maximum_reciprocal_error << " Eh^-1\n"
              << "Laplace certified bound  "
              << laplace_triples->certified_quadrature_error_bound
              << " Eh\n"
              << "triples block screening budget " << triples_screen_budget << " Eh\n"
              << "triples block screening bound " << laplace_triples->block_screening_bound_hartree << " Eh\n"
              << "triples combined error bound "
              << laplace_triples->block_screening_bound_hartree + laplace_triples->certified_quadrature_error_bound << " Eh\n"
              << "triples block screening candidates " << laplace_triples->block_screening_candidates << "\n"
              << "triples block screening skipped " << laplace_triples->block_screening_skipped << "\n"
              << "triples block screening plan bytes " << laplace_triples->block_screening_plan_bytes << "\n"
              << "time triples block screening " << laplace_triples->block_screening_seconds << " s\n"
              << std::fixed
              << "localized triples rows    "
              << laplace_triples->occupied_triples << "\n"
              << "triples symmetry unique rows "
              << laplace_triples->symmetry_unique_moment_rows << "\n"
              << "triples restarted points "
              << localized_triples_restarted_tasks << "\n"
              << "triples moment bytes/rank "
              << laplace_triples->maximum_local_moment_bytes << "\n"
              << "triples planned peak additional bytes/rank "
              << laplace_triples->planned_peak_additional_bytes << "\n"
              << "triples planned scratch bytes/rank "
              << laplace_triples->planned_scratch_bytes << "\n"
              << "triples panel bytes/rank  "
              << laplace_triples->maximum_broadcast_panel_bytes << "\n"
              << "time triples moments      "
              << laplace_triples->moment_construction_seconds << " s\n"
              << "time triples packing rank0 "
              << laplace_triples->moment_packing_seconds << " s\n"
              << "time triples contraction rank0 "
              << laplace_triples->moment_contraction_seconds << " s\n"
              << "time triples propagator   "
              << laplace_triples->propagator_seconds << " s\n"
              << "time disk panel reads rank0 "
              << laplace_triples->disk_panel_read_seconds << " s\n"
              << "time disk virtual transforms rank0 "
              << laplace_triples->disk_panel_transform_seconds << " s\n"
              << "disk parallel panel waves "
              << laplace_triples->disk_panel_waves << "\n"
              << "time triples nM accumulation rank0 "
              << laplace_triples->partition_accumulation_seconds << " s\n"
              << "time triples communication "
              << laplace_triples->communication_seconds << " s\n";
        } else {
          std::cout
              << "localized F offdiag max   "
              << semicanonical_triples->localized_fock_offdiagonal
              << " Eh\n"
              << "semicanonical F offdiag  "
              << semicanonical_triples->semicanonical_fock_offdiagonal
              << " Eh\n"
              << "(T) representation        exact semicanonical streamed\n"
              << "triples backend            "
              << triples_backend_name(
                     semicanonical_triples->triples.backend)
              << "\n"
              << "triples virtual tile      " << triples_tile_size << "\n"
              << std::fixed
              << "localized triples tasks   "
              << semicanonical_triples->triples.tasks << "\n"
              << "triples restarted tasks  "
              << localized_triples_restarted_tasks << "\n"
              << "triples workspace/worker "
              << semicanonical_triples->triples.workspace_per_worker_mib
              << " MiB\n"
              << "tile workspace/task       "
              << semicanonical_triples->triples.tile_workspace_per_task_mib
              << " MiB\n"
              << "semicanonical work/rank  "
              << semicanonical_triples->rotation_peak_additional_mib
              << " MiB (peak additional estimate)\n";
        }
        std::cout << std::fixed
                << "AO ERI resident           0.000000000000 MiB"
                << " (direct shell quartets)\n"
                << "localized Hamiltonian backend "
                << localized_hamiltonian_backend_name(
                       localized_hamiltonian_backend)
                << "\n"
                << "localized semidirect p rank0 "
                << (localized_transform
                        ? localized_transform->diagnostics.local_first_indices
                        : 0U)
                << "\n"
                << "first-index MPI groups    "
                << (localized_transform
                        ? localized_transform->diagnostics
                              .cooperative_first_index_groups
                        : 0U)
                << "\n"
                << "ranks/group rank0         "
                << (localized_transform
                        ? localized_transform->diagnostics
                              .ranks_in_first_index_group
                        : 0U)
                << "\n"
                << "first slices/group rank0  "
                << (localized_transform
                        ? localized_transform->diagnostics
                              .first_indices_in_group
                        : 0U)
                << "\n"
                << "localized semidirect peak "
                << (localized_transform
                        ? static_cast<double>(localized_transform->diagnostics
                                                  .estimated_peak_bytes) /
                              (1024.0 * 1024.0)
                        : 0.0)
                << " MiB/rank0\n"
                << "localized rotation peak   "
                << (localized_hamiltonian_rotation
                        ? static_cast<double>(localized_hamiltonian_rotation
                                                  ->estimated_peak_additional_bytes) /
                              (1024.0 * 1024.0)
                        : 0.0)
                << " MiB/rank\n"
                << "spatial MO ERI resident   "
                << static_cast<double>(
                       localized_hamiltonian.eri_chemist.size()) *
                       sizeof(double) / (1024.0 * 1024.0)
                << " MiB\n"
                << "localized DIIS backend    "
                << (use_block_sparse_ccsd
                        ? "Global Arrays selected blocks"
                        : localized_ccsd.diis_memory.global_arrays
                        ? "Global Arrays distributed"
                        : "replicated C++")
                << "\n"
                << "CCSD solver basis          "
                << ccsd_solver_basis_name(ccsd_solver_basis) << "\n"
                << "CCSD solver backend        "
                << ccsd_solver_backend_name(ccsd_solver_backend) << "\n"
                << "CCSD DIIS Gram backend     "
                << (use_incremental_ccsd_diis_gram
                        ? "incremental cached Gram"
                        : "recomputed equation oracle")
                << "\n"
                << "CCSD replicated DIIS       "
                << (use_root_replicated_ccsd_diis
                        ? "rank-zero owner plus amplitude broadcast"
                        : "redundant all-rank oracle")
                << "\n"
                << "CCSD replicated energy     "
                << (use_distributed_replicated_ccsd_energy
                        ? "distributed MPI/OpenMP scalar reduction"
                        : "redundant all-rank serial oracle")
                << "\n"
                << "CCSD kernel intermediates this run "
                << localized_ccsd.cumulative_timing.intermediates_seconds
                << " s\n"
                << "CCSD intermediate mixed-pair tiles "
                << localized_ccsd.cumulative_timing
                       .intermediate_mixed_pair_tasks
                << "\n"
                << "CCSD intermediate BLAS contractions "
                << localized_ccsd.cumulative_timing
                       .intermediate_blas_contractions
                << "\n"
                << "CCSD kernel singles this run "
                << localized_ccsd.cumulative_timing.singles_seconds << " s\n"
                << "CCSD kernel doubles this run "
                << localized_ccsd.cumulative_timing.doubles_seconds << " s\n"
                << "CCSD doubles occupied-pair tiles "
                << localized_ccsd.cumulative_timing
                       .doubles_occupied_pair_tasks
                << "\n"
                << "CCSD doubles BLAS contractions "
                << localized_ccsd.cumulative_timing
                       .doubles_blas_contractions
                << "\n"
                << "CCSD kernel W_abef this run "
                << localized_ccsd.cumulative_timing.wabef_seconds << " s\n"
                << "CCSD kernel W_abef build this run "
                << localized_ccsd.cumulative_timing.wabef_build_seconds
                << " s\n"
                << "CCSD kernel W_abef BLAS this run "
                << localized_ccsd.cumulative_timing.wabef_contraction_seconds
                << " s\n"
                << "CCSD kernel W_abef accumulation this run "
                << localized_ccsd.cumulative_timing
                       .wabef_accumulation_seconds
                << " s\n"
                << "CCSD W_abef external tiles "
                << localized_ccsd.cumulative_timing.wabef_external_tiles
                << "\n"
                << "CCSD W_abef internal panels "
                << localized_ccsd.cumulative_timing.wabef_internal_panels
                << "\n"
                << "CCSD W_abef output tasks "
                << localized_ccsd.cumulative_timing.wabef_output_tasks
                << "\n"
                << "CCSD W_abef BLAS contractions "
                << localized_ccsd.cumulative_timing.wabef_blas_contractions
                << "\n"
                << "CCSD W_abef factorized cached evaluations "
                << localized_ccsd.cumulative_timing
                       .wabef_factorized_cached_evaluations
                << "\n"
                << "CCSD kernel residual this run "
                << localized_ccsd.cumulative_timing.residual_seconds << " s\n"
                << "CCSD kernel DIIS this run "
                << localized_ccsd.cumulative_timing.diis_seconds << " s\n"
                << "CCSD DIIS error dot products "
                << localized_ccsd.cumulative_timing
                       .diis_error_dot_products
                << "\n"
                << "CCSD DIIS history-vector accumulations "
                << localized_ccsd.cumulative_timing
                       .diis_history_vector_accumulations
                << "\n"
                << "CCSD kernel energy this run "
                << localized_ccsd.cumulative_timing.energy_seconds << " s\n"
                << "CCSD kernel total this run "
                << localized_ccsd.cumulative_timing.iteration_seconds
                << " s\n";
        const auto monomer_count = atoms.size() / 3;
        for (std::size_t order = 1; order <= std::min<std::size_t>(4, monomer_count);
             ++order)
          std::cout << "CCSD M" << order << "                  "
                    << many_body_energy(*ccsd_partition, order) << " Eh\n";
        if (triples_partition)
          for (std::size_t order = 1;
               order <= std::min<std::size_t>(6, monomer_count); ++order)
            std::cout << "(T) M" << order << "                   "
                      << many_body_energy(*triples_partition, order) << " Eh\n";
        else
          std::cout
              << "(T) nM partition          pending exact streamed local "
                 "back-transformation\n";
        std::cout << std::scientific
                  << "CCSD partition error      "
                  << ccsd_partition->sum_error << " Eh\n"
                  << "time CCSD partition       "
                  << ccsd_partition_seconds << " s\n"
                  << "CCSD omitted bound        "
                  << ccsd_partition->omitted_absolute_bound << " Eh\n"
                  << "CCSD unassigned           "
                  << ccsd_partition->unassigned_energy << " Eh / "
                  << ccsd_partition->unassigned_terms << " terms\n"
                  << "CCSD retained/omitted     "
                  << ccsd_partition->retained_terms << " / "
                  << ccsd_partition->omitted_terms << " terms\n";
        if (triples_partition)
          std::cout << "(T) partition error        "
                    << triples_partition->sum_error << " Eh\n"
                    << "(T) omitted bound          "
                    << triples_partition->omitted_absolute_bound << " Eh\n"
                    << "(T) unassigned             "
                    << triples_partition->unassigned_energy << " Eh / "
                    << triples_partition->unassigned_terms << " terms\n";
        std::cout << std::fixed
                << "time localization          "
                << elapsed_seconds(localization_start, after_localization)
                << " s\n"
                << "time localized Hamiltonian "
                << elapsed_seconds(localized_hamiltonian_begin,
                                   after_localized_hamiltonian)
                << " s\n"
                << "time localized CCSD        "
                << canonical_solver_basis_seconds +
                       elapsed_seconds(after_localized_hamiltonian,
                                       after_localized_ccsd)
                << " s\n"
                << "time localized triples     "
                << elapsed_seconds(after_localized_ccsd,
                                   after_localized_triples)
                << " s\n";
        if (partition_prefix != "none" && triples_partition)
          std::cout << "Phase 12 energy CSV       " << partition_prefix
                    << ".energy.csv\n"
                    << "Phase 12 orbital CSV      " << partition_prefix
                    << ".orbitals.csv\n";
        else if (partition_prefix != "none")
          std::cout
              << "Phase 12 CSV status        not written: exact streamed "
                 "(T) nM partition is unfinished\n";
      }
    }

    const double wall_seconds = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();
    if (communicator.rank == 0)
      std::cout << std::fixed << "wall time                   " << wall_seconds
                << " s\n";
    return 0;
  } catch (const std::exception& error) {
    const auto communicator = parallel_runtime.world();
    if (communicator.rank == 0)
      std::cerr << "fatal: " << error.what() << '\n';
    if (communicator.size > 1)
      parallel_runtime.abort(1);
    return 1;
  }
}
