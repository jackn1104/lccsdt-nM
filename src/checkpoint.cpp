#include "localized_cc/checkpoint.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <type_traits>

namespace localized_cc {
namespace {

constexpr std::array<char, 8> magic{{'L', 'C', 'C', 'H', 'K', 'P', 'T', '2'}};
constexpr std::uint32_t format_version = 8;
constexpr std::uint32_t oldest_supported_format_version = 2;
constexpr std::uint64_t maximum_string_bytes = 1U << 20U;
constexpr std::uint64_t maximum_vector_elements = 1ULL << 34U;

template <typename T>
void write_scalar(std::ostream& out, const T value) {
  static_assert(std::is_arithmetic_v<T>);
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!out)
    throw std::runtime_error("failed while writing CCSD(T) checkpoint");
}

template <typename T>
T read_scalar(std::istream& in) {
  static_assert(std::is_arithmetic_v<T>);
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in)
    throw std::runtime_error("truncated CCSD(T) checkpoint");
  return value;
}

void write_string(std::ostream& out, const std::string& value) {
  write_scalar(out, static_cast<std::uint64_t>(value.size()));
  out.write(value.data(), static_cast<std::streamsize>(value.size()));
  if (!out)
    throw std::runtime_error("failed while writing checkpoint string");
}

std::string read_string(std::istream& in) {
  const auto count = read_scalar<std::uint64_t>(in);
  if (count > maximum_string_bytes)
    throw std::runtime_error("checkpoint string exceeds safety limit");
  std::string result(static_cast<std::size_t>(count), '\0');
  in.read(result.data(), static_cast<std::streamsize>(count));
  if (!in)
    throw std::runtime_error("truncated checkpoint string");
  return result;
}

template <typename T>
void write_vector(std::ostream& out, const std::vector<T>& values) {
  static_assert(std::is_arithmetic_v<T>);
  write_scalar(out, static_cast<std::uint64_t>(values.size()));
  if (!values.empty())
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!out)
    throw std::runtime_error("failed while writing checkpoint vector");
}

template <typename T>
std::vector<T> read_vector(std::istream& in) {
  static_assert(std::is_arithmetic_v<T>);
  const auto count = read_scalar<std::uint64_t>(in);
  if (count > maximum_vector_elements ||
      count > static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max() / sizeof(T)))
    throw std::runtime_error("checkpoint vector exceeds safety limit");
  std::vector<T> result(static_cast<std::size_t>(count));
  if (!result.empty())
    in.read(reinterpret_cast<char*>(result.data()),
            static_cast<std::streamsize>(result.size() * sizeof(T)));
  if (!in)
    throw std::runtime_error("truncated checkpoint vector");
  return result;
}

template <std::size_t Rank>
void write_tensor(std::ostream& out, const Tensor<Rank>& tensor) {
  for (const auto dimension : tensor.dimensions())
    write_scalar(out, static_cast<std::uint64_t>(dimension));
  write_vector(out, tensor.data());
}

template <std::size_t Rank>
Tensor<Rank> read_tensor(std::istream& in) {
  std::array<std::size_t, Rank> dimensions{};
  std::uint64_t expected = 1;
  for (auto& dimension : dimensions) {
    const auto raw = read_scalar<std::uint64_t>(in);
    if (raw == 0 || raw > std::numeric_limits<std::size_t>::max())
      throw std::runtime_error("invalid checkpoint tensor dimension");
    dimension = static_cast<std::size_t>(raw);
    if (expected > maximum_vector_elements / raw)
      throw std::runtime_error("checkpoint tensor dimensions overflow");
    expected *= raw;
  }
  auto values = read_vector<double>(in);
  if (values.size() != expected)
    throw std::runtime_error("checkpoint tensor element count mismatch");
  for (const auto value : values)
    if (!std::isfinite(value))
      throw std::runtime_error("checkpoint contains a non-finite tensor value");
  Tensor<Rank> result(dimensions);
  result.data() = std::move(values);
  return result;
}

template <typename State, typename Writer>
void write_optional(std::ostream& out, const std::optional<State>& value,
                    Writer&& writer) {
  write_scalar<std::uint8_t>(out, value.has_value() ? 1U : 0U);
  if (value)
    writer(out, *value);
}

template <typename State, typename Reader>
std::optional<State> read_optional(std::istream& in, Reader&& reader) {
  const auto present = read_scalar<std::uint8_t>(in);
  if (present > 1)
    throw std::runtime_error("invalid checkpoint optional-state flag");
  return present == 0 ? std::nullopt
                      : std::optional<State>{reader(in)};
}

void write_rhf(std::ostream& out, const RhfCheckpointState& state) {
  write_scalar(out, state.nuclear_repulsion_energy);
  write_scalar(out, state.electronic_energy);
  write_scalar(out, state.total_energy);
  write_tensor(out, state.overlap);
  write_tensor(out, state.core_hamiltonian);
  write_tensor(out, state.density);
  write_tensor(out, state.fock);
  write_tensor(out, state.coefficients);
  write_vector(out, state.orbital_energies);
  write_vector(out, state.occupations);
}

RhfCheckpointState read_rhf(std::istream& in) {
  RhfCheckpointState state;
  state.nuclear_repulsion_energy = read_scalar<double>(in);
  state.electronic_energy = read_scalar<double>(in);
  state.total_energy = read_scalar<double>(in);
  state.overlap = read_tensor<2>(in);
  state.core_hamiltonian = read_tensor<2>(in);
  state.density = read_tensor<2>(in);
  state.fock = read_tensor<2>(in);
  state.coefficients = read_tensor<2>(in);
  state.orbital_energies = read_vector<double>(in);
  state.occupations = read_vector<double>(in);
  return state;
}

void write_localization(std::ostream& out,
                        const LocalizationCheckpointState& state) {
  write_string(out, state.method);
  write_tensor(out, state.occupied_coefficients);
  write_tensor(out, state.virtual_coefficients);
  write_tensor(out, state.occupied_rotation);
  write_tensor(out, state.virtual_rotation);
  write_vector(out, state.occupied_monomers);
  write_vector(out, state.virtual_monomers);
  write_vector(out, state.occupied_confidence);
  write_vector(out, state.virtual_confidence);
  write_scalar(out, state.assignment_threshold);
}

LocalizationCheckpointState read_localization(std::istream& in) {
  LocalizationCheckpointState state;
  state.method = read_string(in);
  state.occupied_coefficients = read_tensor<2>(in);
  state.virtual_coefficients = read_tensor<2>(in);
  state.occupied_rotation = read_tensor<2>(in);
  state.virtual_rotation = read_tensor<2>(in);
  state.occupied_monomers = read_vector<int>(in);
  state.virtual_monomers = read_vector<int>(in);
  state.occupied_confidence = read_vector<double>(in);
  state.virtual_confidence = read_vector<double>(in);
  state.assignment_threshold = read_scalar<double>(in);
  return state;
}

void write_localization_optimization(
    std::ostream& out,
    const LocalizationOptimizationCheckpointState& state) {
  write_string(out, state.method);
  write_string(out, state.space);
  write_scalar<std::uint8_t>(out, state.converged ? 1U : 0U);
  write_tensor(out, state.rotation);
  write_scalar(out, state.trust_radius);
  write_scalar(out, static_cast<std::uint64_t>(state.completed_iterations));
  write_scalar(out, static_cast<std::uint64_t>(state.accepted_steps));
  write_scalar(out, static_cast<std::uint64_t>(state.rejected_steps));
  write_vector(out, state.objective_history);
}

LocalizationOptimizationCheckpointState read_localization_optimization(
    std::istream& in) {
  LocalizationOptimizationCheckpointState state;
  state.method = read_string(in);
  state.space = read_string(in);
  state.converged = read_scalar<std::uint8_t>(in) != 0;
  state.rotation = read_tensor<2>(in);
  state.trust_radius = read_scalar<double>(in);
  state.completed_iterations =
      static_cast<std::size_t>(read_scalar<std::uint64_t>(in));
  state.accepted_steps =
      static_cast<std::size_t>(read_scalar<std::uint64_t>(in));
  state.rejected_steps =
      static_cast<std::size_t>(read_scalar<std::uint64_t>(in));
  state.objective_history = read_vector<double>(in);
  if (!std::isfinite(state.trust_radius) || state.trust_radius <= 0.0 ||
      state.objective_history.empty())
    throw std::runtime_error(
        "checkpoint localization optimizer state is outside its domain");
  return state;
}

void write_ccsd(std::ostream& out, const CcsdCheckpointState& state) {
  write_scalar<std::uint8_t>(out, state.localized ? 1U : 0U);
  write_scalar<std::uint8_t>(out, state.converged ? 1U : 0U);
  write_scalar(out, static_cast<std::uint64_t>(state.completed_iterations));
  write_scalar(out, state.correlation_energy);
  write_scalar(out, state.r1_norm);
  write_scalar(out, state.r2_norm);
  write_tensor(out, state.t1);
  write_tensor(out, state.t2);
  write_scalar(out, static_cast<std::uint64_t>(state.history.size()));
  for (const auto& item : state.history) {
    write_scalar(out, static_cast<std::uint64_t>(item.iteration));
    write_scalar(out, item.energy);
    write_scalar(out, item.delta_energy);
    write_scalar(out, item.r1_norm);
    write_scalar(out, item.r2_norm);
  }
}

CcsdCheckpointState read_ccsd(std::istream& in) {
  CcsdCheckpointState state;
  state.localized = read_scalar<std::uint8_t>(in) != 0;
  state.converged = read_scalar<std::uint8_t>(in) != 0;
  state.completed_iterations =
      static_cast<std::size_t>(read_scalar<std::uint64_t>(in));
  state.correlation_energy = read_scalar<double>(in);
  state.r1_norm = read_scalar<double>(in);
  state.r2_norm = read_scalar<double>(in);
  state.t1 = read_tensor<2>(in);
  state.t2 = read_tensor<4>(in);
  const auto count = read_scalar<std::uint64_t>(in);
  if (count > maximum_vector_elements)
    throw std::runtime_error("checkpoint CCSD history exceeds safety limit");
  state.history.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t index = 0; index < count; ++index)
    state.history.push_back(
        {static_cast<std::size_t>(read_scalar<std::uint64_t>(in)),
         read_scalar<double>(in), read_scalar<double>(in),
         read_scalar<double>(in), read_scalar<double>(in)});
  return state;
}

void write_sparse_ccsd(std::ostream& out,
                       const SparseCcsdCheckpointState& state) {
  write_scalar<std::uint8_t>(out, state.converged ? 1U : 0U);
  write_scalar(out, static_cast<std::uint64_t>(state.completed_iterations));
  write_scalar(out, static_cast<std::uint64_t>(state.maximum_monomers));
  write_scalar(out, state.correlation_energy);
  write_scalar(out, state.r1_norm);
  write_scalar(out, state.r2_norm);
  write_tensor(out, state.t1);
  write_vector(out, state.occupied_monomers);
  write_vector(out, state.virtual_monomers);
  write_vector(out, state.t2_block_sizes);
  write_vector(out, state.t2_block_values);
  write_scalar(out, static_cast<std::uint64_t>(state.history.size()));
  for (const auto& item : state.history) {
    write_scalar(out, static_cast<std::uint64_t>(item.iteration));
    write_scalar(out, item.energy);
    write_scalar(out, item.delta_energy);
    write_scalar(out, item.r1_norm);
    write_scalar(out, item.r2_norm);
  }
}

SparseCcsdCheckpointState read_sparse_ccsd(std::istream& in) {
  SparseCcsdCheckpointState state;
  state.converged = read_scalar<std::uint8_t>(in) != 0;
  state.completed_iterations =
      static_cast<std::size_t>(read_scalar<std::uint64_t>(in));
  state.maximum_monomers =
      static_cast<std::size_t>(read_scalar<std::uint64_t>(in));
  state.correlation_energy = read_scalar<double>(in);
  state.r1_norm = read_scalar<double>(in);
  state.r2_norm = read_scalar<double>(in);
  state.t1 = read_tensor<2>(in);
  state.occupied_monomers = read_vector<int>(in);
  state.virtual_monomers = read_vector<int>(in);
  state.t2_block_sizes = read_vector<std::uint64_t>(in);
  state.t2_block_values = read_vector<double>(in);
  const auto count = read_scalar<std::uint64_t>(in);
  if (count > maximum_vector_elements)
    throw std::runtime_error(
        "checkpoint sparse CCSD history exceeds safety limit");
  state.history.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t index = 0; index < count; ++index)
    state.history.push_back(
        {static_cast<std::size_t>(read_scalar<std::uint64_t>(in)),
         read_scalar<double>(in), read_scalar<double>(in),
         read_scalar<double>(in), read_scalar<double>(in)});
  if (!std::isfinite(state.correlation_energy) ||
      !std::isfinite(state.r1_norm) || !std::isfinite(state.r2_norm) ||
      state.maximum_monomers == 0)
    throw std::runtime_error(
        "checkpoint sparse CCSD state is outside its numerical domain");
  std::uint64_t expected_values = 0;
  for (const auto size : state.t2_block_sizes) {
    if (size == 0 || size > maximum_vector_elements - expected_values)
      throw std::runtime_error(
          "checkpoint sparse CCSD block layout exceeds safety limit");
    expected_values += size;
  }
  if (expected_values != state.t2_block_values.size())
    throw std::runtime_error(
        "checkpoint sparse CCSD block/value count mismatch");
  for (const auto value : state.t2_block_values)
    if (!std::isfinite(value))
      throw std::runtime_error(
          "checkpoint sparse CCSD contains a non-finite T2 amplitude");
  return state;
}

void write_canonical_triples(
    std::ostream& out, const CanonicalTriplesCheckpointState& state) {
  write_vector(out, state.bracket_task_energies);
  write_vector(out, state.parenthetical_task_energies);
  write_vector(out, state.completed_tasks);
}

CanonicalTriplesCheckpointState read_canonical_triples(std::istream& in) {
  return {read_vector<double>(in), read_vector<double>(in),
          read_vector<std::uint8_t>(in)};
}

void write_laplace_triples(std::ostream& out,
                           const LaplaceTriplesCheckpointState& state) {
  write_scalar(out, static_cast<std::uint64_t>(state.quadrature_points));
  write_vector(out, state.point_many_body_energies);
  write_vector(out, state.completed_points);
  write_string(out, state.quadrature_scheme);
}

LaplaceTriplesCheckpointState read_laplace_triples(
    std::istream& in, const std::uint32_t file_version) {
  LaplaceTriplesCheckpointState state;
  state.quadrature_points =
      static_cast<std::size_t>(read_scalar<std::uint64_t>(in));
  state.point_many_body_energies = read_vector<double>(in);
  state.completed_points = read_vector<std::uint8_t>(in);
  if (file_version >= 8) state.quadrature_scheme = read_string(in);
  return state;
}

void write_localized_triples(
    std::ostream& out, const LocalizedTriplesCheckpointState& state) {
  write_scalar<std::uint8_t>(out, state.converged ? 1U : 0U);
  write_scalar(out, static_cast<std::uint64_t>(state.completed_iterations));
  write_scalar(out, state.energy);
  write_scalar(out, state.residual_norm);
  write_tensor(out, state.response);
  write_vector(out, state.residual_history);
}

LocalizedTriplesCheckpointState read_localized_triples(std::istream& in) {
  LocalizedTriplesCheckpointState state;
  state.converged = read_scalar<std::uint8_t>(in) != 0;
  state.completed_iterations =
      static_cast<std::size_t>(read_scalar<std::uint64_t>(in));
  state.energy = read_scalar<double>(in);
  state.residual_norm = read_scalar<double>(in);
  state.response = read_tensor<6>(in);
  state.residual_history = read_vector<double>(in);
  return state;
}

void write_basis_projection(
    std::ostream& out, const BasisProjectionCheckpointState& state) {
  write_string(out, state.source_checkpoint);
  write_string(out, state.source_fingerprint);
  write_string(out, state.source_basis);
  write_string(out, state.target_basis);
  write_scalar(out, static_cast<std::uint64_t>(state.source_ao_count));
  write_scalar(out, static_cast<std::uint64_t>(state.target_ao_count));
  write_scalar(out, static_cast<std::uint64_t>(state.target_metric_rank));
  write_scalar(out, state.minimum_projection_gram_eigenvalue);
  write_scalar(out, state.occupied_orthonormality_error);
  write_scalar(out, state.spin_summed_electron_count);
  write_scalar<std::uint8_t>(
      out, state.canonical_amplitudes_projected ? 1U : 0U);
  write_scalar<std::uint8_t>(out,
                             state.localization_seed_projected ? 1U : 0U);
  write_scalar<std::uint8_t>(
      out, state.localized_amplitudes_projected ? 1U : 0U);
}

BasisProjectionCheckpointState read_basis_projection(std::istream& in) {
  BasisProjectionCheckpointState state;
  state.source_checkpoint = read_string(in);
  state.source_fingerprint = read_string(in);
  state.source_basis = read_string(in);
  state.target_basis = read_string(in);
  state.source_ao_count =
      static_cast<std::size_t>(read_scalar<std::uint64_t>(in));
  state.target_ao_count =
      static_cast<std::size_t>(read_scalar<std::uint64_t>(in));
  state.target_metric_rank =
      static_cast<std::size_t>(read_scalar<std::uint64_t>(in));
  state.minimum_projection_gram_eigenvalue = read_scalar<double>(in);
  state.occupied_orthonormality_error = read_scalar<double>(in);
  state.spin_summed_electron_count = read_scalar<double>(in);
  state.canonical_amplitudes_projected =
      read_scalar<std::uint8_t>(in) != 0;
  state.localization_seed_projected = read_scalar<std::uint8_t>(in) != 0;
  state.localized_amplitudes_projected =
      read_scalar<std::uint8_t>(in) != 0;
  return state;
}

void write_file(const std::filesystem::path& path,
                const CalculationCheckpoint& checkpoint) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out)
    throw std::runtime_error("cannot create checkpoint: " + path.string());
  out.write(magic.data(), static_cast<std::streamsize>(magic.size()));
  write_scalar(out, format_version);
  write_scalar<std::uint8_t>(out,
                            std::endian::native == std::endian::little ? 1U : 2U);
  write_string(out, checkpoint.fingerprint);
  write_optional(out, checkpoint.rhf_restart_density,
                 [](std::ostream& stream, const Matrix& matrix) {
                   write_tensor(stream, matrix);
                 });
  write_optional(out, checkpoint.rhf, write_rhf);
  write_optional(out, checkpoint.localization, write_localization);
  write_optional(out, checkpoint.canonical_ccsd, write_ccsd);
  write_optional(out, checkpoint.canonical_triples, write_canonical_triples);
  write_optional(out, checkpoint.localized_ccsd, write_ccsd);
  write_optional(out, checkpoint.localized_triples, write_localized_triples);
  write_optional(out, checkpoint.basis_projection, write_basis_projection);
  write_optional(out, checkpoint.localized_streamed_triples,
                 write_canonical_triples);
  write_optional(out, checkpoint.occupied_localization_optimization,
                 write_localization_optimization);
  write_optional(out, checkpoint.virtual_localization_optimization,
                 write_localization_optimization);
  write_optional(out, checkpoint.localized_laplace_triples,
                 write_laplace_triples);
  write_optional(out, checkpoint.localized_sparse_ccsd,
                 write_sparse_ccsd);
  out.flush();
  if (!out)
    throw std::runtime_error("failed to flush checkpoint: " + path.string());
}

} // namespace

void validate_converged_ccsd_checkpoint(
    const CcsdCheckpointState& state, const bool expected_localized,
    const std::size_t occupied, const std::size_t virtuals,
    const double energy_tolerance, const double residual_tolerance,
    const CcsdResidualMetric residual_metric) {
  if (!std::isfinite(energy_tolerance) || energy_tolerance <= 0.0 ||
      !std::isfinite(residual_tolerance) || residual_tolerance <= 0.0)
    throw std::invalid_argument(
        "CCSD checkpoint tolerances must be finite and positive");
  if (state.localized != expected_localized)
    throw std::runtime_error("CCSD checkpoint orbital representation differs "
                             "from the requested calculation");
  if (!state.converged)
    throw std::runtime_error("CCSD checkpoint is not converged");
  if (occupied == 0 || virtuals == 0 ||
      state.t1.dimensions() !=
          std::array<std::size_t, 2>{occupied, virtuals} ||
      state.t2.dimensions() !=
          std::array<std::size_t, 4>{occupied, occupied, virtuals, virtuals})
    throw std::runtime_error(
        "converged CCSD checkpoint dimensions differ from the current spaces");
  if (state.completed_iterations == 0 || state.history.empty() ||
      state.history.back().iteration != state.completed_iterations)
    throw std::runtime_error(
        "converged CCSD checkpoint iteration history is inconsistent");
  if (!std::isfinite(state.correlation_energy) ||
      !std::isfinite(state.r1_norm) || !std::isfinite(state.r2_norm))
    throw std::runtime_error(
        "converged CCSD checkpoint contains a non-finite result");
  for (const auto value : state.t1.data())
    if (!std::isfinite(value))
      throw std::runtime_error(
          "converged CCSD checkpoint contains a non-finite T1 amplitude");
  for (const auto value : state.t2.data())
    if (!std::isfinite(value))
      throw std::runtime_error(
          "converged CCSD checkpoint contains a non-finite T2 amplitude");
  for (const auto& iteration : state.history)
    if (!std::isfinite(iteration.energy) ||
        !std::isfinite(iteration.delta_energy) ||
        !std::isfinite(iteration.r1_norm) ||
        !std::isfinite(iteration.r2_norm))
      throw std::runtime_error(
          "converged CCSD checkpoint contains non-finite history");
  const auto& final = state.history.back();
  const auto energy_scale =
      std::max(1.0, std::abs(state.correlation_energy));
  const auto storage_tolerance =
      32.0 * std::numeric_limits<double>::epsilon() * energy_scale;
  if (std::abs(state.correlation_energy - final.energy) > storage_tolerance)
    throw std::runtime_error(
        "converged CCSD checkpoint energy/history mismatch");
  const auto singles_elements = state.t1.size();
  const auto doubles_elements = state.t2.size();
  const auto state_r1_measure = ccsd_residual_measure(
      state.r1_norm, singles_elements, residual_metric);
  const auto state_r2_measure = ccsd_residual_measure(
      state.r2_norm, doubles_elements, residual_metric);
  const auto final_r1_measure = ccsd_residual_measure(
      final.r1_norm, singles_elements, residual_metric);
  const auto final_r2_measure = ccsd_residual_measure(
      final.r2_norm, doubles_elements, residual_metric);
  if (state.r1_norm < 0.0 || state.r2_norm < 0.0 ||
      final.r1_norm < 0.0 || final.r2_norm < 0.0)
    throw std::runtime_error("converged CCSD checkpoint has negative residual norms");
  if (t2_antisymmetry_error(state.t2) > 1.0e-10)
    throw std::runtime_error(
        "converged CCSD checkpoint T2 amplitudes are not antisymmetric");
  if (std::abs(final.delta_energy) >= energy_tolerance ||
      std::max(state_r1_measure, state_r2_measure) >= residual_tolerance ||
      std::max(final_r1_measure, final_r2_measure) >= residual_tolerance)
    throw CcsdCheckpointConvergenceError(
        "saved CCSD state does not satisfy the current convergence tolerances");
}

CcsdResult restore_converged_ccsd(
    const CcsdCheckpointState& state, const bool expected_localized,
    const std::size_t occupied, const std::size_t virtuals,
    const double energy_tolerance, const double residual_tolerance,
    const CcsdResidualMetric residual_metric,
    const CcsdAmplitudeBackend amplitude_backend,
    const CcsdWabefBackend wabef_backend) {
  validate_converged_ccsd_checkpoint(state, expected_localized, occupied,
      virtuals, energy_tolerance, residual_tolerance, residual_metric);
  CcsdResult result;
  result.correlation_energy = state.correlation_energy;
  result.t1 = state.t1;
  result.t2 = state.t2;
  result.r1_norm = state.r1_norm;
  result.r2_norm = state.r2_norm;
  result.history = state.history;
  result.converged = true;
  result.amplitude_backend = amplitude_backend;
  result.wabef_backend = wabef_backend;
  return result;
}

CalculationCheckpoint read_checkpoint(const std::filesystem::path& path,
                                      const std::string& expected_fingerprint) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot open checkpoint: " + path.string());
  std::array<char, magic.size()> file_magic{};
  in.read(file_magic.data(), static_cast<std::streamsize>(file_magic.size()));
  if (file_magic != magic)
    throw std::runtime_error("not a localized CCSD(T) checkpoint");
  const auto file_version = read_scalar<std::uint32_t>(in);
  if (file_version < oldest_supported_format_version ||
      file_version > format_version)
    throw std::runtime_error("unsupported CCSD(T) checkpoint version");
  const auto endian = read_scalar<std::uint8_t>(in);
  const auto native = std::endian::native == std::endian::little ? 1U : 2U;
  if (endian != native)
    throw std::runtime_error("checkpoint byte order differs from this host");
  CalculationCheckpoint result;
  result.fingerprint = read_string(in);
  if (!expected_fingerprint.empty() && result.fingerprint != expected_fingerprint)
    throw std::runtime_error(
        "checkpoint fingerprint mismatch (geometry/basis/core/method changed)");
  result.rhf_restart_density = read_optional<Matrix>(
      in, [](std::istream& stream) { return read_tensor<2>(stream); });
  result.rhf = read_optional<RhfCheckpointState>(in, read_rhf);
  result.localization =
      read_optional<LocalizationCheckpointState>(in, read_localization);
  result.canonical_ccsd = read_optional<CcsdCheckpointState>(in, read_ccsd);
  result.canonical_triples =
      read_optional<CanonicalTriplesCheckpointState>(in,
                                                     read_canonical_triples);
  result.localized_ccsd = read_optional<CcsdCheckpointState>(in, read_ccsd);
  result.localized_triples =
      read_optional<LocalizedTriplesCheckpointState>(in,
                                                     read_localized_triples);
  if (file_version >= 3)
    result.basis_projection =
        read_optional<BasisProjectionCheckpointState>(in,
                                                      read_basis_projection);
  if (file_version >= 4)
    result.localized_streamed_triples =
        read_optional<CanonicalTriplesCheckpointState>(
            in, read_canonical_triples);
  if (file_version >= 5) {
    result.occupied_localization_optimization =
        read_optional<LocalizationOptimizationCheckpointState>(
            in, read_localization_optimization);
    result.virtual_localization_optimization =
        read_optional<LocalizationOptimizationCheckpointState>(
            in, read_localization_optimization);
  }
  if (file_version >= 6)
    result.localized_laplace_triples =
        read_optional<LaplaceTriplesCheckpointState>(
            in, [file_version](std::istream& stream) {
              return read_laplace_triples(stream, file_version);
            });
  if (file_version >= 7)
    result.localized_sparse_ccsd =
        read_optional<SparseCcsdCheckpointState>(in, read_sparse_ccsd);
  if (in.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("checkpoint contains unexpected trailing data");
  return result;
}

void write_checkpoint_transactional(
    const std::filesystem::path& path,
    const CalculationCheckpoint& checkpoint) {
  if (checkpoint.fingerprint.empty())
    throw std::invalid_argument("checkpoint fingerprint may not be empty");
  const auto parent = path.parent_path();
  if (!parent.empty() && !std::filesystem::exists(parent))
    throw std::runtime_error("checkpoint directory does not exist: " +
                             parent.string());
  auto temporary = path;
  temporary += ".tmp";
  if (std::filesystem::exists(temporary))
    throw std::runtime_error("stale checkpoint temporary file exists: " +
                             temporary.string());
  try {
    write_file(temporary, checkpoint);
    static_cast<void>(read_checkpoint(temporary, checkpoint.fingerprint));
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
      // POSIX rename replaces atomically. Some filesystems require removal;
      // refusing that non-atomic fallback protects the previous checkpoint.
      throw std::runtime_error("atomic checkpoint replacement failed: " +
                               error.message());
    }
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

namespace {
std::uint64_t hamiltonian_checksum(const SpatialRhfHamiltonian& h) {
  std::uint64_t result=1469598103934665603ULL;
  auto word=[&](std::uint64_t value) {result=(result^value)*1099511628211ULL;};
  auto number=[&](double value) {
    if(!std::isfinite(value)) throw std::runtime_error("nonfinite Hamiltonian checkpoint value");
    word(std::bit_cast<std::uint64_t>(value));
  };
  word(h.nocc);word(h.nvirt);word(h.frozen_core);
  number(h.reference_energy);number(h.nuclear_energy);number(h.orthonormality_error);
  for(const auto* values:{&h.fock.data(),&h.hcore.data(),&h.eri_chemist.data()})
    for(double value:*values) number(value);
  for(const auto* indices:{&h.core_indices,&h.active_occ_indices,&h.virtual_indices}) {
    word(indices->size());for(auto index:*indices) word(index);
  }
  return result;
}
void read_matching_matrix(std::istream& in,const Matrix& expected) {
  for(auto dimension:expected.dimensions())
    if(read_scalar<std::uint64_t>(in)!=dimension)
      throw std::runtime_error("Hamiltonian checkpoint orbital/RHF dimension mismatch");
  if(read_scalar<std::uint64_t>(in)!=expected.size())
    throw std::runtime_error("Hamiltonian checkpoint orbital/RHF size mismatch");
  for(double value:expected.data())
    if(read_scalar<double>(in)!=value)
      throw std::runtime_error("Hamiltonian checkpoint orbital/RHF identity mismatch");
}
}
void write_hamiltonian_checkpoint(const std::filesystem::path& path,
    const std::string& contract,const Matrix& occupied,const Matrix& virtuals,
    const Matrix& rhf_fock,const SpatialRhfHamiltonian& h) {
  auto temporary=path;temporary+=".tmp";
  if(std::filesystem::exists(path)||std::filesystem::exists(temporary))
    throw std::runtime_error("refusing to overwrite Hamiltonian sidecar or temporary file: "+path.string());
  const auto checksum=hamiltonian_checksum(h);
  std::ofstream out(temporary,std::ios::binary);
  write_string(out,"LCCSD_LOCAL_HAMILTONIAN_v1");
  write_string(out,contract);
  write_tensor(out,occupied);write_tensor(out,virtuals);write_tensor(out,rhf_fock);
  write_scalar(out,static_cast<std::uint64_t>(h.nocc));
  write_scalar(out,static_cast<std::uint64_t>(h.nvirt));
  write_scalar(out,static_cast<std::uint64_t>(h.frozen_core));
  write_scalar(out,h.reference_energy);write_scalar(out,h.nuclear_energy);
  write_scalar(out,h.orthonormality_error);
  write_tensor(out,h.fock);write_tensor(out,h.hcore);
  write_vector(out,h.eri_chemist.data());
  write_vector(out,h.core_indices);write_vector(out,h.active_occ_indices);write_vector(out,h.virtual_indices);
  write_scalar(out,checksum);
  out.close();
  if(!out) throw std::runtime_error("Hamiltonian sidecar write failed; temporary file preserved");
  std::filesystem::rename(temporary,path);
}
SpatialRhfHamiltonian read_hamiltonian_checkpoint(const std::filesystem::path& path,
    const std::string& contract,const Matrix& occupied,const Matrix& virtuals,
    const Matrix& rhf_fock,std::size_t maximum_bytes) {
  std::ifstream in(path,std::ios::binary);
  if(read_string(in)!="LCCSD_LOCAL_HAMILTONIAN_v1" || read_string(in)!=contract)
    throw std::runtime_error("Hamiltonian sidecar format/contract mismatch");
  read_matching_matrix(in,occupied);read_matching_matrix(in,virtuals);read_matching_matrix(in,rhf_fock);
  SpatialRhfHamiltonian h;
  h.nocc=read_scalar<std::uint64_t>(in);h.nvirt=read_scalar<std::uint64_t>(in);
  h.frozen_core=read_scalar<std::uint64_t>(in);
  if(h.nocc!=occupied.dimensions()[1] || h.nvirt!=virtuals.dimensions()[1])
    throw std::runtime_error("Hamiltonian sidecar active-space mismatch");
  if(h.frozen_core>occupied.dimensions()[0])
    throw std::runtime_error("Hamiltonian sidecar frozen-core dimension mismatch");
  const auto n=h.nocc+h.nvirt;
  const auto elements=SymmetricEri4::stored_elements(n);
  if(elements>maximum_bytes/sizeof(double) || 2*n*n>maximum_bytes/sizeof(double)-elements)
    throw std::runtime_error("Hamiltonian sidecar exceeds memory allowance");
  h.reference_energy=read_scalar<double>(in);h.nuclear_energy=read_scalar<double>(in);
  h.orthonormality_error=read_scalar<double>(in);
  auto read_square=[&]() {
    if(read_scalar<std::uint64_t>(in)!=n || read_scalar<std::uint64_t>(in)!=n ||
       read_scalar<std::uint64_t>(in)!=n*n)
      throw std::runtime_error("Hamiltonian sidecar matrix dimension mismatch");
    Matrix result({n,n});
    in.read(reinterpret_cast<char*>(result.data().data()),static_cast<std::streamsize>(n*n*sizeof(double)));
    if(!in) throw std::runtime_error("truncated Hamiltonian sidecar matrix");
    return result;
  };
  h.fock=read_square();h.hcore=read_square();
  if(read_scalar<std::uint64_t>(in)!=elements)
    throw std::runtime_error("Hamiltonian sidecar ERI size mismatch");
  h.eri_chemist=SymmetricEri4(n);
  in.read(reinterpret_cast<char*>(h.eri_chemist.data().data()),static_cast<std::streamsize>(elements*sizeof(double)));
  if(!in) throw std::runtime_error("truncated Hamiltonian sidecar ERIs");
  auto read_indices=[&](std::size_t expected) {
    const auto count=read_scalar<std::uint64_t>(in);
    if(count!=expected) throw std::runtime_error("Hamiltonian sidecar index count mismatch");
    std::vector<std::size_t> indices(expected);
    for(auto& index:indices) {
      index=read_scalar<std::size_t>(in);
      if(index>=n+h.frozen_core) throw std::runtime_error("Hamiltonian sidecar index out of range");
    }
    return indices;
  };
  h.core_indices=read_indices(h.frozen_core);
  h.active_occ_indices=read_indices(h.nocc);h.virtual_indices=read_indices(h.nvirt);
  if(read_scalar<std::uint64_t>(in)!=hamiltonian_checksum(h) || in.peek()!=std::char_traits<char>::eof())
    throw std::runtime_error("Hamiltonian sidecar checksum/trailing-data failure");
  return h;
}

} // namespace localized_cc
