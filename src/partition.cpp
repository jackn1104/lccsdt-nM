#include "localized_cc/partition.hpp"

#include "localized_cc/ccsd.hpp"
#include "localized_cc/selected_ccsd_storage.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace localized_cc {
namespace {

struct AuditedCcsdTerm {
  double contribution{};
  unsigned char many_body_order{};
};

std::set<int> monomers(const std::vector<int>& holes,
                       const std::vector<int>& particles) {
  std::set<int> result(holes.begin(), holes.end());
  result.insert(particles.begin(), particles.end());
  return result;
}

std::string many_body_key(const std::vector<int>& holes,
                          const std::vector<int>& particles) {
  return "M" + std::to_string(monomers(holes, particles).size());
}

std::string detailed_key(std::vector<int> holes,
                         std::vector<int> particles) {
  std::sort(holes.begin(), holes.end());
  std::sort(particles.begin(), particles.end());
  const auto participating_monomers = monomers(holes, particles);
  std::ostringstream stream;
  stream << 'M' << participating_monomers.size() << "|h:";
  for (const auto label : holes)
    stream << label << ',';
  stream << "|p:";
  for (const auto label : particles)
    stream << label << ',';
  stream << "|q:";
  int maximum_charge = 0;
  for (const auto label : participating_monomers) {
    const auto charge =
        static_cast<int>(std::count(holes.begin(), holes.end(), label)) -
        static_cast<int>(std::count(particles.begin(), particles.end(), label));
    maximum_charge = std::max(maximum_charge, std::abs(charge));
    stream << label << ':' << charge << ',';
  }
  stream << '|'
         << (maximum_charge == 0
                 ? "neutral"
                 : (maximum_charge == 1 ? "singly_ionic" : "doubly_ionic"));
  return stream.str();
}

void validate_screening(const PartitionScreeningOptions& screening) {
  if (!std::isfinite(screening.contribution_threshold) ||
      screening.contribution_threshold < 0.0)
    throw std::invalid_argument(
        "partition contribution threshold must be finite and nonnegative");
  if (!std::isfinite(screening.max_omitted_absolute) ||
      screening.max_omitted_absolute < 0.0)
    throw std::invalid_argument(
        "partition omitted-energy bound must be finite and nonnegative");
}

EnergyPartition make_partition(const PartitionScreeningOptions& screening) {
  validate_screening(screening);
  EnergyPartition result;
  result.contribution_threshold = screening.contribution_threshold;
  result.max_omitted_absolute = screening.max_omitted_absolute;
  return result;
}

void validate_labels(const std::vector<int>& labels) {
  if (std::any_of(labels.begin(), labels.end(),
                  [](const int label) { return label < -1; }))
    throw std::invalid_argument(
        "monomer labels must be nonnegative or -1 for unassigned");
}

void accumulate_contribution(EnergyPartition& partition,
                             const std::vector<int>& holes,
                             const std::vector<int>& particles,
                             const double contribution) {
  if (!std::isfinite(contribution))
    throw std::runtime_error("non-finite localized energy contribution");
  if (partition.contribution_threshold > 0.0 &&
      std::abs(contribution) <= partition.contribution_threshold) {
    partition.omitted_energy += contribution;
    partition.omitted_absolute_bound += std::abs(contribution);
    ++partition.omitted_terms;
    return;
  }
  ++partition.retained_terms;
  if (std::any_of(holes.begin(), holes.end(),
                  [](const int label) { return label < 0; }) ||
      std::any_of(particles.begin(), particles.end(),
                  [](const int label) { return label < 0; })) {
    partition.unassigned_energy += contribution;
    ++partition.unassigned_terms;
    return;
  }
  const auto detail = detailed_key(holes, particles);
  const auto order = many_body_key(holes, particles);
  partition.classes[detail] += contribution;
  ++partition.class_term_counts[detail];
  partition.many_body_classes[order] += contribution;
  ++partition.many_body_term_counts[order];
  ++partition.assigned_terms;
}

void merge_partition(EnergyPartition& destination,
                     const EnergyPartition& source) {
  for (const auto& [key, energy] : source.classes)
    destination.classes[key] += energy;
  for (const auto& [key, count] : source.class_term_counts)
    destination.class_term_counts[key] += count;
  for (const auto& [key, energy] : source.many_body_classes)
    destination.many_body_classes[key] += energy;
  for (const auto& [key, count] : source.many_body_term_counts)
    destination.many_body_term_counts[key] += count;
  destination.omitted_energy += source.omitted_energy;
  destination.omitted_absolute_bound += source.omitted_absolute_bound;
  destination.unassigned_energy += source.unassigned_energy;
  destination.assigned_terms += source.assigned_terms;
  destination.unassigned_terms += source.unassigned_terms;
  destination.retained_terms += source.retained_terms;
  destination.omitted_terms += source.omitted_terms;
}

void finalize_partition(EnergyPartition& partition, const double parent_energy) {
  partition.parent_energy = parent_energy;
  partition.sum = 0.0;
  double detailed_absolute_sum = 0.0;
  for (const auto& [key, energy] : partition.classes) {
    static_cast<void>(key);
    partition.sum += energy;
    detailed_absolute_sum += std::abs(energy);
  }
  double many_body_sum = 0.0;
  double many_body_absolute_sum = 0.0;
  for (const auto& [key, energy] : partition.many_body_classes) {
    static_cast<void>(key);
    many_body_sum += energy;
    many_body_absolute_sum += std::abs(energy);
  }
  const auto class_rounding_allowance =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max({1.0, detailed_absolute_sum, many_body_absolute_sum});
  if (std::abs(many_body_sum - partition.sum) > class_rounding_allowance)
    throw std::runtime_error(
        "detailed and many-body localized partitions do not reconstruct");
  const auto detailed_count = std::accumulate(
      partition.class_term_counts.begin(), partition.class_term_counts.end(),
      std::size_t{0}, [](const std::size_t total, const auto& entry) {
        return total + entry.second;
      });
  const auto many_body_count = std::accumulate(
      partition.many_body_term_counts.begin(),
      partition.many_body_term_counts.end(), std::size_t{0},
      [](const std::size_t total, const auto& entry) {
        return total + entry.second;
      });
  if (detailed_count != partition.assigned_terms ||
      many_body_count != partition.assigned_terms ||
      partition.retained_terms !=
          partition.assigned_terms + partition.unassigned_terms)
    throw std::runtime_error("localized partition term counts do not reconstruct");
  partition.sum_error =
      partition.sum + partition.unassigned_energy +
      partition.omitted_energy - partition.parent_energy;
  const auto rounding_allowance =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, partition.omitted_absolute_bound);
  if (partition.omitted_absolute_bound >
      partition.max_omitted_absolute + rounding_allowance) {
    std::ostringstream message;
    message << "localized contribution screen omitted an absolute bound of "
            << partition.omitted_absolute_bound << " Eh, exceeding the "
            << partition.max_omitted_absolute << " Eh limit";
    throw std::runtime_error(message.str());
  }
  if (std::abs(partition.parent_energy -
               (partition.sum + partition.unassigned_energy)) >
      partition.omitted_absolute_bound + std::abs(partition.sum_error) +
          rounding_allowance)
    throw std::runtime_error(
        "localized partition violates its certified omitted-energy bound");
}

std::size_t thread_count() {
#if defined(_OPENMP)
  return static_cast<std::size_t>(omp_get_max_threads());
#else
  return 1;
#endif
}

std::size_t current_thread() {
#if defined(_OPENMP)
  return static_cast<std::size_t>(omp_get_thread_num());
#else
  return 0;
#endif
}

std::size_t confident_monomer_span(
    const int occupied_i, const int occupied_j,
    const int virtual_a, const int virtual_b) {
  if (occupied_i < 0 || occupied_j < 0 || virtual_a < 0 || virtual_b < 0)
    throw std::invalid_argument(
        "CCSD energy-term audit requires confident monomer labels");
  const std::array<int, 4> labels{
      occupied_i, occupied_j, virtual_a, virtual_b};
  std::array<int, 4> unique{};
  std::size_t count = 0;
  for (const int label : labels) {
    if (std::find(unique.begin(), unique.begin() +
                                    static_cast<std::ptrdiff_t>(count),
                  label) ==
        unique.begin() + static_cast<std::ptrdiff_t>(count))
      unique[count++] = label;
  }
  return count;
}

CcsdEnergyTermAuditRow make_audit_row(
    const std::string& criterion, const double tolerance,
    const double maximum_term, const double signed_energy,
    const double absolute_bound, const std::size_t omitted_terms,
    const std::array<std::size_t, 4>& omitted_many_body_terms) {
  return CcsdEnergyTermAuditRow{
      criterion, tolerance, maximum_term, signed_energy, absolute_bound,
      omitted_terms, omitted_many_body_terms};
}

} // namespace

CcsdEnergyTermAudit audit_ccsd_energy_terms(
    const SpatialRhfHamiltonian& hamiltonian, const Matrix& t1,
    const Tensor4& t2, const std::vector<int>& occupied_labels,
    const std::vector<int>& virtual_labels, const double tolerance_hartree) {
  if (!(tolerance_hartree >= 0.0) || !std::isfinite(tolerance_hartree))
    throw std::invalid_argument(
        "CCSD energy-term audit tolerance must be finite and nonnegative");
  const auto occupied = 2 * hamiltonian.nocc;
  const auto virtuals = 2 * hamiltonian.nvirt;
  if (t1.dimensions() != std::array<std::size_t, 2>{occupied, virtuals} ||
      t2.dimensions() !=
          std::array<std::size_t, 4>{occupied, occupied, virtuals, virtuals} ||
      occupied_labels.size() != occupied ||
      virtual_labels.size() != virtuals)
    throw std::invalid_argument(
        "CCSD energy-term audit dimensions are inconsistent");
  validate_labels(occupied_labels);
  validate_labels(virtual_labels);
  if (std::any_of(occupied_labels.begin(), occupied_labels.end(),
                  [](const int label) { return label < 0; }) ||
      std::any_of(virtual_labels.begin(), virtual_labels.end(),
                  [](const int label) { return label < 0; }))
    throw std::invalid_argument(
        "CCSD energy-term audit requires confident monomer labels");

  const SpinOrbitalEriView eri(hamiltonian);
  const auto spin_fock =
      expand_spatial_one_body_to_spin_orbital(hamiltonian.fock);
  CcsdEnergyTermAudit audit;
  audit.tolerance_hartree = tolerance_hartree;
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t a = 0; a < virtuals; ++a)
      audit.singles_energy_hartree +=
          spin_fock(i, occupied + a) * t1(i, a);

  audit.total_doubles_terms = t2.size();
  if (audit.total_doubles_terms >
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::overflow_error(
        "CCSD energy-term audit extent exceeds ptrdiff_t");
  std::vector<AuditedCcsdTerm> terms(audit.total_doubles_terms);
  std::vector<int> nonfinite(thread_count(), 0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t raw = 0;
       raw < static_cast<std::ptrdiff_t>(audit.total_doubles_terms); ++raw) {
    auto flat = static_cast<std::size_t>(raw);
    const auto b = flat % virtuals;
    flat /= virtuals;
    const auto a = flat % virtuals;
    flat /= virtuals;
    const auto j = flat % occupied;
    const auto i = flat / occupied;
    const double integral = eri(i, j, occupied + a, occupied + b);
    const double contribution =
        0.25 * integral * t2(i, j, a, b) +
        0.5 * integral * t1(i, a) * t1(j, b);
    const auto order = confident_monomer_span(
        occupied_labels[i], occupied_labels[j], virtual_labels[a],
        virtual_labels[b]);
    if (!std::isfinite(contribution) || order == 0 || order > 4) {
      nonfinite[current_thread()] = 1;
      continue;
    }
    terms[static_cast<std::size_t>(raw)] =
        AuditedCcsdTerm{contribution, static_cast<unsigned char>(order)};
  }
  if (std::any_of(nonfinite.begin(), nonfinite.end(),
                  [](const int value) { return value != 0; }))
    throw std::runtime_error(
        "CCSD energy-term audit produced invalid contributions");

  std::sort(terms.begin(), terms.end(),
            [](const AuditedCcsdTerm& left,
               const AuditedCcsdTerm& right) {
              const double left_magnitude = std::abs(left.contribution);
              const double right_magnitude = std::abs(right.contribution);
              if (left_magnitude != right_magnitude)
                return left_magnitude < right_magnitude;
              if (left.contribution != right.contribution)
                return left.contribution < right.contribution;
              return left.many_body_order < right.many_body_order;
            });
  audit.exact_zero_terms = static_cast<std::size_t>(std::count_if(
      terms.begin(), terms.end(), [](const AuditedCcsdTerm& term) {
        return term.contribution == 0.0;
      }));

  double signed_sum = 0.0;
  double absolute_sum = 0.0;
  std::array<std::size_t, 4> counts{};
  CcsdEnergyTermAuditRow certified = make_audit_row(
      "certified_absolute", tolerance_hartree, 0.0, 0.0, 0.0, 0, {});
  CcsdEnergyTermAuditRow observed = make_audit_row(
      "observed_signed", tolerance_hartree, 0.0, 0.0, 0.0, 0, {});
  std::size_t begin = 0;
  while (begin < terms.size()) {
    const double magnitude = std::abs(terms[begin].contribution);
    std::size_t end = begin;
    while (end < terms.size() &&
           std::abs(terms[end].contribution) == magnitude) {
      signed_sum += terms[end].contribution;
      absolute_sum += std::abs(terms[end].contribution);
      ++counts[static_cast<std::size_t>(terms[end].many_body_order - 1U)];
      ++end;
    }
    if (absolute_sum <= tolerance_hartree)
      certified = make_audit_row(
          "certified_absolute", tolerance_hartree, magnitude, signed_sum,
          absolute_sum, end, counts);
    if (std::abs(signed_sum) <= tolerance_hartree)
      observed = make_audit_row(
          "observed_signed", tolerance_hartree, magnitude, signed_sum,
          absolute_sum, end, counts);
    begin = end;
  }
  for (const auto& term : terms)
    audit.doubles_energy_hartree += term.contribution;
  audit.rows = {certified, observed};
  return audit;
}

void write_ccsd_energy_term_audit_csv(
    const std::filesystem::path& path,
    const CcsdEnergyTermAudit& audit) {
  if (std::filesystem::exists(path))
    throw std::runtime_error(
        "refusing to overwrite existing CCSD energy-term audit CSV");
  std::ofstream output(path, std::ios::out | std::ios::trunc);
  if (!output)
    throw std::runtime_error("unable to create CCSD energy-term audit CSV");
  output << "criterion,tolerance_hartree,maximum_omitted_term_hartree,"
            "omitted_signed_energy_hartree,omitted_absolute_bound_hartree,"
            "omitted_terms,total_terms,omitted_percent,M1_terms,M2_terms,"
            "M3_terms,M4_terms,exact_zero_terms,singles_energy_hartree,"
            "doubles_energy_hartree,reconstructed_ccsd_energy_hartree\n";
  output << std::setprecision(17);
  for (const auto& row : audit.rows) {
    const double percentage = audit.total_doubles_terms == 0
                                  ? 0.0
                                  : 100.0 * static_cast<double>(row.omitted_terms) /
                                        static_cast<double>(audit.total_doubles_terms);
    output << row.criterion << ',' << row.tolerance_hartree << ','
           << row.maximum_omitted_term_hartree << ','
           << row.omitted_signed_energy_hartree << ','
           << row.omitted_absolute_bound_hartree << ','
           << row.omitted_terms << ',' << audit.total_doubles_terms << ','
           << percentage;
    for (const auto count : row.omitted_many_body_terms)
      output << ',' << count;
    output << ',' << audit.exact_zero_terms << ','
           << audit.singles_energy_hartree << ','
           << audit.doubles_energy_hartree << ','
           << audit.singles_energy_hartree + audit.doubles_energy_hartree
           << '\n';
  }
  if (!output)
    throw std::runtime_error("failed while writing CCSD energy-term audit CSV");
}

std::vector<int> expand_spatial_labels(const std::vector<int>& labels) {
  std::vector<int> result;
  result.reserve(2 * labels.size());
  for (const auto label : labels) {
    if (label < -1)
      throw std::invalid_argument("monomer label must be -1 or nonnegative");
    result.push_back(label);
    result.push_back(label);
  }
  return result;
}

template <typename Eri, typename T2>
EnergyPartition partition_ccsd_impl(
    const Matrix& fock, const Eri& eri, const Matrix& t1,
    const T2& t2, const std::size_t nocc,
    const std::vector<int>& occupied_labels,
    const std::vector<int>& virtual_labels,
    const PartitionScreeningOptions& screening) {
  const auto nvirt = t1.dimensions()[1];
  if (occupied_labels.size() != nocc || virtual_labels.size() != nvirt)
    throw std::invalid_argument("CCSD label count mismatch");
  validate_labels(occupied_labels);
  validate_labels(virtual_labels);
  auto result = make_partition(screening);
  for (std::size_t i = 0; i < nocc; ++i) {
    for (std::size_t a = 0; a < nvirt; ++a) {
      const std::vector<int> holes{occupied_labels[i]};
      const std::vector<int> particles{virtual_labels[a]};
      const double contribution = fock(i, nocc + a) * t1(i, a);
      result.singles_fia_t1 += contribution;
      accumulate_contribution(result, holes, particles, contribution);
    }
  }

  const auto count = t2.size();
  if (count >
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::overflow_error("CCSD partition extent exceeds ptrdiff_t");
  std::vector<EnergyPartition> local;
  local.reserve(thread_count());
  for (std::size_t thread = 0; thread < thread_count(); ++thread)
    local.push_back(make_partition(screening));
  // Aggregate numeric label tuples before formatting detailed CSV keys.
  // Storage scales with encountered monomer patterns, not orbital quartets.
  struct Bin { double energy{}; std::size_t terms{}; };
  struct alignas(64) ThreadBins {
    std::map<std::array<int, 4>, Bin> assigned;
    double parent{};
    bool nonfinite{};
  };
  std::vector<ThreadBins> bins(thread_count());
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t raw = 0; raw < static_cast<std::ptrdiff_t>(count);
       ++raw) {
    auto flat = static_cast<std::size_t>(raw);
    const auto b = flat % nvirt;
    flat /= nvirt;
    const auto a = flat % nvirt;
    flat /= nvirt;
    const auto j = flat % nocc;
    const auto i = flat / nocc;
    const double integral = eri(i, j, nocc + a, nocc + b);
    const double contribution =
        0.25 * integral * t2(i, j, a, b) +
        0.5 * integral * t1(i, a) * t1(j, b);
    const auto thread = current_thread();
    auto& bucket = bins[thread];
    auto& partition = local[thread];
    if (!std::isfinite(contribution)) {
      bucket.nonfinite = true;
      continue;
    }
    // Independent total includes screened and unassigned terms as before.
    bucket.parent += contribution;
    if (partition.contribution_threshold > 0.0 &&
        std::abs(contribution) <= partition.contribution_threshold) {
      partition.omitted_energy += contribution;
      partition.omitted_absolute_bound += std::abs(contribution);
      ++partition.omitted_terms;
      continue;
    }
    ++partition.retained_terms;
    std::array<int, 4> key{occupied_labels[i], occupied_labels[j],
                           virtual_labels[a], virtual_labels[b]};
    if (std::any_of(key.begin(), key.end(), [](int label) { return label < 0; })) {
      partition.unassigned_energy += contribution;
      ++partition.unassigned_terms;
      continue;
    }
    if (key[0] > key[1]) std::swap(key[0], key[1]);
    if (key[2] > key[3]) std::swap(key[2], key[3]);
    auto& bin = bucket.assigned[key];
    bin.energy += contribution;
    ++bin.terms;
    ++partition.assigned_terms;
  }
  if (std::any_of(bins.begin(), bins.end(),
                  [](const ThreadBins& value) { return value.nonfinite; }))
    throw std::runtime_error("non-finite localized CCSD energy contribution");
  double parent_energy = result.singles_fia_t1;
  for (std::size_t thread = 0; thread < bins.size(); ++thread) {
    parent_energy += bins[thread].parent;
    for (const auto& [key, bin] : bins[thread].assigned) {
      const std::vector<int> holes{key[0], key[1]};
      const std::vector<int> particles{key[2], key[3]};
      const auto detail = detailed_key(holes, particles);
      const auto order = many_body_key(holes, particles);
      local[thread].classes[detail] += bin.energy;
      local[thread].class_term_counts[detail] += bin.terms;
      local[thread].many_body_classes[order] += bin.energy;
      local[thread].many_body_term_counts[order] += bin.terms;
    }
  }
  for (const auto& thread_partition : local)
    merge_partition(result, thread_partition);
  finalize_partition(result, parent_energy);
  return result;
}

EnergyPartition partition_ccsd(
    const Matrix& fock, const Tensor4& eri, const Matrix& t1,
    const Tensor4& t2, const std::size_t nocc,
    const std::vector<int>& occupied_labels,
    const std::vector<int>& virtual_labels,
    const PartitionScreeningOptions& screening) {
  return partition_ccsd_impl(fock, eri, t1, t2, nocc, occupied_labels,
                             virtual_labels, screening);
}

EnergyPartition partition_ccsd(
    const Matrix& spatial_fock, const SelectedIntegralStore& eri,
    const Matrix& t1, const Tensor4& t2, const std::size_t spatial_nocc,
    const std::vector<int>& occupied_labels,
    const std::vector<int>& virtual_labels,
    const PartitionScreeningOptions& screening) {
  const auto spin_fock = expand_spatial_one_body_to_spin_orbital(spatial_fock);
  return partition_ccsd_impl(spin_fock, eri, t1, t2, 2 * spatial_nocc,
                             occupied_labels, virtual_labels, screening);
}

EnergyPartition partition_ccsd(
    const SpatialRhfHamiltonian& hamiltonian, const Matrix& t1,
    const Tensor4& t2, const std::vector<int>& occupied_labels,
    const std::vector<int>& virtual_labels,
    const PartitionScreeningOptions& screening) {
  const auto spin_fock =
      expand_spatial_one_body_to_spin_orbital(hamiltonian.fock);
  const SpinOrbitalEriView spin_eri(hamiltonian);
  return partition_ccsd_impl(spin_fock, spin_eri, t1, t2,
                             2 * hamiltonian.nocc, occupied_labels,
                             virtual_labels, screening);
}

EnergyPartition partition_ccsd(
    const SpatialRhfHamiltonian& hamiltonian, const Matrix& t1,
    const SelectedDoublesView& t2,
    const std::vector<int>& occupied_labels,
    const std::vector<int>& virtual_labels,
    const PartitionScreeningOptions& screening) {
  const auto spin_fock =
      expand_spatial_one_body_to_spin_orbital(hamiltonian.fock);
  const SpinOrbitalEriView spin_eri(hamiltonian);
  return partition_ccsd_impl(spin_fock, spin_eri, t1, t2,
                             2 * hamiltonian.nocc, occupied_labels,
                             virtual_labels, screening);
}

EnergyPartition partition_triples(
    const Tensor6& energy_contribution,
    const std::vector<int>& occupied_labels,
    const std::vector<int>& virtual_labels,
    const PartitionScreeningOptions& screening) {
  const auto dimensions = energy_contribution.dimensions();
  if (occupied_labels.size() != dimensions[0] ||
      virtual_labels.size() != dimensions[3])
    throw std::invalid_argument("triples label count mismatch");
  validate_labels(occupied_labels);
  validate_labels(virtual_labels);
  const auto count = energy_contribution.size();
  if (count >
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::overflow_error("triples partition extent exceeds ptrdiff_t");
  std::vector<EnergyPartition> local;
  local.reserve(thread_count());
  for (std::size_t thread = 0; thread < thread_count(); ++thread)
    local.push_back(make_partition(screening));
  std::vector<int> nonfinite_by_thread(thread_count(), 0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t raw = 0; raw < static_cast<std::ptrdiff_t>(count);
       ++raw) {
    auto flat = static_cast<std::size_t>(raw);
    const auto c = flat % dimensions[5];
    flat /= dimensions[5];
    const auto b = flat % dimensions[4];
    flat /= dimensions[4];
    const auto a = flat % dimensions[3];
    flat /= dimensions[3];
    const auto k = flat % dimensions[2];
    flat /= dimensions[2];
    const auto j = flat % dimensions[1];
    const auto i = flat / dimensions[1];
    const std::vector<int> holes{occupied_labels[i], occupied_labels[j],
                                 occupied_labels[k]};
    const std::vector<int> particles{virtual_labels[a], virtual_labels[b],
                                     virtual_labels[c]};
    const auto contribution = energy_contribution(i, j, k, a, b, c);
    const auto thread = current_thread();
    if (!std::isfinite(contribution)) {
      nonfinite_by_thread[thread] = 1;
      continue;
    }
    accumulate_contribution(local[thread], holes, particles, contribution);
  }

  if (std::any_of(nonfinite_by_thread.begin(), nonfinite_by_thread.end(),
                  [](const int value) { return value != 0; }))
    throw std::runtime_error("non-finite localized triples energy contribution");

  auto result = make_partition(screening);
  for (const auto& thread_partition : local)
    merge_partition(result, thread_partition);
  const auto parent = std::accumulate(energy_contribution.data().begin(),
                                      energy_contribution.data().end(), 0.0);
  finalize_partition(result, parent);
  return result;
}

EnergyPartition partition_triples_single_monomer(
    const double parent_energy, const std::vector<int>& occupied_labels,
    const std::vector<int>& virtual_labels,
    const PartitionScreeningOptions& screening) {
  validate_screening(screening);
  validate_labels(occupied_labels);
  validate_labels(virtual_labels);
  if (!std::isfinite(parent_energy))
    throw std::invalid_argument(
        "single-monomer triples energy must be finite");
  if (occupied_labels.empty() || virtual_labels.empty())
    throw std::invalid_argument(
        "single-monomer triples partition requires orbital labels");
  if (screening.contribution_threshold != 0.0)
    throw std::invalid_argument(
        "single-monomer streamed triples partition cannot apply a "
        "term-level contribution threshold");
  const auto label = occupied_labels.front();
  if (label < 0 ||
      std::any_of(occupied_labels.begin(), occupied_labels.end(),
                  [label](const int value) { return value != label; }) ||
      std::any_of(virtual_labels.begin(), virtual_labels.end(),
                  [label](const int value) { return value != label; }))
    throw std::invalid_argument(
        "single-monomer triples shortcut requires one confident common label");
  auto result = make_partition(screening);
  accumulate_contribution(result, {label, label, label},
                          {label, label, label}, parent_energy);
  finalize_partition(result, parent_energy);
  return result;
}

EnergyPartition partition_triples_many_body_sums(
    const double parent_energy,
    const std::vector<double>& many_body_energies,
    const std::vector<std::size_t>& many_body_terms,
    const PartitionScreeningOptions& screening) {
  validate_screening(screening);
  if (!std::isfinite(parent_energy) || many_body_energies.size() != 6 ||
      many_body_terms.size() != 6)
    throw std::invalid_argument(
        "streamed triples partition requires finite M1--M6 data");
  if (screening.contribution_threshold != 0.0)
    throw std::invalid_argument(
        "streamed triples many-body sums cannot apply a post-contraction "
        "term threshold");
  auto result = make_partition(screening);
  for (std::size_t order = 0; order < 6; ++order) {
    if (!std::isfinite(many_body_energies[order]))
      throw std::invalid_argument(
          "streamed triples partition contains a non-finite energy");
    const auto key = "M" + std::to_string(order + 1);
    const auto detail = key + "|streamed_local_external_indices";
    result.many_body_classes[key] = many_body_energies[order];
    result.many_body_term_counts[key] = many_body_terms[order];
    result.classes[detail] = many_body_energies[order];
    result.class_term_counts[detail] = many_body_terms[order];
    result.assigned_terms += many_body_terms[order];
  }
  result.retained_terms = result.assigned_terms;
  finalize_partition(result, parent_energy);
  return result;
}

} // namespace localized_cc
