#include "localized_cc/partition.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace localized_cc {
namespace {

std::string csv_field(const std::string_view value) {
  if (value.find_first_of(",\"\n\r") == std::string_view::npos)
    return std::string(value);
  std::string escaped{"\""};
  for (const char character : value) {
    if (character == '"')
      escaped.push_back('"');
    escaped.push_back(character);
  }
  escaped.push_back('"');
  return escaped;
}

template <class Writer>
void write_new_file_transactionally(const std::filesystem::path& path,
                                    Writer&& writer) {
  if (path.empty())
    throw std::invalid_argument("CSV output path may not be empty");
  if (std::filesystem::exists(path))
    throw std::runtime_error("refusing to overwrite scientific CSV output: " +
                             path.string());
  const auto parent = path.parent_path();
  if (!parent.empty() && !std::filesystem::is_directory(parent))
    throw std::runtime_error("CSV output directory does not exist: " +
                             parent.string());
  auto temporary = path;
  temporary += ".tmp";
  if (std::filesystem::exists(temporary))
    throw std::runtime_error("stale CSV temporary file exists: " +
                             temporary.string());

  try {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output)
      throw std::runtime_error("cannot create CSV output: " +
                               temporary.string());
    output << std::setprecision(17);
    writer(output);
    output.flush();
    if (!output)
      throw std::runtime_error("failed while writing CSV output: " +
                               temporary.string());
    output.close();
    std::filesystem::rename(temporary, path);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

void write_partition_rows(std::ostream& output, const std::string_view component,
                          const EnergyPartition& partition) {
  const auto total_terms = partition.retained_terms + partition.omitted_terms;
  const auto summary = [&](const std::string_view name, const double value,
                           const std::size_t count) {
    output << component << ",summary," << name << ',' << value << ',' << count
           << '\n';
  };
  summary("parent_energy", partition.parent_energy, total_terms);
  summary("assigned_energy", partition.sum, partition.assigned_terms);
  summary("unassigned_energy", partition.unassigned_energy,
          partition.unassigned_terms);
  summary("retained_energy", partition.sum + partition.unassigned_energy,
          partition.retained_terms);
  summary("omitted_energy", partition.omitted_energy,
          partition.omitted_terms);
  summary("omitted_absolute_bound", partition.omitted_absolute_bound,
          partition.omitted_terms);
  summary("reconstruction_error", partition.sum_error, total_terms);
  summary("contribution_threshold", partition.contribution_threshold, 0);
  summary("maximum_omitted_absolute", partition.max_omitted_absolute, 0);

  for (const auto& [key, energy] : partition.many_body_classes) {
    const auto count = partition.many_body_term_counts.at(key);
    output << component << ",many_body," << csv_field(key) << ',' << energy
           << ',' << count << '\n';
  }
  for (const auto& [key, energy] : partition.classes) {
    const auto count = partition.class_term_counts.at(key);
    output << component << ",detailed," << csv_field(key) << ',' << energy
           << ',' << count << '\n';
  }
}

} // namespace

void write_partition_csv(const std::filesystem::path& path,
                         const EnergyPartition& ccsd,
                         const EnergyPartition& triples) {
  write_new_file_transactionally(path, [&](std::ostream& output) {
    output << "component,aggregation,class,energy_hartree,term_count\n";
    write_partition_rows(output, "CCSD", ccsd);
    write_partition_rows(output, "(T)", triples);
  });
}

void write_orbital_assignments_csv(
    const std::filesystem::path& path, const std::string& localizer,
    const double confidence_threshold,
    const std::vector<int>& occupied_labels,
    const std::vector<double>& occupied_confidence,
    const std::vector<int>& virtual_labels,
    const std::vector<double>& virtual_confidence) {
  if (!std::isfinite(confidence_threshold) || confidence_threshold <= 0.0 ||
      confidence_threshold > 1.0)
    throw std::invalid_argument("assignment confidence threshold must be in (0,1]");
  if (occupied_labels.size() != occupied_confidence.size() ||
      virtual_labels.size() != virtual_confidence.size())
    throw std::invalid_argument("orbital assignment label/confidence mismatch");
  write_new_file_transactionally(path, [&](std::ostream& output) {
    output << "localizer,space,spatial_orbital,monomer,confidence,"
              "confidence_threshold,passes_threshold\n";
    const auto rows = [&](const std::string_view space,
                          const std::vector<int>& labels,
                          const std::vector<double>& confidence) {
      for (std::size_t orbital = 0; orbital < labels.size(); ++orbital) {
        if (labels[orbital] < -1 || !std::isfinite(confidence[orbital]))
          throw std::invalid_argument("invalid orbital assignment value");
        const bool passes_threshold =
            confidence[orbital] >= confidence_threshold;
        if (passes_threshold != (labels[orbital] >= 0))
          throw std::invalid_argument(
              "orbital label is inconsistent with its confidence threshold");
        output << csv_field(localizer) << ',' << space << ',' << orbital << ','
               << (labels[orbital] < 0 ? "unassigned"
                                       : std::to_string(labels[orbital]))
               << ',' << confidence[orbital] << ','
               << confidence_threshold << ','
               << (passes_threshold ? "true" : "false")
               << '\n';
      }
    };
    rows("occupied", occupied_labels, occupied_confidence);
    rows("virtual", virtual_labels, virtual_confidence);
  });
}

} // namespace localized_cc
