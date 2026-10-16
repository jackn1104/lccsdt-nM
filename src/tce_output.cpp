#include "localized_cc/tce_output.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

namespace localized_cc {
namespace {

struct LocatedEnergy {
  std::size_t line{};
  double value{};
};

double value_after_equals(const std::string& line, std::size_t line_number) {
  const auto equals = line.find('=');
  if (equals == std::string::npos) {
    throw std::runtime_error("missing '=' on TCE energy line " +
                             std::to_string(line_number));
  }
  std::size_t parsed = 0;
  const std::string text = line.substr(equals + 1);
  const double value = std::stod(text, &parsed);
  if (parsed == 0 || !std::isfinite(value)) {
    throw std::runtime_error("invalid TCE energy on line " +
                             std::to_string(line_number));
  }
  return value;
}

} // namespace

std::vector<TceEnergyRecord> parse_tce_energy_records(std::istream& input) {
  constexpr const char* scf_marker = "Total SCF energy =";
  constexpr const char* ccsd_marker =
      "CCSD correlation energy / hartree =";
  constexpr const char* triples_marker =
      "CCSD(T)  correction energy / hartree =";

  std::vector<TceEnergyRecord> records;
  std::optional<LocatedEnergy> scf;
  std::optional<LocatedEnergy> ccsd;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.find(scf_marker) != std::string::npos) {
      scf = LocatedEnergy{line_number,
                          value_after_equals(line, line_number)};
      ccsd.reset();
      continue;
    }
    if (line.find(ccsd_marker) != std::string::npos) {
      if (!scf) {
        throw std::runtime_error("CCSD correlation precedes SCF energy on line " +
                                 std::to_string(line_number));
      }
      ccsd = LocatedEnergy{line_number,
                           value_after_equals(line, line_number)};
      continue;
    }
    // The parentheses in this exact marker deliberately exclude CCSD[T].
    if (line.find(triples_marker) != std::string::npos) {
      if (!scf || !ccsd) {
        throw std::runtime_error(
            "CCSD(T) correction lacks its SCF/CCSD parent on line " +
            std::to_string(line_number));
      }
      records.push_back(TceEnergyRecord{
          scf->line, ccsd->line, line_number, scf->value, ccsd->value,
          value_after_equals(line, line_number)});
    }
  }
  return records;
}

} // namespace localized_cc
