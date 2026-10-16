#include "localized_cc/benchmark_output.hpp"

#include <cmath>
#include <sstream>
#include <string>

namespace localized_cc {
namespace {

void parse_value(const std::string& line, const std::string& label,
                 std::optional<double>& destination) {
  const auto position = line.find(label);
  if (position == std::string::npos)
    return;
  std::istringstream value_stream(line.substr(position + label.size()));
  double value = 0.0;
  if (value_stream >> value && std::isfinite(value))
    destination = value;
}

double elapsed_to_seconds(const std::string& value, bool& valid) {
  std::istringstream parts(value);
  std::string token;
  double components[3]{};
  std::size_t count = 0;
  while (std::getline(parts, token, ':') && count < 3) {
    std::istringstream number(token);
    if (!(number >> components[count]) || !std::isfinite(components[count])) {
      valid = false;
      return 0.0;
    }
    ++count;
  }
  valid = count == 2 || count == 3;
  if (!valid)
    return 0.0;
  return count == 2 ? 60.0 * components[0] + components[1]
                    : 3600.0 * components[0] + 60.0 * components[1] +
                          components[2];
}

} // namespace

CcsdtBenchmarkRecord parse_ccsdt_benchmark_output(std::istream& input) {
  CcsdtBenchmarkRecord result;
  std::string line;
  while (std::getline(input, line)) {
    parse_value(line, "RHF                       ", result.rhf_hartree);
    parse_value(line, "canonical CCSD corr       ",
                result.ccsd_correlation_hartree);
    parse_value(line, "canonical (T)             ",
                result.triples_parenthetical_hartree);
    parse_value(line, "canonical CCSD(T)         ", result.total_ccsdt_hartree);
    parse_value(line, "localized CCSD corr       ",
                result.ccsd_correlation_hartree);
    parse_value(line, "localized (T)             ",
                result.triples_parenthetical_hartree);
    parse_value(line, "localized CCSD(T)         ", result.total_ccsdt_hartree);
    parse_value(line, "wall time                   ", result.wall_seconds);
    parse_value(line, "Maximum resident set size (kbytes):",
                result.maximum_rss_kib);
    constexpr auto elapsed_label =
        "Elapsed (wall clock) time (h:mm:ss or m:ss):";
    const auto elapsed = line.find(elapsed_label);
    if (elapsed != std::string::npos && !result.wall_seconds) {
      bool valid = false;
      const auto seconds = elapsed_to_seconds(
          line.substr(elapsed + std::char_traits<char>::length(elapsed_label)),
          valid);
      if (valid)
        result.wall_seconds = seconds;
    }
  }
  return result;
}

} // namespace localized_cc
