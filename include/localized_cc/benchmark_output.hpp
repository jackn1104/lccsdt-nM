#pragma once

#include <istream>
#include <optional>

namespace localized_cc {

struct CcsdtBenchmarkRecord {
  std::optional<double> rhf_hartree;
  std::optional<double> ccsd_correlation_hartree;
  std::optional<double> triples_parenthetical_hartree;
  std::optional<double> total_ccsdt_hartree;
  std::optional<double> wall_seconds;
  std::optional<double> maximum_rss_kib;
};

// Parse the stable labels emitted by localized_ccsd_t and /usr/bin/time -v.
// Missing fields remain empty; they are never inferred.
CcsdtBenchmarkRecord parse_ccsdt_benchmark_output(std::istream& input);

} // namespace localized_cc
