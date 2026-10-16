#pragma once

#include <cstddef>
#include <istream>
#include <vector>

namespace localized_cc {

struct TceEnergyRecord {
  std::size_t scf_line{};
  std::size_t ccsd_line{};
  std::size_t triples_line{};
  double rhf_hartree{};
  double ccsd_correlation_hartree{};
  double triples_parenthetical_hartree{};

  [[nodiscard]] double ccsdt_total_hartree() const noexcept {
    return rhf_hartree + ccsd_correlation_hartree +
           triples_parenthetical_hartree;
  }
};

// Extract complete RHF/CCSD/(T) records from an NWChem TCE text output.
// CCSD[T] lines are intentionally ignored. A new SCF energy begins a new
// candidate record, so incomplete calculations are not returned.
[[nodiscard]] std::vector<TceEnergyRecord>
parse_tce_energy_records(std::istream& input);

} // namespace localized_cc
