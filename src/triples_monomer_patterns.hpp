#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <span>
#include <vector>

namespace localized_cc::detail {

// Internal hot-loop helper: all extents and pattern IDs are constructed and
// validated by the sector planner. The last slot is an independent total.
inline void accumulate_triples_energy_row(
    std::span<const double> left, std::span<const double> response,
    std::span<const std::size_t> virtual_patterns,
    const unsigned char* orders, double weight,
    std::array<double, 7>& energies) {
  for (std::size_t column = 0; column < left.size(); ++column) {
    const double contribution = weight * left[column] * response[column];
    energies[orders[virtual_patterns[column]] - 1] -= contribution;
    energies[6] -= contribution;
  }
}

// Exact label-set compression, not a locality approximation. Arbitrary
// nonnegative monomer IDs are supported (no 64-monomer bitmask limit).
struct TripleMonomerPatterns {
  std::vector<std::array<int, 3>> patterns;
  std::vector<std::size_t> columns;
  std::vector<std::size_t> counts;

  template <typename Triples>
  TripleMonomerPatterns(const Triples& triples, const std::vector<int>& labels) {
    std::map<std::array<int, 3>, std::size_t> index;
    columns.reserve(triples.size());
    for (const auto& triple : triples) {
      std::array<int, 3> key{labels.at(triple.spin[0] / 2),
                             labels.at(triple.spin[1] / 2),
                             labels.at(triple.spin[2] / 2)};
      std::sort(key.begin(), key.end());
      const auto last = std::unique(key.begin(), key.end());
      std::fill(last, key.end(), -1);
      const auto [entry, inserted] = index.emplace(key, patterns.size());
      if (inserted) {
        patterns.push_back(key);
        counts.push_back(0);
      }
      columns.push_back(entry->second);
      ++counts[entry->second];
    }
  }
};

inline unsigned char triple_pattern_span(const std::array<int, 3>& occupied,
                                         const std::array<int, 3>& virtuals) {
  unsigned char count = 0;
  for (const int label : occupied) count += label >= 0;
  for (const int label : virtuals)
    if (label >= 0 && std::find(occupied.begin(), occupied.end(), label) == occupied.end())
      ++count;
  return count;
}
} // namespace localized_cc::detail
