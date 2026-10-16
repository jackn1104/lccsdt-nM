#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

namespace modernqc::molecule {

struct Atom {
  std::string symbol;
  int atomic_number;
  std::array<double, 3> position_bohr;
};

[[nodiscard]] int atomic_number(std::string_view symbol);
[[nodiscard]] std::string canonical_element_symbol(std::string_view symbol);

}  // namespace modernqc::molecule
