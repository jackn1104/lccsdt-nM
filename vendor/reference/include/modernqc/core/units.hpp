#pragma once

#include <string_view>

namespace modernqc::core {

// CODATA 2018: 1 bohr = 0.529177210903 angstrom.
inline constexpr double angstrom_to_bohr = 1.8897261246257702;
inline constexpr double bohr_to_angstrom = 1.0 / angstrom_to_bohr;

enum class CoordinateUnit {
  angstrom,
  bohr,
};

[[nodiscard]] constexpr double to_bohr(double value, CoordinateUnit unit) noexcept {
  return unit == CoordinateUnit::angstrom ? value * angstrom_to_bohr : value;
}

[[nodiscard]] constexpr std::string_view unit_name(CoordinateUnit unit) noexcept {
  return unit == CoordinateUnit::angstrom ? "angstrom" : "bohr";
}

}  // namespace modernqc::core
