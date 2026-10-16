#pragma once

#include <array>
#include <string>
#include <vector>

namespace localized_cc {

inline constexpr double angstrom_to_bohr = 1.8897261246257702;

struct CartesianAtom {
  std::string symbol;
  int atomic_number{};
  std::array<double, 3> position_bohr{};
};

struct WaterZMatrix {
  double oh1_angstrom{};
  double oh2_angstrom{};
  double hoh_degrees{};

  [[nodiscard]] bool has_c2v_symmetry(double tolerance = 1.0e-12) const;
  [[nodiscard]] std::string point_group(double tolerance = 1.0e-12) const;
  [[nodiscard]] std::vector<CartesianAtom> cartesian_atoms(
      double symmetry_tolerance = 1.0e-12) const;
};

WaterZMatrix read_water_zmatrix(const std::string& path);

} // namespace localized_cc
