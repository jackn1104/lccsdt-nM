#include "localized_cc/water_zmatrix.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace localized_cc {

bool WaterZMatrix::has_c2v_symmetry(const double tolerance) const {
  if (!(tolerance >= 0.0) || !std::isfinite(tolerance))
    throw std::invalid_argument("Z-matrix symmetry tolerance must be finite and nonnegative");
  return std::abs(oh1_angstrom - oh2_angstrom) <= tolerance;
}

std::string WaterZMatrix::point_group(const double tolerance) const {
  return has_c2v_symmetry(tolerance) ? "C2v" : "Cs";
}

std::vector<CartesianAtom> WaterZMatrix::cartesian_atoms(
    const double symmetry_tolerance) const {
  const double angle = hoh_degrees * std::acos(-1.0) / 180.0;
  if (has_c2v_symmetry(symmetry_tolerance)) {
    const double x = oh1_angstrom * std::sin(0.5 * angle) * angstrom_to_bohr;
    const double z = oh1_angstrom * std::cos(0.5 * angle) * angstrom_to_bohr;
    return {{"O", 8, {0.0, 0.0, 0.0}},
            {"H", 1, {x, 0.0, z}},
            {"H", 1, {-x, 0.0, z}}};
  }
  return {{"O", 8, {0.0, 0.0, 0.0}},
          {"H", 1, {0.0, 0.0, oh1_angstrom * angstrom_to_bohr}},
          {"H", 1,
           {oh2_angstrom * std::sin(angle) * angstrom_to_bohr, 0.0,
            oh2_angstrom * std::cos(angle) * angstrom_to_bohr}}};
}

WaterZMatrix read_water_zmatrix(const std::string& path) {
  std::ifstream input(path);
  std::string units_keyword;
  std::string units;
  std::string oxygen;
  std::string hydrogen1;
  std::string hydrogen1_reference;
  std::string hydrogen2;
  std::string hydrogen2_bond_reference;
  std::string hydrogen2_angle_reference;
  WaterZMatrix result;
  if (!input || !(input >> units_keyword >> units >> oxygen >> hydrogen1 >>
                  hydrogen1_reference >> result.oh1_angstrom >> hydrogen2 >>
                  hydrogen2_bond_reference >> result.oh2_angstrom >>
                  hydrogen2_angle_reference >> result.hoh_degrees))
    throw std::runtime_error("invalid water Z-matrix: " + path);
  std::string trailing;
  if (input >> trailing)
    throw std::runtime_error("unexpected data after water Z-matrix: " + path);
  if (units_keyword != "units" || units != "angstrom" || oxygen != "O" ||
      hydrogen1 != "H1" || hydrogen2 != "H2" ||
      hydrogen1_reference != "O" || hydrogen2_bond_reference != "O" ||
      hydrogen2_angle_reference != "H1")
    throw std::runtime_error(
        "water Z-matrix must use O, H1 O R, H2 O R H1 A in angstrom");
  if (!(result.oh1_angstrom > 0.0) || !(result.oh2_angstrom > 0.0) ||
      !std::isfinite(result.oh1_angstrom) ||
      !std::isfinite(result.oh2_angstrom) ||
      !(result.hoh_degrees > 0.0) || !(result.hoh_degrees < 180.0) ||
      !std::isfinite(result.hoh_degrees))
    throw std::runtime_error("invalid water Z-matrix bond or angle: " + path);
  return result;
}

} // namespace localized_cc
