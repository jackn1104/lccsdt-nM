#pragma once

#include "modernqc/core/units.hpp"
#include "modernqc/molecule/atom.hpp"

#include <filesystem>
#include <istream>
#include <vector>

namespace modernqc::molecule {

[[nodiscard]] std::vector<Atom> read_xyz(
    std::istream& input, core::CoordinateUnit input_unit);

[[nodiscard]] std::vector<Atom> read_xyz_file(
    const std::filesystem::path& path, core::CoordinateUnit input_unit);

}  // namespace modernqc::molecule
