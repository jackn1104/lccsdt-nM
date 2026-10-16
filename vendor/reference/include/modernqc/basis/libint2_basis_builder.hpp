#pragma once

#include "modernqc/basis/basis_set.hpp"
#include "modernqc/molecule/molecule.hpp"

#include <string_view>

namespace modernqc::basis {

// Constructs a backend-neutral basis description. Libint2 types remain confined
// to the implementation file.
[[nodiscard]] BasisSet build_libint2_basis(
    const molecule::Molecule& molecule, std::string_view basis_name,
    bool spherical);

}  // namespace modernqc::basis
