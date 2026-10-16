#include "modernqc/basis/libint2_basis_builder.hpp"

#include "modernqc/basis/basis_name.hpp"

#include <libint2/basis.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace modernqc::basis {
namespace {

[[nodiscard]] std::vector<libint2::Atom> to_libint_atoms(
    const molecule::Molecule& molecule) {
  std::vector<libint2::Atom> result;
  result.reserve(molecule.atoms().size());
  for(const molecule::Atom& atom : molecule.atoms()) {
    result.push_back(libint2::Atom{
        .atomic_number = atom.atomic_number,
        .x = atom.position_bohr[0],
        .y = atom.position_bohr[1],
        .z = atom.position_bohr[2],
    });
  }
  return result;
}

}  // namespace

BasisSet build_libint2_basis(const molecule::Molecule& molecule,
                             std::string_view basis_name, bool spherical) {
  if(!spherical) {
    throw std::invalid_argument(
        "the initial validated basis path requires spherical functions");
  }

  const std::string canonical_name = canonical_basis_name(basis_name);
  const std::vector<libint2::Atom> atoms = to_libint_atoms(molecule);
  libint2::BasisSet backend_basis{canonical_name, atoms, true};
  backend_basis.set_pure(true);

  const long backend_nbf = backend_basis.nbf();
  const long backend_max_l = backend_basis.max_l();
  if(backend_nbf <= 0 || backend_max_l < 0) {
    throw std::runtime_error("Libint2 produced invalid basis dimensions");
  }
  if(static_cast<unsigned long long>(backend_nbf) >
     static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("Libint2 AO count does not fit in size_t");
  }

  const std::vector<std::size_t>& shell_to_ao = backend_basis.shell2bf();
  const std::vector<long> shell_to_atom = backend_basis.shell2atom(atoms);
  if(shell_to_ao.size() != backend_basis.size() ||
     shell_to_atom.size() != backend_basis.size()) {
    throw std::runtime_error("Libint2 returned inconsistent shell metadata");
  }

  BasisSet result{
      .canonical_name = canonical_name,
      .spherical = true,
      .number_of_aos = static_cast<std::size_t>(backend_nbf),
      .maximum_primitives = backend_basis.max_nprim(),
      .maximum_angular_momentum = static_cast<int>(backend_max_l),
      .shells = {},
      .ao_to_shell = std::vector<std::size_t>(
          static_cast<std::size_t>(backend_nbf), 0),
  };
  result.shells.reserve(backend_basis.size());

  for(std::size_t shell_index = 0; shell_index < backend_basis.size();
      ++shell_index) {
    const libint2::Shell& backend_shell = backend_basis[shell_index];
    if(backend_shell.contr.size() != 1) {
      throw std::runtime_error(
          "segmented basis shell unexpectedly has multiple contractions");
    }
    if(shell_to_atom[shell_index] < 0 ||
       static_cast<std::size_t>(shell_to_atom[shell_index]) >=
           molecule.atoms().size()) {
      throw std::runtime_error("Libint2 shell could not be assigned to an atom");
    }

    const std::size_t first_ao = shell_to_ao[shell_index];
    const std::size_t function_count = backend_shell.size();
    if(first_ao > result.number_of_aos ||
       function_count > result.number_of_aos - first_ao) {
      throw std::runtime_error("Libint2 shell AO range is inconsistent");
    }

    const libint2::Shell::Contraction& contraction = backend_shell.contr[0];
    result.shells.push_back(Shell{
        .atom_index = static_cast<std::size_t>(shell_to_atom[shell_index]),
        .first_ao = first_ao,
        .function_count = function_count,
        .angular_momentum = contraction.l,
        .spherical = contraction.pure,
        .exponents = std::vector<double>(backend_shell.alpha.begin(),
                                         backend_shell.alpha.end()),
        .contraction_coefficients =
            std::vector<double>(contraction.coeff.begin(),
                                contraction.coeff.end()),
    });
    for(std::size_t ao = first_ao; ao < first_ao + function_count; ++ao) {
      result.ao_to_shell[ao] = shell_index;
    }
  }

  return result;
}

}  // namespace modernqc::basis
