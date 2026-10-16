#pragma once

#include "modernqc/molecule/atom.hpp"

#include <cstddef>
#include <vector>

namespace modernqc::molecule {

class Molecule {
 public:
  Molecule(std::vector<Atom> atoms, int charge, int multiplicity,
           std::vector<int> monomer_ids = {});

  [[nodiscard]] const std::vector<Atom>& atoms() const noexcept;
  [[nodiscard]] int charge() const noexcept;
  [[nodiscard]] int multiplicity() const noexcept;
  [[nodiscard]] int electron_count() const noexcept;
  [[nodiscard]] std::size_t occupied_orbitals_rhf() const;
  [[nodiscard]] double nuclear_repulsion_energy() const;
  [[nodiscard]] const std::vector<int>& monomer_ids() const noexcept;

 private:
  std::vector<Atom> atoms_;
  int charge_;
  int multiplicity_;
  int electron_count_;
  std::vector<int> monomer_ids_;
};

[[nodiscard]] std::vector<int> contiguous_water_monomers(
    const std::vector<Atom>& atoms, std::size_t atoms_per_monomer);

}  // namespace modernqc::molecule
