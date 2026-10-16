#include "modernqc/molecule/molecule.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace modernqc::molecule {

int atomic_number(std::string_view symbol) {
  if(symbol == "H") {
    return 1;
  }
  if(symbol == "O") {
    return 8;
  }
  throw std::invalid_argument("unsupported element symbol '" + std::string{symbol} +
                              "'; initial support is limited to H and O");
}

std::string canonical_element_symbol(std::string_view symbol) {
  if(symbol == "H" || symbol == "h") {
    return "H";
  }
  if(symbol == "O" || symbol == "o") {
    return "O";
  }
  throw std::invalid_argument("unsupported element symbol '" + std::string{symbol} +
                              "'; initial support is limited to H and O");
}

Molecule::Molecule(std::vector<Atom> atoms, int charge, int multiplicity,
                   std::vector<int> monomer_ids)
    : atoms_{std::move(atoms)},
      charge_{charge},
      multiplicity_{multiplicity},
      electron_count_{0},
      monomer_ids_{std::move(monomer_ids)} {
  if(atoms_.empty()) {
    throw std::invalid_argument("molecule must contain at least one atom");
  }
  if(multiplicity_ < 1) {
    throw std::invalid_argument("spin multiplicity must be positive");
  }

  long long nuclear_charge = 0;
  for(const Atom& atom : atoms_) {
    if(atom.atomic_number != atomic_number(atom.symbol)) {
      throw std::invalid_argument("atom symbol and atomic number are inconsistent");
    }
    if(!std::all_of(atom.position_bohr.begin(), atom.position_bohr.end(),
                    [](double value) { return std::isfinite(value); })) {
      throw std::invalid_argument("atom coordinates must be finite");
    }
    nuclear_charge += atom.atomic_number;
  }
  const long long electrons = nuclear_charge - charge_;
  if(electrons <= 0 || electrons > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("molecular charge produces an invalid electron count");
  }
  electron_count_ = static_cast<int>(electrons);
  const int unpaired_electrons = multiplicity_ - 1;
  if(unpaired_electrons > electron_count_ ||
     (electron_count_ - unpaired_electrons) % 2 != 0) {
    throw std::invalid_argument(
        "electron count, charge, and spin multiplicity are inconsistent");
  }
  if(multiplicity_ != 1 || electron_count_ % 2 != 0) {
    throw std::invalid_argument(
        "initial RHF implementation requires a closed-shell singlet");
  }

  if(monomer_ids_.empty()) {
    monomer_ids_.assign(atoms_.size(), -1);
  }
  if(monomer_ids_.size() != atoms_.size()) {
    throw std::invalid_argument("monomer ID count must equal the atom count");
  }
}

const std::vector<Atom>& Molecule::atoms() const noexcept { return atoms_; }

int Molecule::charge() const noexcept { return charge_; }

int Molecule::multiplicity() const noexcept { return multiplicity_; }

int Molecule::electron_count() const noexcept { return electron_count_; }

std::size_t Molecule::occupied_orbitals_rhf() const {
  return static_cast<std::size_t>(electron_count_ / 2);
}

double Molecule::nuclear_repulsion_energy() const {
  double energy = 0.0;
  for(std::size_t atom_a = 0; atom_a < atoms_.size(); ++atom_a) {
    for(std::size_t atom_b = atom_a + 1; atom_b < atoms_.size(); ++atom_b) {
      double squared_distance = 0.0;
      for(std::size_t axis = 0; axis < 3; ++axis) {
        const double difference =
            atoms_[atom_a].position_bohr[axis] - atoms_[atom_b].position_bohr[axis];
        squared_distance += difference * difference;
      }
      if(squared_distance <= 1.0e-24) {
        throw std::invalid_argument("coincident nuclei are not allowed");
      }
      energy += static_cast<double>(atoms_[atom_a].atomic_number *
                                    atoms_[atom_b].atomic_number) /
                std::sqrt(squared_distance);
    }
  }
  return energy;
}

const std::vector<int>& Molecule::monomer_ids() const noexcept {
  return monomer_ids_;
}

std::vector<int> contiguous_water_monomers(const std::vector<Atom>& atoms,
                                           std::size_t atoms_per_monomer) {
  if(atoms_per_monomer != 3) {
    throw std::invalid_argument(
        "water monomer assignment requires atoms_per_monomer = 3");
  }
  if(atoms.empty() || atoms.size() % atoms_per_monomer != 0) {
    throw std::invalid_argument(
        "atom count must be a nonzero multiple of atoms_per_monomer");
  }

  std::vector<int> ids(atoms.size(), -1);
  for(std::size_t begin = 0; begin < atoms.size(); begin += atoms_per_monomer) {
    int oxygen_count = 0;
    int hydrogen_count = 0;
    for(std::size_t offset = 0; offset < atoms_per_monomer; ++offset) {
      oxygen_count += atoms[begin + offset].symbol == "O" ? 1 : 0;
      hydrogen_count += atoms[begin + offset].symbol == "H" ? 1 : 0;
      ids[begin + offset] = static_cast<int>(begin / atoms_per_monomer);
    }
    if(oxygen_count != 1 || hydrogen_count != 2) {
      throw std::invalid_argument(
          "each requested water monomer must contain exactly one O and two H atoms");
    }
  }
  return ids;
}

}  // namespace modernqc::molecule
