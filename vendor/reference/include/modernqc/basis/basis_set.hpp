#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace modernqc::basis {

struct Shell {
  std::size_t atom_index;
  std::size_t first_ao;
  std::size_t function_count;
  int angular_momentum;
  bool spherical;
  std::vector<double> exponents;
  std::vector<double> contraction_coefficients;
};

struct BasisSet {
  std::string canonical_name;
  bool spherical;
  std::size_t number_of_aos;
  std::size_t maximum_primitives;
  int maximum_angular_momentum;
  std::vector<Shell> shells;
  std::vector<std::size_t> ao_to_shell;
};

}  // namespace modernqc::basis
