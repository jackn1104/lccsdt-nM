#include "modernqc/integrals/integral_provider.hpp"

#include "modernqc/basis/basis_name.hpp"
#include "modernqc/basis/libint2_basis_builder.hpp"
#include "modernqc/linalg/matrix.hpp"

#include <libint2/engine.h>
#include <libint2/cgshell_ordering.h>
#include <libint2/initialize.h>
#include <libint2/solidharmonics.h>

#if !defined(LIBINT_SHGSHELL_ORDERING) || \
    !defined(LIBINT_SHGSHELL_ORDERING_STANDARD)
#error "ModernQC requires Libint2 to expose its solid-harmonic AO ordering"
#elif LIBINT_SHGSHELL_ORDERING != LIBINT_SHGSHELL_ORDERING_STANDARD
#error "ModernQC requires Libint2 standard solid-harmonic AO ordering"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef MODERNQC_HAS_OPENMP
#include <omp.h>
#endif

extern "C" {
void dgemm_(const char* transa, const char* transb, const int* m,
            const int* n, const int* k, const double* alpha,
            const double* a, const int* lda, const double* b,
            const int* ldb, const double* beta, double* c,
            const int* ldc);
}

namespace modernqc::integrals {
namespace {

class Libint2Runtime {
 public:
  Libint2Runtime() { libint2::initialize(); }
  ~Libint2Runtime() { libint2::finalize(); }
  Libint2Runtime(const Libint2Runtime&) = delete;
  Libint2Runtime& operator=(const Libint2Runtime&) = delete;
};

void ensure_libint2_runtime() {
  static const Libint2Runtime runtime;
  static_cast<void>(runtime);
}

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

[[nodiscard]] basis::BasisSet copy_basis_metadata(
    const libint2::BasisSet& backend_basis,
    const std::vector<libint2::Atom>& atoms,
    std::string canonical_name) {
  const long backend_nbf = backend_basis.nbf();
  const long backend_max_l = backend_basis.max_l();
  if(backend_nbf <= 0 || backend_max_l < 0 ||
     static_cast<unsigned long long>(backend_nbf) >
         static_cast<unsigned long long>(
             std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(
        "reference basis produced invalid dimensions");
  }
  const auto& shell_to_ao = backend_basis.shell2bf();
  const std::vector<long> shell_to_atom = backend_basis.shell2atom(atoms);
  if(shell_to_ao.size() != backend_basis.size() ||
     shell_to_atom.size() != backend_basis.size()) {
    throw std::runtime_error(
        "reference basis produced inconsistent shell metadata");
  }

  basis::BasisSet result{
      .canonical_name = std::move(canonical_name),
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
    const libint2::Shell& shell = backend_basis[shell_index];
    if(shell.contr.size() != 1) {
      throw std::runtime_error(
          "reference basis must be segmented into one contraction per shell");
    }
    if(shell_to_atom[shell_index] < 0 ||
       static_cast<std::size_t>(shell_to_atom[shell_index]) >= atoms.size()) {
      throw std::runtime_error(
          "reference basis shell could not be assigned to an atom");
    }
    const auto& contraction = shell.contr.front();
    const std::size_t first_ao = shell_to_ao[shell_index];
    const std::size_t function_count = shell.size();
    result.shells.push_back(basis::Shell{
        .atom_index = static_cast<std::size_t>(
            shell_to_atom[shell_index]),
        .first_ao = first_ao,
        .function_count = function_count,
        .angular_momentum = contraction.l,
        .spherical = contraction.pure,
        .exponents = std::vector<double>(shell.alpha.begin(),
                                         shell.alpha.end()),
        .contraction_coefficients =
            std::vector<double>(contraction.coeff.begin(),
                                contraction.coeff.end()),
    });
    for(std::size_t ao = first_ao; ao < first_ao + function_count; ++ao) {
      result.ao_to_shell.at(ao) = shell_index;
    }
  }
  return result;
}

[[nodiscard]] linalg::Matrix compute_one_body(
    const libint2::BasisSet& basis, const std::vector<libint2::Atom>& atoms,
    libint2::Operator operation) {
  const long backend_nbf = basis.nbf();
  if(backend_nbf <= 0) {
    throw std::runtime_error("Libint2 basis has an invalid AO count");
  }
  if(basis.max_l() < 0 ||
     basis.max_l() > static_cast<long>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "Libint2 basis angular momentum is outside the engine range");
  }
  const std::size_t nbf = static_cast<std::size_t>(backend_nbf);
  linalg::Matrix result{nbf, nbf};
  libint2::Engine engine{operation, basis.max_nprim(),
                         static_cast<int>(basis.max_l()), 0};
  if(operation == libint2::Operator::nuclear) {
    using Charge = std::pair<double, std::array<double, 3>>;
    std::vector<Charge> charges;
    charges.reserve(atoms.size());
    for(const libint2::Atom& atom : atoms) {
      charges.emplace_back(
          static_cast<double>(atom.atomic_number),
          std::array<double, 3>{atom.x, atom.y, atom.z});
    }
    engine.set_params(charges);
  }

  const std::vector<std::size_t>& shell_to_ao = basis.shell2bf();
  const auto& buffers = engine.results();
  for(std::size_t shell_1 = 0; shell_1 < basis.size(); ++shell_1) {
    const std::size_t ao_1 = shell_to_ao[shell_1];
    const std::size_t functions_1 = basis[shell_1].size();
    for(std::size_t shell_2 = 0; shell_2 <= shell_1; ++shell_2) {
      const std::size_t ao_2 = shell_to_ao[shell_2];
      const std::size_t functions_2 = basis[shell_2].size();
      engine.compute(basis[shell_1], basis[shell_2]);
      const double* buffer = buffers[0];
      if(buffer == nullptr) {
        continue;
      }
      for(std::size_t function_2 = 0; function_2 < functions_2;
          ++function_2) {
        for(std::size_t function_1 = 0; function_1 < functions_1;
            ++function_1) {
          const double value =
              buffer[function_1 * functions_2 + function_2];
          result(ao_1 + function_1, ao_2 + function_2) = value;
          if(shell_1 != shell_2) {
            result(ao_2 + function_2, ao_1 + function_1) = value;
          }
        }
      }
    }
  }
  linalg::require_finite(result, "one-electron integral matrix");
  return result;
}

[[nodiscard]] linalg::Matrix compute_cross_overlap(
    const libint2::BasisSet& target, const libint2::BasisSet& source) {
  const long target_nbf_backend = target.nbf();
  const long source_nbf_backend = source.nbf();
  if(target_nbf_backend <= 0 || source_nbf_backend <= 0) {
    throw std::runtime_error(
        "Libint2 cross-overlap basis has an invalid AO count");
  }
  const long maximum_l = std::max(target.max_l(), source.max_l());
  const std::size_t maximum_primitives =
      std::max(target.max_nprim(), source.max_nprim());
  if(maximum_l < 0 ||
     maximum_l > static_cast<long>(std::numeric_limits<int>::max()) ||
     maximum_primitives == 0) {
    throw std::runtime_error(
        "Libint2 cross-overlap basis is outside the engine range");
  }
  const std::size_t target_nbf =
      static_cast<std::size_t>(target_nbf_backend);
  const std::size_t source_nbf =
      static_cast<std::size_t>(source_nbf_backend);
  linalg::Matrix result{target_nbf, source_nbf};
  libint2::Engine engine{libint2::Operator::overlap,
                         maximum_primitives,
                         static_cast<int>(maximum_l), 0};
  const auto& target_shell_to_ao = target.shell2bf();
  const auto& source_shell_to_ao = source.shell2bf();
  const auto& buffers = engine.results();
  for(std::size_t target_shell = 0; target_shell < target.size();
      ++target_shell) {
    const std::size_t target_ao = target_shell_to_ao[target_shell];
    const std::size_t target_functions = target[target_shell].size();
    for(std::size_t source_shell = 0; source_shell < source.size();
        ++source_shell) {
      const std::size_t source_ao = source_shell_to_ao[source_shell];
      const std::size_t source_functions = source[source_shell].size();
      engine.compute(target[target_shell], source[source_shell]);
      const double* buffer = buffers[0];
      if(buffer == nullptr) {
        continue;
      }
      for(std::size_t target_function = 0;
          target_function < target_functions; ++target_function) {
        for(std::size_t source_function = 0;
            source_function < source_functions; ++source_function) {
          result(target_ao + target_function, source_ao + source_function) =
              buffer[target_function * source_functions + source_function];
        }
      }
    }
  }
  linalg::require_finite(result, "cross-basis AO overlap");
  return result;
}

[[nodiscard]] std::vector<double> compute_shell_pair_schwarz_bounds(
    const libint2::BasisSet& basis) {
  const std::size_t shell_count = basis.size();
  if(shell_count != 0 &&
     shell_count > std::numeric_limits<std::size_t>::max() / shell_count) {
    throw std::overflow_error("shell-pair bound allocation size overflow");
  }
  std::vector<double> bounds(shell_count * shell_count, 0.0);
  libint2::Engine engine{libint2::Operator::coulomb, basis.max_nprim(),
                         static_cast<int>(basis.max_l()), 0, 0.0};
  const auto& buffers = engine.results();
  for(std::size_t shell_1 = 0; shell_1 < shell_count; ++shell_1) {
    const std::size_t functions_1 = basis[shell_1].size();
    for(std::size_t shell_2 = 0; shell_2 <= shell_1; ++shell_2) {
      const std::size_t functions_2 = basis[shell_2].size();
      engine.compute(basis[shell_1], basis[shell_2], basis[shell_1],
                     basis[shell_2]);
      const double* buffer = buffers[0];
      if(buffer == nullptr) {
        throw std::runtime_error(
            "no-screen Libint2 ERI engine returned a null shell-pair norm");
      }
      double maximum_diagonal = 0.0;
      for(std::size_t function_1 = 0; function_1 < functions_1;
          ++function_1) {
        for(std::size_t function_2 = 0; function_2 < functions_2;
            ++function_2) {
          const std::size_t index =
              (((function_1 * functions_2) + function_2) * functions_1 +
               function_1) *
                  functions_2 +
              function_2;
          const double diagonal = buffer[index];
          if(!std::isfinite(diagonal)) {
            throw std::runtime_error(
                "nonfinite diagonal ERI while building Schwarz bounds");
          }
          // (mu nu|mu nu) is nonnegative analytically. abs protects the
          // upper bound against an insignificant negative roundoff value.
          maximum_diagonal =
              std::max(maximum_diagonal, std::abs(diagonal));
        }
      }
      const double bound = std::sqrt(maximum_diagonal);
      bounds[shell_1 * shell_count + shell_2] = bound;
      bounds[shell_2 * shell_count + shell_1] = bound;
    }
  }
  return bounds;
}

[[nodiscard]] double maximum_density_block(
    const linalg::Matrix& density, std::size_t first_row,
    std::size_t row_count, std::size_t first_column,
    std::size_t column_count) {
  double maximum = 0.0;
  for(std::size_t column = 0; column < column_count; ++column) {
    for(std::size_t row = 0; row < row_count; ++row) {
      maximum = std::max(
          maximum,
          std::abs(density(first_row + row, first_column + column)));
    }
  }
  return maximum;
}

[[nodiscard]] double density_aware_fock_bound(
    const linalg::Matrix& density, double eri_bound, double degeneracy,
    std::size_t ao_1, std::size_t functions_1, std::size_t ao_2,
    std::size_t functions_2, std::size_t ao_3, std::size_t functions_3,
    std::size_t ao_4, std::size_t functions_4) {
  const double density_12 = maximum_density_block(
      density, ao_1, functions_1, ao_2, functions_2);
  const double density_34 = maximum_density_block(
      density, ao_3, functions_3, ao_4, functions_4);
  const double density_13 = maximum_density_block(
      density, ao_1, functions_1, ao_3, functions_3);
  const double density_24 = maximum_density_block(
      density, ao_2, functions_2, ao_4, functions_4);
  const double density_14 = maximum_density_block(
      density, ao_1, functions_1, ao_4, functions_4);
  const double density_23 = maximum_density_block(
      density, ao_2, functions_2, ao_3, functions_3);

  const double coulomb_12 =
      0.5 * static_cast<double>(functions_3) *
      static_cast<double>(functions_4) * density_34;
  const double coulomb_34 =
      0.5 * static_cast<double>(functions_1) *
      static_cast<double>(functions_2) * density_12;
  const double exchange_13 =
      0.125 * static_cast<double>(functions_2) *
      static_cast<double>(functions_4) * density_24;
  const double exchange_24 =
      0.125 * static_cast<double>(functions_1) *
      static_cast<double>(functions_3) * density_13;
  const double exchange_14 =
      0.125 * static_cast<double>(functions_2) *
      static_cast<double>(functions_3) * density_23;
  const double exchange_23 =
      0.125 * static_cast<double>(functions_1) *
      static_cast<double>(functions_4) * density_14;
  // A shell quartet can update the same matrix element through more than one
  // symmetry-related contraction when shell indices coincide. Summing all
  // six bounds, rather than taking their maximum, remains conservative in
  // those degenerate cases.
  const double density_factor =
      coulomb_12 + coulomb_34 + exchange_13 + exchange_24 + exchange_14 +
      exchange_23;
  const double result = degeneracy * eri_bound * density_factor;
  if(!std::isfinite(result)) {
    throw std::runtime_error(
        "nonfinite density-aware Fock contribution bound");
  }
  return result;
}

struct ShellQuartet {
  std::size_t shell_1;
  std::size_t shell_2;
  std::size_t shell_3;
  std::size_t shell_4;
};

[[nodiscard]] std::vector<ShellQuartet> enumerate_unique_shell_quartets(
    std::size_t shell_count) {
  std::vector<ShellQuartet> quartets;
  for(std::size_t shell_1 = 0; shell_1 < shell_count; ++shell_1) {
    for(std::size_t shell_2 = 0; shell_2 <= shell_1; ++shell_2) {
      for(std::size_t shell_3 = 0; shell_3 <= shell_1; ++shell_3) {
        const std::size_t maximum_shell_4 =
            shell_1 == shell_3 ? shell_2 : shell_3;
        for(std::size_t shell_4 = 0; shell_4 <= maximum_shell_4;
            ++shell_4) {
          quartets.push_back(ShellQuartet{
              .shell_1 = shell_1,
              .shell_2 = shell_2,
              .shell_3 = shell_3,
              .shell_4 = shell_4,
          });
        }
      }
    }
  }
  return quartets;
}

[[nodiscard]] double integer_power(double value, int exponent) {
  double result = 1.0;
  for(int index = 0; index < exponent; ++index) {
    result *= value;
  }
  return result;
}

[[nodiscard]] double binomial(const int n, const int k) {
  if(k < 0 || k > n)
    return 0.0;
  double result = 1.0;
  for(int index = 1; index <= k; ++index)
    result *= static_cast<double>(n - k + index) /
              static_cast<double>(index);
  return result;
}

[[nodiscard]] double gaussian_even_moment(const int power,
                                          const double exponent) {
  if(power % 2 != 0)
    return 0.0;
  const int half = power / 2;
  double odd_double_factorial = 1.0;
  for(int value = 1; value < 2 * half; value += 2)
    odd_double_factorial *= static_cast<double>(value);
  return odd_double_factorial * std::sqrt(std::acos(-1.0)) /
         (std::pow(2.0, half) *
          std::pow(exponent, static_cast<double>(half) + 0.5));
}

[[nodiscard]] double one_dimensional_position_moment(
    const double alpha, const double center_a, const int angular_a,
    const double beta, const double center_b, const int angular_b,
    const int position_power) {
  const double exponent = alpha + beta;
  const double product_center =
      (alpha * center_a + beta * center_b) / exponent;
  double result = 0.0;
  for(int pa = 0; pa <= angular_a; ++pa) {
    const double ca = binomial(angular_a, pa) *
                      integer_power(product_center - center_a,
                                    angular_a - pa);
    for(int pb = 0; pb <= angular_b; ++pb) {
      const double cb = binomial(angular_b, pb) *
                        integer_power(product_center - center_b,
                                      angular_b - pb);
      for(int pr = 0; pr <= position_power; ++pr) {
        const double cr = binomial(position_power, pr) *
                          integer_power(product_center,
                                        position_power - pr);
        result += ca * cb * cr *
                  gaussian_even_moment(pa + pb + pr, exponent);
      }
    }
  }
  return result;
}

struct CartesianExponents {
  int x{};
  int y{};
  int z{};
};

[[nodiscard]] std::vector<CartesianExponents> cartesian_exponents(
    const int angular_momentum) {
  std::vector<CartesianExponents> result;
  int x = 0;
  int y = 0;
  int z = 0;
  FOR_CART(x, y, z, angular_momentum)
  result.push_back({x, y, z});
  END_FOR_CART
  return result;
}

[[nodiscard]] linalg::Matrix compute_cartesian_monomial(
    const libint2::BasisSet& basis, const int power_x, const int power_y,
    const int power_z) {
  const auto nbf = static_cast<std::size_t>(basis.nbf());
  linalg::Matrix result{nbf, nbf};
  const auto& shell_to_ao = basis.shell2bf();
  const auto shell_count = basis.size();
  for(const auto& shell : basis)
    if(shell.contr.size() != 1)
      throw std::runtime_error(
          "Cartesian moments require segmented Libint2 shells");
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for(std::ptrdiff_t raw_shell_a = 0;
      raw_shell_a < static_cast<std::ptrdiff_t>(shell_count); ++raw_shell_a) {
    const auto shell_a_index = static_cast<std::size_t>(raw_shell_a);
    const auto& shell_a = basis[shell_a_index];
    const auto& contraction_a = shell_a.contr[0];
    const auto exponents_a = cartesian_exponents(contraction_a.l);
    for(std::size_t shell_b_index = 0; shell_b_index <= shell_a_index;
        ++shell_b_index) {
      const auto& shell_b = basis[shell_b_index];
      const auto& contraction_b = shell_b.contr[0];
      const auto exponents_b = cartesian_exponents(contraction_b.l);
      std::vector<double> cartesian_block(
          exponents_a.size() * exponents_b.size(), 0.0);
      for(std::size_t function_a = 0; function_a < exponents_a.size();
          ++function_a) {
        const auto angular_a = exponents_a[function_a];
        for(std::size_t function_b = 0; function_b < exponents_b.size();
            ++function_b) {
          const auto angular_b = exponents_b[function_b];
          long double value = 0.0L;
          for(std::size_t primitive_a = 0;
              primitive_a < shell_a.alpha.size(); ++primitive_a) {
            const double alpha = shell_a.alpha[primitive_a];
            for(std::size_t primitive_b = 0;
                primitive_b < shell_b.alpha.size(); ++primitive_b) {
              const double beta = shell_b.alpha[primitive_b];
              const double exponent = alpha + beta;
              double distance_squared = 0.0;
              for(std::size_t axis = 0; axis < 3; ++axis) {
                const double displacement = shell_a.O[axis] - shell_b.O[axis];
                distance_squared += displacement * displacement;
              }
              const double product_factor =
                  std::exp(-alpha * beta * distance_squared / exponent);
              const double ix = one_dimensional_position_moment(
                  alpha, shell_a.O[0], angular_a.x, beta, shell_b.O[0],
                  angular_b.x, power_x);
              const double iy = one_dimensional_position_moment(
                  alpha, shell_a.O[1], angular_a.y, beta, shell_b.O[1],
                  angular_b.y, power_y);
              const double iz = one_dimensional_position_moment(
                  alpha, shell_a.O[2], angular_a.z, beta, shell_b.O[2],
                  angular_b.z, power_z);
              value += static_cast<long double>(contraction_a.coeff[primitive_a]) *
                       static_cast<long double>(contraction_b.coeff[primitive_b]) *
                       static_cast<long double>(product_factor * ix * iy * iz);
            }
          }
          cartesian_block[function_a * exponents_b.size() + function_b] =
              static_cast<double>(value);
        }
      }
      std::vector<double> spherical_block(shell_a.size() * shell_b.size());
      if(contraction_a.pure && contraction_b.pure) {
        libint2::solidharmonics::tform(
            contraction_a.l, contraction_b.l, cartesian_block.data(),
            spherical_block.data());
      } else {
        libint2::solidharmonics::tform(
            contraction_a, contraction_b, cartesian_block.data(),
            spherical_block.data());
      }
      const auto ao_a = shell_to_ao[shell_a_index];
      const auto ao_b = shell_to_ao[shell_b_index];
      for(std::size_t function_a = 0; function_a < shell_a.size();
          ++function_a)
        for(std::size_t function_b = 0; function_b < shell_b.size();
            ++function_b) {
          const double value =
              spherical_block[function_a * shell_b.size() + function_b];
          result(ao_a + function_a, ao_b + function_b) = value;
          result(ao_b + function_b, ao_a + function_a) = value;
        }
    }
  }
  linalg::require_finite(result, "Cartesian AO position moment");
  return result;
}

}  // namespace

struct Libint2IntegralProvider::Impl {
  basis::BasisSet metadata;
  std::vector<libint2::Atom> atoms;
  libint2::BasisSet backend_basis;
  std::vector<double> shell_pair_schwarz_bounds;

  Impl(const molecule::Molecule& molecule, std::string_view basis_name,
       bool spherical)
      : metadata{basis::build_libint2_basis(molecule, basis_name, spherical)},
        atoms{to_libint_atoms(molecule)},
        backend_basis{basis::canonical_basis_name(basis_name), atoms, true} {
    backend_basis.set_pure(spherical);
    // Runtime solid-harmonic ordering selection was added in Libint 2.8.
#if LIBINT_MAJOR_VERSION == 2 && LIBINT_MINOR_VERSION >= 8
    if(libint2::solid_harmonics_ordering() !=
       libint2::SHGShellOrdering_Standard) {
      throw std::runtime_error(
          "ModernQC requires Libint2 standard solid-harmonic AO ordering");
    }
#endif
    if(static_cast<std::size_t>(backend_basis.nbf()) !=
           metadata.number_of_aos ||
       backend_basis.size() != metadata.shells.size()) {
      throw std::runtime_error(
          "Libint2 basis and copied ModernQC metadata disagree");
    }
    shell_pair_schwarz_bounds =
        compute_shell_pair_schwarz_bounds(backend_basis);
  }
};

Libint2IntegralProvider::Libint2IntegralProvider(
    const molecule::Molecule& molecule, std::string_view basis_name,
    bool spherical) {
  ensure_libint2_runtime();
  implementation_ =
      std::make_unique<Impl>(molecule, basis_name, spherical);
}

Libint2IntegralProvider::~Libint2IntegralProvider() = default;

Libint2IntegralProvider::Libint2IntegralProvider(
    Libint2IntegralProvider&&) noexcept = default;

Libint2IntegralProvider& Libint2IntegralProvider::operator=(
    Libint2IntegralProvider&&) noexcept = default;

const basis::BasisSet& Libint2IntegralProvider::basis_metadata() const noexcept {
  return implementation_->metadata;
}

OneElectronIntegrals Libint2IntegralProvider::compute_one_electron() const {
  linalg::Matrix overlap =
      compute_one_body(implementation_->backend_basis, implementation_->atoms,
                       libint2::Operator::overlap);
  linalg::Matrix kinetic =
      compute_one_body(implementation_->backend_basis, implementation_->atoms,
                       libint2::Operator::kinetic);
  linalg::Matrix nuclear =
      compute_one_body(implementation_->backend_basis, implementation_->atoms,
                       libint2::Operator::nuclear);
  linalg::Matrix core = linalg::add(kinetic, nuclear);
  return OneElectronIntegrals{
      .overlap = std::move(overlap),
      .kinetic = std::move(kinetic),
      .nuclear_attraction = std::move(nuclear),
      .core_hamiltonian = std::move(core),
  };
}

linalg::Matrix Libint2IntegralProvider::compute_cross_overlap(
    const Libint2IntegralProvider& source) const {
  if(implementation_->atoms.size() != source.implementation_->atoms.size()) {
    throw std::invalid_argument(
        "cross-overlap providers have different atom counts");
  }
  constexpr double coordinate_tolerance_bohr = 1.0e-12;
  for(std::size_t atom = 0; atom < implementation_->atoms.size(); ++atom) {
    const auto& target_atom = implementation_->atoms[atom];
    const auto& source_atom = source.implementation_->atoms[atom];
    if(target_atom.atomic_number != source_atom.atomic_number ||
       std::abs(target_atom.x - source_atom.x) > coordinate_tolerance_bohr ||
       std::abs(target_atom.y - source_atom.y) > coordinate_tolerance_bohr ||
       std::abs(target_atom.z - source_atom.z) > coordinate_tolerance_bohr) {
      throw std::invalid_argument(
          "cross-overlap providers have different atoms or coordinates");
    }
  }
  if(implementation_->metadata.spherical !=
     source.implementation_->metadata.spherical) {
    throw std::invalid_argument(
        "cross-overlap providers use different spherical/cartesian conventions");
  }
  return integrals::compute_cross_overlap(implementation_->backend_basis,
                                          source.implementation_->backend_basis);
}

ReferenceBasisOverlap
Libint2IntegralProvider::compute_ano_reference_overlap() const {
  for(const libint2::Atom& atom : implementation_->atoms) {
    if(atom.atomic_number != 1 && atom.atomic_number != 8) {
      throw std::invalid_argument(
          "ANO meta-Lowdin compatibility currently supports only H and O");
    }
  }
  libint2::BasisSet reference_basis{
      "ano-rcc", implementation_->atoms, true};
  reference_basis.set_pure(true);
  basis::BasisSet metadata = copy_basis_metadata(
      reference_basis, implementation_->atoms, "ANO-RCC");
  linalg::Matrix overlap = integrals::compute_cross_overlap(
      implementation_->backend_basis, reference_basis);
  return ReferenceBasisOverlap{
      .source_basis = std::move(metadata),
      .target_source_overlap = std::move(overlap),
  };
}

localization::CartesianMomentIntegrals
Libint2IntegralProvider::compute_cartesian_moments_through_fourth() const {
  const auto& basis = implementation_->backend_basis;
  constexpr std::size_t radix = 5;
  const auto index = [](const std::size_t x, const std::size_t y,
                        const std::size_t z) {
    return (x * radix + y) * radix + z;
  };
  std::array<linalg::Matrix, radix * radix * radix> unique;
  for(std::size_t x = 0; x <= 4; ++x)
    for(std::size_t y = 0; y <= 4 - x; ++y)
      for(std::size_t z = 0; z <= 4 - x - y; ++z)
        unique[index(x, y, z)] = compute_cartesian_monomial(
            basis, static_cast<int>(x), static_cast<int>(y),
            static_cast<int>(z));

  const auto reference_overlap = compute_one_body(
      basis, implementation_->atoms, libint2::Operator::overlap);
  const auto& analytic_overlap = unique[index(0, 0, 0)];
  double overlap_error = 0.0;
  for(std::size_t element = 0; element < reference_overlap.size(); ++element)
    overlap_error = std::max(
        overlap_error,
        std::abs(reference_overlap.values()[element] -
                 analytic_overlap.values()[element]));
  if(overlap_error > 2.0e-10)
    throw std::runtime_error(
        "analytic Cartesian moment backend failed its Libint2 overlap check: " +
        std::to_string(overlap_error));

  localization::CartesianMomentIntegrals result;
  for(std::size_t a = 0; a < 3; ++a) {
    std::array<std::size_t, 3> powers{};
    ++powers[a];
    result.first[a] = unique[index(powers[0], powers[1], powers[2])];
    for(std::size_t b = 0; b < 3; ++b) {
      ++powers[b];
      result.second[3 * a + b] =
          unique[index(powers[0], powers[1], powers[2])];
      for(std::size_t c = 0; c < 3; ++c) {
        ++powers[c];
        result.third[(3 * a + b) * 3 + c] =
            unique[index(powers[0], powers[1], powers[2])];
        for(std::size_t d = 0; d < 3; ++d) {
          ++powers[d];
          result.fourth[((3 * a + b) * 3 + c) * 3 + d] =
              unique[index(powers[0], powers[1], powers[2])];
          --powers[d];
        }
        --powers[c];
      }
      --powers[b];
    }
  }
  return result;
}

linalg::Matrix Libint2IntegralProvider::evaluate_basis_values(
    const std::vector<std::array<double, 3>>& points_bohr) const {
  if(points_bohr.empty()) {
    throw std::invalid_argument(
        "basis-function evaluation requires at least one point");
  }
  for(const std::array<double, 3>& point : points_bohr) {
    for(const double coordinate : point) {
      if(!std::isfinite(coordinate)) {
        throw std::invalid_argument(
            "basis-function evaluation point is nonfinite");
      }
    }
  }

  const libint2::BasisSet& basis = implementation_->backend_basis;
  const std::vector<std::size_t>& shell_to_ao = basis.shell2bf();
  linalg::Matrix values{points_bohr.size(),
                        implementation_->metadata.number_of_aos};
  for(std::size_t point_index = 0; point_index < points_bohr.size();
      ++point_index) {
    const std::array<double, 3>& point = points_bohr[point_index];
    for(std::size_t shell_index = 0; shell_index < basis.size();
        ++shell_index) {
      const libint2::Shell& shell = basis[shell_index];
      if(shell.contr.size() != 1) {
        throw std::runtime_error(
            "basis evaluation requires segmented one-contraction shells");
      }
      const libint2::Shell::Contraction& contraction = shell.contr[0];
      const int angular_momentum = contraction.l;
      const double x = point[0] - shell.O[0];
      const double y = point[1] - shell.O[1];
      const double z = point[2] - shell.O[2];
      const double radius_squared = x * x + y * y + z * z;
      double radial = 0.0;
      for(std::size_t primitive = 0; primitive < shell.alpha.size();
          ++primitive) {
        radial += contraction.coeff[primitive] *
                  std::exp(-shell.alpha[primitive] * radius_squared);
      }

      const std::size_t cartesian_count =
          contraction.cartesian_size();
      std::vector<double> cartesian(cartesian_count, 0.0);
      int lx = 0;
      int ly = 0;
      int lz = 0;
      std::size_t cartesian_index = 0;
      FOR_CART(lx, ly, lz, angular_momentum)
      cartesian[cartesian_index] =
          radial * integer_power(x, lx) * integer_power(y, ly) *
          integer_power(z, lz);
      ++cartesian_index;
      END_FOR_CART
      if(cartesian_index != cartesian_count) {
        throw std::runtime_error(
            "Libint2 Cartesian AO ordering produced an inconsistent size");
      }

      const std::size_t first_ao = shell_to_ao[shell_index];
      if(contraction.pure) {
        const auto& transform =
            libint2::solidharmonics::SolidHarmonicsCoefficients<double>::
                instance(static_cast<unsigned int>(angular_momentum));
        const std::size_t pure_count =
            static_cast<std::size_t>(2 * angular_momentum + 1);
        for(std::size_t pure = 0; pure < pure_count; ++pure) {
          double value = 0.0;
          const std::size_t count =
              static_cast<std::size_t>(transform.nnz(pure));
          const auto* indices = transform.row_idx(pure);
          const double* coefficients = transform.row_values(pure);
          for(std::size_t term = 0; term < count; ++term) {
            value += coefficients[term] * cartesian[indices[term]];
          }
          values(point_index, first_ao + pure) = value;
        }
      } else {
        for(std::size_t cartesian_ao = 0;
            cartesian_ao < cartesian_count; ++cartesian_ao) {
          values(point_index, first_ao + cartesian_ao) =
              cartesian[cartesian_ao];
        }
      }
    }
  }
  linalg::require_finite(values, "AO basis values");
  return values;
}

ValidationAoEriTensor Libint2IntegralProvider::compute_validation_ao_eris(
    std::size_t maximum_bytes) const {
  if(maximum_bytes == 0) {
    throw std::invalid_argument(
        "validation AO ERI byte limit must be positive");
  }
  const std::size_t nbf = implementation_->metadata.number_of_aos;
  std::size_t element_count = 1;
  for(int dimension = 0; dimension < 4; ++dimension) {
    if(element_count >
       std::numeric_limits<std::size_t>::max() / nbf) {
      throw std::overflow_error(
          "validation AO ERI element count overflows size_t");
    }
    element_count *= nbf;
  }
  if(element_count >
     std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    throw std::overflow_error(
        "validation AO ERI byte count overflows size_t");
  }
  const std::size_t required_bytes = element_count * sizeof(double);
  if(required_bytes > maximum_bytes) {
    throw std::runtime_error(
        "validation AO ERI tensor requires " +
        std::to_string(required_bytes) +
        " bytes, exceeding the explicit limit of " +
        std::to_string(maximum_bytes) + " bytes");
  }

  ValidationAoEriTensor result{
      .ao_count = nbf,
      .values = std::vector<double>(element_count, 0.0),
  };
  const auto set_value =
      [&](std::size_t mu, std::size_t nu, std::size_t lambda,
          std::size_t sigma, double value) {
        result.values[(((mu * nbf) + nu) * nbf + lambda) * nbf + sigma] =
            value;
      };
  const libint2::BasisSet& basis = implementation_->backend_basis;
  const std::vector<std::size_t>& shell_to_ao = basis.shell2bf();
  libint2::Engine engine{libint2::Operator::coulomb, basis.max_nprim(),
                         static_cast<int>(basis.max_l()), 0, 0.0};
  const std::vector<ShellQuartet> quartets =
      enumerate_unique_shell_quartets(basis.size());
  for(const ShellQuartet& quartet : quartets) {
    const std::size_t shell_1 = quartet.shell_1;
    const std::size_t shell_2 = quartet.shell_2;
    const std::size_t shell_3 = quartet.shell_3;
    const std::size_t shell_4 = quartet.shell_4;
    engine.compute(basis[shell_1], basis[shell_2], basis[shell_3],
                   basis[shell_4]);
    const double* buffer = engine.results()[0];
    if(buffer == nullptr) {
      throw std::runtime_error(
          "validation no-screen Libint2 ERI engine returned a null quartet");
    }
    const std::size_t first_1 = shell_to_ao[shell_1];
    const std::size_t first_2 = shell_to_ao[shell_2];
    const std::size_t first_3 = shell_to_ao[shell_3];
    const std::size_t first_4 = shell_to_ao[shell_4];
    const std::size_t count_1 = basis[shell_1].size();
    const std::size_t count_2 = basis[shell_2].size();
    const std::size_t count_3 = basis[shell_3].size();
    const std::size_t count_4 = basis[shell_4].size();
    std::size_t index = 0;
    for(std::size_t function_1 = 0; function_1 < count_1; ++function_1) {
      const std::size_t mu = first_1 + function_1;
      for(std::size_t function_2 = 0; function_2 < count_2;
          ++function_2) {
        const std::size_t nu = first_2 + function_2;
        for(std::size_t function_3 = 0; function_3 < count_3;
            ++function_3) {
          const std::size_t lambda = first_3 + function_3;
          for(std::size_t function_4 = 0; function_4 < count_4;
              ++function_4, ++index) {
            const std::size_t sigma = first_4 + function_4;
            const double value = buffer[index];
            if(!std::isfinite(value)) {
              throw std::runtime_error(
                  "validation AO ERI tensor contains a nonfinite value");
            }
            set_value(mu, nu, lambda, sigma, value);
            set_value(nu, mu, lambda, sigma, value);
            set_value(mu, nu, sigma, lambda, value);
            set_value(nu, mu, sigma, lambda, value);
            set_value(lambda, sigma, mu, nu, value);
            set_value(sigma, lambda, mu, nu, value);
            set_value(lambda, sigma, nu, mu, value);
            set_value(sigma, lambda, nu, mu, value);
          }
        }
      }
    }
  }
  return result;
}

FirstIndexTransformedEriResult
Libint2IntegralProvider::compute_first_index_transformed_eris(
    const linalg::Matrix& coefficients,
    const std::vector<std::size_t>& orbital_indices,
    const std::size_t maximum_additional_memory_bytes,
    const ParallelBuildOptions& shell_quartet_parallel) const {
  const std::size_t ao_count = implementation_->metadata.number_of_aos;
  if(coefficients.rows() != ao_count || coefficients.columns() == 0) {
    throw std::invalid_argument(
        "first-index transform coefficient dimensions do not match the AO "
        "basis");
  }
  linalg::require_finite(coefficients,
                         "first-index transform coefficients");
  if(maximum_additional_memory_bytes == 0) {
    throw std::invalid_argument(
        "first-index transform memory limit must be positive");
  }
  if(shell_quartet_parallel.ranks == 0 ||
     shell_quartet_parallel.rank >= shell_quartet_parallel.ranks ||
     shell_quartet_parallel.threads != 1) {
    throw std::invalid_argument(
        "first-index shell-quartet topology requires a valid rank and "
        "exactly one thread per caller");
  }
  for(std::size_t index = 0; index < orbital_indices.size(); ++index) {
    if(orbital_indices[index] >= coefficients.columns()) {
      throw std::out_of_range(
          "first-index transform orbital index is out of range");
    }
    if(index != 0 && orbital_indices[index - 1] >= orbital_indices[index]) {
      throw std::invalid_argument(
          "first-index transform orbital indices must be strictly "
          "increasing");
    }
  }

  std::size_t element_count = orbital_indices.size();
  for(int dimension = 0; dimension < 3; ++dimension) {
    if(ao_count != 0 &&
       element_count >
           std::numeric_limits<std::size_t>::max() / ao_count) {
      throw std::overflow_error(
          "first-index transformed ERI element count overflows size_t");
    }
    element_count *= ao_count;
  }
  if(element_count >
     std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    throw std::overflow_error(
        "first-index transformed ERI byte count overflows size_t");
  }
  const std::size_t required_bytes = element_count * sizeof(double);
  if(required_bytes > maximum_additional_memory_bytes) {
    throw std::runtime_error(
        "first-index transformed ERI slice requires " +
        std::to_string(required_bytes) +
        " bytes, exceeding the explicit limit of " +
        std::to_string(maximum_additional_memory_bytes) + " bytes");
  }

  FirstIndexTransformedEriResult result{
      .ao_count = ao_count,
      .orbital_indices = orbital_indices,
      .values = std::vector<double>(element_count, 0.0),
      .estimated_peak_additional_bytes = required_bytes,
      .unique_shell_quartets = 0,
      .evaluated_shell_quartets = 0,
  };
  if(orbital_indices.empty()) {
    return result;
  }

  const libint2::BasisSet& basis = implementation_->backend_basis;
  const std::vector<std::size_t>& shell_to_ao = basis.shell2bf();
  const std::vector<ShellQuartet> quartets =
      enumerate_unique_shell_quartets(basis.size());
  if(quartets.size() >
     static_cast<std::size_t>(
         std::numeric_limits<std::uint64_t>::max())) {
    throw std::overflow_error(
        "first-index transform shell-quartet count exceeds uint64");
  }
  result.unique_shell_quartets =
      static_cast<std::uint64_t>(quartets.size());
  libint2::Engine engine{libint2::Operator::coulomb, basis.max_nprim(),
                         static_cast<int>(basis.max_l()), 0, 0.0};
  const auto value_offset =
      [ao_count](const std::size_t owned_orbital,
                 const std::size_t nu,
                 const std::size_t lambda,
                 const std::size_t sigma) {
        return (((owned_orbital * ao_count) + nu) * ao_count + lambda) *
                   ao_count +
               sigma;
      };

  for(std::size_t quartet_index = shell_quartet_parallel.rank;
      quartet_index < quartets.size();
      quartet_index += shell_quartet_parallel.ranks) {
    const ShellQuartet& quartet = quartets[quartet_index];
    const std::size_t shell_1 = quartet.shell_1;
    const std::size_t shell_2 = quartet.shell_2;
    const std::size_t shell_3 = quartet.shell_3;
    const std::size_t shell_4 = quartet.shell_4;
    engine.compute(basis[shell_1], basis[shell_2], basis[shell_3],
                   basis[shell_4]);
    const double* buffer = engine.results()[0];
    if(buffer == nullptr) {
      throw std::runtime_error(
          "first-index no-screen Libint2 ERI engine returned a null "
          "quartet");
    }
    ++result.evaluated_shell_quartets;
    const std::size_t first_1 = shell_to_ao[shell_1];
    const std::size_t first_2 = shell_to_ao[shell_2];
    const std::size_t first_3 = shell_to_ao[shell_3];
    const std::size_t first_4 = shell_to_ao[shell_4];
    const std::size_t count_1 = basis[shell_1].size();
    const std::size_t count_2 = basis[shell_2].size();
    const std::size_t count_3 = basis[shell_3].size();
    const std::size_t count_4 = basis[shell_4].size();
    std::size_t buffer_index = 0;
    for(std::size_t function_1 = 0; function_1 < count_1;
        ++function_1) {
      const std::size_t mu = first_1 + function_1;
      for(std::size_t function_2 = 0; function_2 < count_2;
          ++function_2) {
        const std::size_t nu = first_2 + function_2;
        for(std::size_t function_3 = 0; function_3 < count_3;
            ++function_3) {
          const std::size_t lambda = first_3 + function_3;
          for(std::size_t function_4 = 0; function_4 < count_4;
              ++function_4, ++buffer_index) {
            const std::size_t sigma = first_4 + function_4;
            const double value = buffer[buffer_index];
            if(!std::isfinite(value)) {
              throw std::runtime_error(
                  "first-index transformed ERI contains a nonfinite shell "
                  "value");
            }
            // Shell-level unique-quartet traversal still visits symmetry-
            // equivalent function quartets when two or more shell indices
            // coincide.  Retain one canonical AO quartet before expanding
            // the eightfold permutational symmetry; otherwise those cases
            // would be accumulated more than once (the dense validation
            // oracle assigns, rather than adds, its symmetry copies).
            if((shell_1 == shell_2 && mu < nu) ||
               (shell_3 == shell_4 && lambda < sigma) ||
               (shell_1 == shell_3 && shell_2 == shell_4 &&
                std::pair{mu, nu} < std::pair{lambda, sigma})) {
              continue;
            }
            const std::array<std::array<std::size_t, 4>, 8> candidates{{
                {mu, nu, lambda, sigma},
                {nu, mu, lambda, sigma},
                {mu, nu, sigma, lambda},
                {nu, mu, sigma, lambda},
                {lambda, sigma, mu, nu},
                {sigma, lambda, mu, nu},
                {lambda, sigma, nu, mu},
                {sigma, lambda, nu, mu},
            }};
            std::array<std::array<std::size_t, 4>, 8> unique{};
            std::size_t unique_count = 0;
            for(const auto& candidate : candidates) {
              bool duplicate = false;
              for(std::size_t prior = 0; prior < unique_count; ++prior) {
                if(unique[prior] == candidate) {
                  duplicate = true;
                  break;
                }
              }
              if(duplicate) {
                continue;
              }
              unique[unique_count++] = candidate;
              for(std::size_t owned = 0; owned < orbital_indices.size();
                  ++owned) {
                result.values[value_offset(owned, candidate[1], candidate[2],
                                           candidate[3])] +=
                    coefficients(candidate[0], orbital_indices[owned]) *
                    value;
              }
            }
          }
        }
      }
    }
  }
  for(const double value : result.values) {
    if(!std::isfinite(value)) {
      throw std::runtime_error(
          "first-index transformed ERI slice contains a nonfinite value");
    }
  }
  return result;
}

void Libint2IntegralProvider::for_each_batched_first_index_ao_block(
    const std::vector<linalg::Matrix>& occupied_coefficients_by_point,
    const BatchedFirstIndexAoOptions& options,
    const BatchedFirstIndexAoBlockConsumer& consume) const {
  const std::size_t ao_count = implementation_->metadata.number_of_aos;
  if(occupied_coefficients_by_point.empty() || !consume ||
     options.occupied_count == 0 ||
     options.occupied_first >
         occupied_coefficients_by_point.front().columns() ||
     options.occupied_count >
         occupied_coefficients_by_point.front().columns() -
             options.occupied_first ||
     options.target_lambda_ao_block_size == 0 || options.threads == 0 ||
     !std::isfinite(options.schwarz_threshold) ||
     options.schwarz_threshold < 0.0 ||
     options.maximum_additional_memory_bytes == 0) {
    throw std::invalid_argument(
        "batched first-index AO transform options are invalid");
  }
  for(const auto& coefficients : occupied_coefficients_by_point) {
    if(coefficients.rows() != ao_count ||
       coefficients.columns() !=
           occupied_coefficients_by_point.front().columns()) {
      throw std::invalid_argument(
          "batched first-index AO coefficient dimensions are inconsistent");
    }
    linalg::require_finite(
        coefficients, "batched first-index AO coefficients");
  }
#ifndef MODERNQC_HAS_OPENMP
  if(options.threads != 1U) {
    throw std::invalid_argument(
        "batched first-index AO threads require OpenMP support");
  }
#else
  if(options.threads >
     static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "batched first-index AO thread count exceeds int");
  }
#endif

  const auto checked_product =
      [](std::initializer_list<std::size_t> factors,
         const char* description) {
        std::size_t result = 1;
        for(const std::size_t factor : factors) {
          if(factor == 0 ||
             result > std::numeric_limits<std::size_t>::max() / factor) {
            throw std::overflow_error(
                std::string{description} + " overflows size_t");
          }
          result *= factor;
        }
        return result;
      };
  const auto checked_bytes =
      [](std::size_t elements, const char* description) {
        if(elements >
           std::numeric_limits<std::size_t>::max() / sizeof(double)) {
          throw std::overflow_error(
              std::string{description} + " overflows size_t");
        }
        return elements * sizeof(double);
      };
  const auto checked_add =
      [](std::size_t left, std::size_t right,
         const char* description) {
        if(left > std::numeric_limits<std::size_t>::max() - right) {
          throw std::overflow_error(
              std::string{description} + " overflows size_t");
        }
        return left + right;
      };

  const libint2::BasisSet& basis = implementation_->backend_basis;
  const std::vector<std::size_t>& shell_to_ao = basis.shell2bf();
  const std::size_t shell_count = basis.size();
  std::size_t lambda_shell_first = 0;
  std::size_t block_ordinal = 0;
  while(lambda_shell_first < shell_count) {
    std::size_t lambda_shell_count = 0;
    std::size_t lambda_ao_count = 0;
    while(lambda_shell_first + lambda_shell_count < shell_count) {
      const std::size_t next_count =
          basis[lambda_shell_first + lambda_shell_count].size();
      if(lambda_shell_count != 0 &&
         lambda_ao_count + next_count >
             options.target_lambda_ao_block_size) {
        break;
      }
      lambda_ao_count += next_count;
      ++lambda_shell_count;
    }
    const std::size_t lambda_ao_first =
        shell_to_ao[lambda_shell_first];
    const std::size_t elements_per_point = checked_product(
        {lambda_ao_count, ao_count, options.occupied_count, ao_count},
        "batched first-index AO block elements");
    const std::size_t all_point_elements = checked_product(
        {occupied_coefficients_by_point.size(), elements_per_point},
        "batched first-index AO all-point elements");
    std::size_t maximum_shell_remainder = 0;
    for(std::size_t lambda_shell = lambda_shell_first;
        lambda_shell < lambda_shell_first + lambda_shell_count;
        ++lambda_shell) {
      for(std::size_t nu_shell = 0; nu_shell < shell_count;
          ++nu_shell) {
        for(std::size_t sigma_shell = 0; sigma_shell < shell_count;
            ++sigma_shell) {
          maximum_shell_remainder = std::max(
              maximum_shell_remainder,
              checked_product(
                  {basis[nu_shell].size(), basis[lambda_shell].size(),
                   basis[sigma_shell].size()},
                  "batched first-index AO shell remainder"));
        }
      }
    }
    const std::size_t thread_elements = checked_product(
        {options.threads, occupied_coefficients_by_point.size(),
         options.occupied_count, maximum_shell_remainder},
        "batched first-index AO worker elements");
    const std::size_t estimated_peak = checked_add(
        checked_bytes(all_point_elements,
                      "batched first-index AO block storage"),
        checked_bytes(thread_elements,
                      "batched first-index AO worker storage"),
        "batched first-index AO peak bytes");
    if(estimated_peak > options.maximum_additional_memory_bytes) {
      throw std::runtime_error(
          "batched first-index AO block requires " +
          std::to_string(estimated_peak) +
          " bytes, exceeding the explicit limit of " +
          std::to_string(options.maximum_additional_memory_bytes));
    }
    if(options.report_progress) {
      std::cerr
          << "progress batched four-center AO block "
          << (block_ordinal + 1U) << " started; lambda_ao_first="
          << lambda_ao_first << " lambda_ao_count=" << lambda_ao_count
          << " quadrature_points="
          << occupied_coefficients_by_point.size()
          << " estimated_peak_bytes=" << estimated_peak << '\n';
    }
    const auto block_start = std::chrono::steady_clock::now();
    std::vector<std::vector<double>> point_values(
        occupied_coefficients_by_point.size(),
        std::vector<double>(elements_per_point, 0.0));
    std::vector<std::exception_ptr> thread_errors(options.threads);
    std::vector<std::uint64_t> thread_evaluated(options.threads, 0U);
    std::vector<std::uint64_t> thread_screened(options.threads, 0U);
    const auto transform_block =
        [&](std::size_t thread, std::size_t actual_threads,
            libint2::Engine& engine) {
          std::vector<double> shell_transformed;
          const std::size_t shell_tasks = checked_product(
              {shell_count, lambda_shell_count},
              "batched first-index AO shell tasks");
          for(std::size_t task = thread; task < shell_tasks;
              task += actual_threads) {
            const std::size_t shell_2 = task / lambda_shell_count;
            const std::size_t shell_3 =
                lambda_shell_first + task % lambda_shell_count;
            const std::size_t first_2 = shell_to_ao[shell_2];
            const std::size_t first_3 = shell_to_ao[shell_3];
            const std::size_t count_2 = basis[shell_2].size();
            const std::size_t count_3 = basis[shell_3].size();
            for(std::size_t shell_1 = 0; shell_1 < shell_count;
                ++shell_1) {
              const std::size_t first_1 = shell_to_ao[shell_1];
              const std::size_t count_1 = basis[shell_1].size();
              for(std::size_t shell_4 = 0; shell_4 < shell_count;
                  ++shell_4) {
                const double schwarz_bound =
                    implementation_->shell_pair_schwarz_bounds
                        [shell_1 * shell_count + shell_2] *
                    implementation_->shell_pair_schwarz_bounds
                        [shell_3 * shell_count + shell_4];
                if(options.schwarz_threshold > 0.0 &&
                   schwarz_bound < options.schwarz_threshold) {
                  ++thread_screened[thread];
                  continue;
                }
                ++thread_evaluated[thread];
                engine.compute(basis[shell_1], basis[shell_2],
                               basis[shell_3], basis[shell_4]);
                const double* buffer = engine.results()[0];
                if(buffer == nullptr) {
                  throw std::runtime_error(
                      "batched first-index AO engine returned null");
                }
                const std::size_t first_4 = shell_to_ao[shell_4];
                const std::size_t count_4 = basis[shell_4].size();
                const std::size_t remainder = checked_product(
                    {count_2, count_3, count_4},
                    "batched first-index AO shell remainder");
                shell_transformed.assign(
                    checked_product(
                        {occupied_coefficients_by_point.size(),
                         options.occupied_count, remainder},
                        "batched first-index AO transformed shell"),
                    0.0);
                for(std::size_t point = 0;
                    point < occupied_coefficients_by_point.size();
                    ++point) {
                  const auto& coefficients =
                      occupied_coefficients_by_point[point];
                  for(std::size_t local_i = 0;
                      local_i < options.occupied_count; ++local_i) {
                    double* transformed =
                        shell_transformed.data() +
                        (point * options.occupied_count + local_i) *
                            remainder;
                    for(std::size_t function_1 = 0;
                        function_1 < count_1; ++function_1) {
                      const double coefficient = coefficients(
                          first_1 + function_1,
                          options.occupied_first + local_i);
                      const double* shell_values =
                          buffer + function_1 * remainder;
#if defined(MODERNQC_HAS_OPENMP) && defined(__GNUC__) && \
    !defined(__clang__)
#pragma omp simd
#endif
                      for(std::size_t rest = 0; rest < remainder; ++rest) {
                        transformed[rest] +=
                            coefficient * shell_values[rest];
                      }
                    }
                    std::vector<double>& output = point_values[point];
                    for(std::size_t function_2 = 0;
                        function_2 < count_2; ++function_2) {
                      const std::size_t nu = first_2 + function_2;
                      for(std::size_t function_3 = 0;
                          function_3 < count_3; ++function_3) {
                        const std::size_t lambda_local =
                            first_3 + function_3 - lambda_ao_first;
                        for(std::size_t function_4 = 0;
                            function_4 < count_4; ++function_4) {
                          const std::size_t sigma = first_4 + function_4;
                          const std::size_t rest =
                              ((function_2 * count_3) + function_3) *
                                  count_4 +
                              function_4;
                          const std::size_t output_index =
                              ((((lambda_local * ao_count + sigma) *
                                      options.occupied_count +
                                  local_i) *
                                     ao_count) +
                               nu);
                          output[output_index] += transformed[rest];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        };
#ifdef MODERNQC_HAS_OPENMP
#pragma omp parallel num_threads(static_cast<int>(options.threads))
    {
      const std::size_t thread =
          static_cast<std::size_t>(omp_get_thread_num());
      const std::size_t actual_threads =
          static_cast<std::size_t>(omp_get_num_threads());
      try {
        libint2::Engine engine{
            libint2::Operator::coulomb, basis.max_nprim(),
            static_cast<int>(basis.max_l()), 0, 0.0};
        transform_block(thread, actual_threads, engine);
      } catch(...) {
        thread_errors[thread] = std::current_exception();
      }
    }
#else
    try {
      libint2::Engine engine{
          libint2::Operator::coulomb, basis.max_nprim(),
          static_cast<int>(basis.max_l()), 0, 0.0};
      transform_block(0U, 1U, engine);
    } catch(...) {
      thread_errors[0] = std::current_exception();
    }
#endif
    for(const auto& error : thread_errors) {
      if(error) {
        std::rethrow_exception(error);
      }
    }
    std::uint64_t evaluated = 0;
    std::uint64_t screened = 0;
    for(std::size_t thread = 0; thread < options.threads; ++thread) {
      evaluated += thread_evaluated[thread];
      screened += thread_screened[thread];
    }
    const std::size_t ordered = checked_product(
        {shell_count, shell_count, lambda_shell_count, shell_count},
        "batched first-index AO ordered quartets");
    if(evaluated + screened != static_cast<std::uint64_t>(ordered)) {
      throw std::runtime_error(
          "batched first-index AO quartet counters are inconsistent");
    }
    if(options.report_progress) {
      std::cerr << "progress batched four-center AO block "
                << (block_ordinal + 1U)
                << " complete; evaluated_shell_quartets=" << evaluated
                << " screened_shell_quartets=" << screened
                << " seconds="
                << std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - block_start)
                       .count()
                << '\n';
    }
    consume(BatchedFirstIndexAoBlock{
        .ao_count = ao_count,
        .lambda_ao_first = lambda_ao_first,
        .lambda_ao_count = lambda_ao_count,
        .occupied_count = options.occupied_count,
        .point_values = std::move(point_values),
        .estimated_peak_additional_bytes = estimated_peak,
        .ordered_shell_quartets = static_cast<std::uint64_t>(ordered),
        .evaluated_shell_quartets = evaluated,
        .schwarz_screened_shell_quartets = screened,
    });
    lambda_shell_first += lambda_shell_count;
    ++block_ordinal;
  }
}

OvovTileResult Libint2IntegralProvider::compute_ovov_tile(
    const linalg::Matrix& occupied_coefficients,
    const linalg::Matrix& virtual_coefficients,
    const OvovTileOptions& options) const {
  const std::size_t ao_count =
      implementation_->metadata.number_of_aos;
  if(occupied_coefficients.rows() != ao_count ||
     virtual_coefficients.rows() != ao_count ||
     occupied_coefficients.columns() == 0 ||
     virtual_coefficients.columns() == 0) {
    throw std::invalid_argument(
        "OVOV tile coefficient dimensions do not match the AO basis");
  }
  linalg::require_finite(occupied_coefficients,
                         "OVOV occupied coefficients");
  linalg::require_finite(virtual_coefficients,
                         "OVOV virtual coefficients");
  const auto range_is_invalid =
      [](std::size_t first, std::size_t count,
         std::size_t extent) {
        return count == 0 || first > extent || count > extent - first;
      };
  if(range_is_invalid(options.left_occupied_first,
                      options.left_occupied_count,
                      occupied_coefficients.columns()) ||
     range_is_invalid(options.right_occupied_first,
                      options.right_occupied_count,
                      occupied_coefficients.columns()) ||
     range_is_invalid(options.left_virtual_first,
                      options.left_virtual_count,
                      virtual_coefficients.columns()) ||
     range_is_invalid(options.right_virtual_first,
     options.right_virtual_count,
                      virtual_coefficients.columns()) ||
     options.threads == 0 ||
     !std::isfinite(options.schwarz_threshold) ||
     options.schwarz_threshold < 0.0 ||
     options.maximum_additional_memory_bytes == 0) {
    throw std::invalid_argument(
        "OVOV tile ranges, thread count, or memory limit are invalid");
  }
#ifndef MODERNQC_HAS_OPENMP
  if(options.threads != 1) {
    throw std::invalid_argument(
        "multiple OVOV transform threads require OpenMP support");
  }
#else
  if(options.threads >
     static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "OVOV transform thread count exceeds the OpenMP integer range");
  }
#endif

  const auto checked_product =
      [](std::initializer_list<std::size_t> factors,
         const char* name) {
        std::size_t product = 1;
        for(const std::size_t factor : factors) {
          if(factor == 0 ||
             product >
                 std::numeric_limits<std::size_t>::max() / factor) {
            throw std::overflow_error(
                std::string{name} + " element count overflows size_t");
          }
          product *= factor;
        }
        return product;
      };
  const auto checked_bytes =
      [](std::size_t elements, const char* name) {
        if(elements >
           std::numeric_limits<std::size_t>::max() / sizeof(double)) {
          throw std::overflow_error(
              std::string{name} + " byte count overflows size_t");
        }
        return elements * sizeof(double);
      };
  const auto checked_add =
      [](std::size_t& total, std::size_t value, const char* name) {
        if(total >
           std::numeric_limits<std::size_t>::max() - value) {
          throw std::overflow_error(
              std::string{name} + " byte count overflows size_t");
        }
        total += value;
      };

  // Prefer a conventional separable four-pass AO-to-MO transformation when
  // its two largest intermediates fit the caller's explicit memory limit.
  // The pair-matrix fallback below contracts both ket AO indices into (j,b)
  // simultaneously and is useful only when these staged intermediates do not
  // fit; its arithmetic cost is much higher for a complete virtual space.
  const libint2::BasisSet& staged_basis = implementation_->backend_basis;
  const std::vector<std::size_t>& staged_shell_to_ao =
      staged_basis.shell2bf();
  const std::size_t staged_shell_count = staged_basis.size();
  const std::size_t staged_left_pairs = checked_product(
      {options.left_occupied_count, options.left_virtual_count},
      "staged OVOV left pairs");
  const std::size_t staged_right_pairs = checked_product(
      {options.right_occupied_count, options.right_virtual_count},
      "staged OVOV right pairs");
  const std::size_t ao_squared =
      checked_product({ao_count, ao_count}, "staged OVOV AO squared");
  const std::size_t ao_cubed =
      checked_product({ao_squared, ao_count}, "staged OVOV AO cubed");
  const std::size_t first_stage_elements = checked_product(
      {options.left_occupied_count, ao_cubed},
      "staged OVOV first-index intermediate");
  const std::size_t second_stage_elements = checked_product(
      {staged_left_pairs, ao_squared},
      "staged OVOV second-index intermediate");
  const std::size_t staged_result_elements = checked_product(
      {staged_left_pairs, staged_right_pairs},
      "staged OVOV result");
  std::size_t maximum_shell_functions = 0;
  for(const auto& shell : staged_basis) {
    maximum_shell_functions =
        std::max(maximum_shell_functions, shell.size());
  }
  const std::size_t maximum_quartet_rest = checked_product(
      {maximum_shell_functions, maximum_shell_functions,
       maximum_shell_functions},
      "staged OVOV shell-quartet remainder");
  const std::size_t first_thread_elements = checked_product(
      {maximum_quartet_rest, options.left_occupied_count},
      "staged OVOV first-pass thread workspace");
  const std::size_t final_thread_elements =
      checked_product({ao_count, options.right_occupied_count},
                      "staged OVOV third-pass thread workspace") +
      checked_product({options.right_virtual_count,
                       options.right_occupied_count},
                      "staged OVOV fourth-pass thread workspace");
  const std::size_t first_thread_bytes = checked_bytes(
      checked_product({options.threads, first_thread_elements},
                      "staged OVOV first-pass worker storage"),
      "staged OVOV first-pass worker storage");
  const std::size_t final_thread_bytes = checked_bytes(
      checked_product({options.threads, final_thread_elements},
                      "staged OVOV final-pass worker storage"),
      "staged OVOV final-pass worker storage");
  std::size_t staged_first_peak = 0;
  checked_add(staged_first_peak,
              checked_bytes(first_stage_elements,
                            "staged OVOV first-index intermediate"),
              "staged OVOV first peak");
  checked_add(staged_first_peak,
              checked_bytes(second_stage_elements,
                            "staged OVOV second-index intermediate"),
              "staged OVOV first peak");
  checked_add(staged_first_peak, first_thread_bytes,
              "staged OVOV first peak");
  std::size_t staged_final_peak = 0;
  checked_add(staged_final_peak,
              checked_bytes(second_stage_elements,
                            "staged OVOV second-index intermediate"),
              "staged OVOV final peak");
  checked_add(staged_final_peak,
              checked_bytes(staged_result_elements,
                            "staged OVOV result"),
              "staged OVOV final peak");
  checked_add(staged_final_peak, final_thread_bytes,
              "staged OVOV final peak");
  const std::size_t staged_peak_bytes =
      std::max(staged_first_peak, staged_final_peak);

  if(staged_peak_bytes <= options.maximum_additional_memory_bytes) {
    const auto report_staged_progress =
        [&](const char* message) {
          if(options.report_progress) {
            std::cerr << "progress " << message << '\n';
          }
        };
    if(options.report_progress) {
      std::cerr << "progress separable four-pass OVOV planned; "
                << "first_intermediate_bytes="
                << checked_bytes(
                       first_stage_elements,
                       "staged OVOV first-index intermediate")
                << " second_intermediate_bytes="
                << checked_bytes(
                       second_stage_elements,
                       "staged OVOV second-index intermediate")
                << " estimated_peak_bytes=" << staged_peak_bytes << '\n';
    }
    const std::size_t fused_second_rows = checked_product(
        {options.left_occupied_count, ao_squared},
        "staged OVOV fused second-index rows");
    if(fused_second_rows >
       static_cast<std::size_t>(
           std::numeric_limits<std::ptrdiff_t>::max())) {
      throw std::overflow_error(
          "staged OVOV fused row count exceeds ptrdiff_t");
    }

    linalg::Matrix second_stage;
    std::array<std::uint64_t, 2> quartet_counters{};
    {
      const auto first_index_start =
          std::chrono::steady_clock::now();
      report_staged_progress(
          "separable four-pass OVOV first-index transform started");
      std::vector<double> first_stage(first_stage_elements, 0.0);
      std::vector<std::exception_ptr> thread_errors(options.threads);
      std::vector<std::uint64_t> thread_evaluated(options.threads, 0U);
      std::vector<std::uint64_t> thread_screened(options.threads, 0U);
      const auto transform_first_index =
          [&](std::size_t thread, std::size_t actual_threads,
              libint2::Engine& engine) {
            std::vector<double> shell_transformed;
            const std::size_t shell_pair_tasks = checked_product(
                {staged_shell_count, staged_shell_count},
                "staged OVOV first-pass shell tasks");
            for(std::size_t task = thread; task < shell_pair_tasks;
                task += actual_threads) {
              const std::size_t shell_2 = task / staged_shell_count;
              const std::size_t shell_3 = task % staged_shell_count;
              const std::size_t first_2 = staged_shell_to_ao[shell_2];
              const std::size_t first_3 = staged_shell_to_ao[shell_3];
              const std::size_t count_2 = staged_basis[shell_2].size();
              const std::size_t count_3 = staged_basis[shell_3].size();
              for(std::size_t shell_1 = 0;
                  shell_1 < staged_shell_count; ++shell_1) {
                const std::size_t first_1 =
                    staged_shell_to_ao[shell_1];
                const std::size_t count_1 =
                    staged_basis[shell_1].size();
                for(std::size_t shell_4 = 0;
                    shell_4 < staged_shell_count; ++shell_4) {
                  const double schwarz_bound =
                      implementation_->shell_pair_schwarz_bounds
                          [shell_1 * staged_shell_count + shell_2] *
                      implementation_->shell_pair_schwarz_bounds
                          [shell_3 * staged_shell_count + shell_4];
                  if(options.schwarz_threshold > 0.0 &&
                     schwarz_bound < options.schwarz_threshold) {
                    ++thread_screened[thread];
                    continue;
                  }
                  ++thread_evaluated[thread];
                  engine.compute(staged_basis[shell_1],
                                 staged_basis[shell_2],
                                 staged_basis[shell_3],
                                 staged_basis[shell_4]);
                  const double* buffer = engine.results()[0];
                  if(buffer == nullptr) {
                    throw std::runtime_error(
                        "staged OVOV engine returned a null quartet");
                  }
                  const std::size_t first_4 =
                      staged_shell_to_ao[shell_4];
                  const std::size_t count_4 =
                      staged_basis[shell_4].size();
                  const std::size_t rest = checked_product(
                      {count_2, count_3, count_4},
                      "staged OVOV shell remainder");
                  shell_transformed.assign(checked_product(
                      {rest, options.left_occupied_count},
                      "staged OVOV transformed shell block"),
                      0.0);
                  // These shell contractions execute inside the outer
                  // OpenMP region. Keep them as deterministic, vectorizable
                  // loops so an OpenMP-threaded BLAS library is never entered
                  // concurrently by many workers. The only threaded BLAS
                  // operation in this path is the large fused second pass.
                  for(std::size_t local_i = 0;
                      local_i < options.left_occupied_count; ++local_i) {
                    double* transformed =
                        shell_transformed.data() + local_i * rest;
                    for(std::size_t function_1 = 0;
                        function_1 < count_1; ++function_1) {
                      const double coefficient = occupied_coefficients(
                          first_1 + function_1,
                          options.left_occupied_first + local_i);
                      const double* shell_values =
                          buffer + function_1 * rest;
#if defined(MODERNQC_HAS_OPENMP) && defined(__GNUC__) && \
    !defined(__clang__)
#pragma omp simd
#endif
                      for(std::size_t rest_index = 0;
                          rest_index < rest; ++rest_index) {
                        transformed[rest_index] +=
                            coefficient * shell_values[rest_index];
                      }
                    }
                    for(std::size_t function_2 = 0;
                        function_2 < count_2; ++function_2) {
                      const std::size_t nu = first_2 + function_2;
                      for(std::size_t function_3 = 0;
                          function_3 < count_3; ++function_3) {
                        const std::size_t lambda = first_3 + function_3;
                        for(std::size_t function_4 = 0;
                            function_4 < count_4; ++function_4) {
                          const std::size_t sigma = first_4 + function_4;
                          const std::size_t rest_index =
                              ((function_2 * count_3) + function_3) *
                                  count_4 +
                              function_4;
                          const std::size_t output_index =
                              (((nu * options.left_occupied_count + local_i) *
                                    ao_count +
                                lambda) *
                                   ao_count) +
                              sigma;
                          first_stage[output_index] +=
                              shell_transformed
                                  [local_i * rest + rest_index];
                        }
                      }
                    }
                  }
                }
              }
            }
          };
#ifdef MODERNQC_HAS_OPENMP
#pragma omp parallel num_threads(static_cast<int>(options.threads))
      {
        const std::size_t thread =
            static_cast<std::size_t>(omp_get_thread_num());
        const std::size_t actual_threads =
            static_cast<std::size_t>(omp_get_num_threads());
        try {
          libint2::Engine engine{
              libint2::Operator::coulomb, staged_basis.max_nprim(),
              static_cast<int>(staged_basis.max_l()), 0, 0.0};
          transform_first_index(thread, actual_threads, engine);
        } catch(...) {
          thread_errors[thread] = std::current_exception();
        }
      }
#else
      try {
        libint2::Engine engine{
            libint2::Operator::coulomb, staged_basis.max_nprim(),
            static_cast<int>(staged_basis.max_l()), 0, 0.0};
        transform_first_index(0U, 1U, engine);
      } catch(...) {
        thread_errors[0] = std::current_exception();
      }
#endif
      for(const auto& error : thread_errors) {
        if(error) {
          std::rethrow_exception(error);
        }
      }
      for(std::size_t thread = 0; thread < options.threads; ++thread) {
        quartet_counters[0] += thread_evaluated[thread];
        quartet_counters[1] += thread_screened[thread];
      }
      if(options.report_progress) {
        std::cerr
            << "progress separable four-pass OVOV first-index transform "
               "complete; seconds="
            << std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - first_index_start)
                   .count()
            << '\n';
      }

      const auto second_index_start =
          std::chrono::steady_clock::now();
      report_staged_progress(
          "separable four-pass OVOV second-index blocked OpenMP "
          "transform started");
      second_stage = linalg::Matrix{
          fused_second_rows, options.left_virtual_count};
      // Store the first intermediate as [(nu,i),(lambda,sigma)].  All
      // occupied blocks can then share one cache-blocked matrix multiplication
      // instead of entering BLAS concurrently from independent OpenMP
      // workers. The
      // result is ordered [a,i,lambda,sigma]; the final passes map it back to
      // the public [i,a,j,b] order.
      constexpr std::size_t row_block_size = 256;
      constexpr std::size_t column_block_size = 32;
      const std::size_t row_blocks =
          (fused_second_rows + row_block_size - 1U) / row_block_size;
      if(row_blocks > static_cast<std::size_t>(
                          std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::overflow_error(
            "staged OVOV fused row-block count exceeds ptrdiff_t");
      }
      const double* canonical_virtual =
          virtual_coefficients.data() +
          options.left_virtual_first * ao_count;
#ifdef MODERNQC_HAS_OPENMP
#pragma omp parallel for schedule(static) \
    num_threads(static_cast<int>(options.threads))
#endif
      for(std::ptrdiff_t raw_block = 0;
          raw_block < static_cast<std::ptrdiff_t>(row_blocks);
          ++raw_block) {
        const std::size_t row_first =
            static_cast<std::size_t>(raw_block) * row_block_size;
        const std::size_t row_count = std::min(
            row_block_size, fused_second_rows - row_first);
        for(std::size_t column_first = 0;
            column_first < options.left_virtual_count;
            column_first += column_block_size) {
          const std::size_t column_count = std::min(
              column_block_size,
              options.left_virtual_count - column_first);
          for(std::size_t column = 0; column < column_count; ++column) {
            double* output = second_stage.data() +
                             (column_first + column) * fused_second_rows +
                             row_first;
            std::fill_n(output, row_count, 0.0);
          }
          for(std::size_t nu = 0; nu < ao_count; ++nu) {
            const double* input =
                first_stage.data() + nu * fused_second_rows + row_first;
            for(std::size_t column = 0; column < column_count; ++column) {
              const std::size_t virtual_column = column_first + column;
              const double scale =
                  canonical_virtual[virtual_column * ao_count + nu];
              double* output = second_stage.data() +
                               virtual_column * fused_second_rows +
                               row_first;
#if defined(MODERNQC_HAS_OPENMP) && defined(__GNUC__) && \
    !defined(__clang__)
#pragma omp simd
#endif
              for(std::size_t row = 0; row < row_count; ++row) {
                output[row] += input[row] * scale;
              }
            }
          }
        }
      }
      if(options.report_progress) {
        std::cerr
            << "progress separable four-pass OVOV second-index blocked "
               "OpenMP "
               "transform complete; seconds="
            << std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - second_index_start)
                   .count()
            << '\n';
      }
    }

    const auto final_indices_start =
        std::chrono::steady_clock::now();
    report_staged_progress(
        "separable four-pass OVOV final two SIMD transforms started");
    std::vector<double> transformed_values(
        staged_result_elements, 0.0);
    std::vector<std::exception_ptr> final_thread_errors(options.threads);
    const auto transform_final_indices =
        [&](std::size_t thread, std::size_t actual_threads) {
          std::vector<double> third_stage(
              ao_count * options.right_occupied_count, 0.0);
          std::vector<double> fourth_stage(
              options.right_virtual_count *
                  options.right_occupied_count,
              0.0);
          for(std::size_t source_left_pair = thread;
              source_left_pair < staged_left_pairs;
              source_left_pair += actual_threads) {
            const std::size_t local_a =
                source_left_pair / options.left_occupied_count;
            const std::size_t local_i =
                source_left_pair % options.left_occupied_count;
            const std::size_t left_pair =
                local_i * options.left_virtual_count + local_a;
            const double* source =
                second_stage.data() + source_left_pair * ao_squared;
            for(std::size_t local_j = 0;
                local_j < options.right_occupied_count; ++local_j) {
              double* third =
                  third_stage.data() + local_j * ao_count;
              for(std::size_t sigma = 0; sigma < ao_count; ++sigma) {
                double sum = 0.0;
#if defined(MODERNQC_HAS_OPENMP) && defined(__GNUC__) && \
    !defined(__clang__)
#pragma omp simd reduction(+ : sum)
#endif
                for(std::size_t lambda = 0; lambda < ao_count; ++lambda) {
                  sum += source[lambda * ao_count + sigma] *
                         occupied_coefficients(
                             lambda,
                             options.right_occupied_first + local_j);
                }
                third[sigma] = sum;
              }
              double* fourth = fourth_stage.data() +
                               local_j * options.right_virtual_count;
              for(std::size_t local_b = 0;
                  local_b < options.right_virtual_count; ++local_b) {
                double sum = 0.0;
#if defined(MODERNQC_HAS_OPENMP) && defined(__GNUC__) && \
    !defined(__clang__)
#pragma omp simd reduction(+ : sum)
#endif
                for(std::size_t sigma = 0; sigma < ao_count; ++sigma) {
                  sum += virtual_coefficients(
                             sigma,
                             options.right_virtual_first + local_b) *
                         third[sigma];
                }
                fourth[local_b] = sum;
              }
            }
            for(std::size_t local_j = 0;
                local_j < options.right_occupied_count; ++local_j) {
              for(std::size_t local_b = 0;
                  local_b < options.right_virtual_count; ++local_b) {
                const std::size_t right_pair =
                    local_j * options.right_virtual_count + local_b;
                transformed_values
                    [right_pair * staged_left_pairs + left_pair] =
                    fourth_stage
                        [local_j * options.right_virtual_count + local_b];
              }
            }
          }
        };
#ifdef MODERNQC_HAS_OPENMP
#pragma omp parallel num_threads(static_cast<int>(options.threads))
    {
      const std::size_t thread =
          static_cast<std::size_t>(omp_get_thread_num());
      const std::size_t actual_threads =
          static_cast<std::size_t>(omp_get_num_threads());
      try {
        transform_final_indices(thread, actual_threads);
      } catch(...) {
        final_thread_errors[thread] = std::current_exception();
      }
    }
#else
    try {
      transform_final_indices(0U, 1U);
    } catch(...) {
      final_thread_errors[0] = std::current_exception();
    }
#endif
    for(const auto& error : final_thread_errors) {
      if(error) {
        std::rethrow_exception(error);
      }
    }
    if(options.report_progress) {
      std::cerr
          << "progress separable four-pass OVOV final two transforms "
             "complete; seconds="
          << std::chrono::duration<double>(
                 std::chrono::steady_clock::now() - final_indices_start)
                 .count()
          << '\n';
    }
    for(const double value : transformed_values) {
      if(!std::isfinite(value)) {
        throw std::runtime_error(
            "staged OVOV result contains a nonfinite value");
      }
    }
    const std::size_t ordered_quartets = checked_product(
        {staged_shell_count, staged_shell_count, staged_shell_count,
         staged_shell_count},
        "staged OVOV ordered shell quartets");
    if(quartet_counters[0] + quartet_counters[1] !=
       static_cast<std::uint64_t>(ordered_quartets)) {
      throw std::runtime_error(
          "staged OVOV shell-quartet counters are inconsistent");
    }
    std::vector<double> retained_left_pair_ao_intermediate;
    if(options.retain_left_pair_ao_intermediate) {
      retained_left_pair_ao_intermediate =
          std::move(second_stage.values());
    }
    return OvovTileResult{
        .left_occupied_count = options.left_occupied_count,
        .left_virtual_count = options.left_virtual_count,
        .right_occupied_count = options.right_occupied_count,
        .right_virtual_count = options.right_virtual_count,
        .values = std::move(transformed_values),
        .left_pair_ao_intermediate =
            std::move(retained_left_pair_ao_intermediate),
        .ao_count = ao_count,
        .estimated_peak_additional_bytes = staged_peak_bytes,
        .ordered_shell_quartets =
            static_cast<std::uint64_t>(ordered_quartets),
        .evaluated_shell_quartets = quartet_counters[0],
        .schwarz_screened_shell_quartets = quartet_counters[1],
    };
  }
  if(options.require_separable_four_pass ||
     options.retain_left_pair_ao_intermediate) {
    throw std::runtime_error(
        "separable four-pass OVOV transformation requires " +
        std::to_string(staged_peak_bytes) +
        " bytes, exceeding the explicit limit of " +
        std::to_string(options.maximum_additional_memory_bytes) +
        " bytes; the higher-scaling pair-matrix fallback is disabled");
  }

  if(ao_count == std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("OVOV unique AO-pair extent overflows size_t");
  }
  const std::size_t unique_ao_pair_count =
      checked_product({ao_count, ao_count + 1U},
                      "OVOV unique AO-pair space") /
      2U;
  const libint2::BasisSet& basis = implementation_->backend_basis;
  const std::vector<std::size_t>& shell_to_ao = basis.shell2bf();
  const std::size_t shell_count = basis.size();
  if(shell_count == std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("OVOV unique shell-pair extent overflows size_t");
  }
  const std::size_t unique_shell_pair_count =
      checked_product({shell_count, shell_count + 1U},
                      "OVOV unique shell-pair space") /
      2U;
  const std::size_t transform_quartet_count =
      checked_product({unique_shell_pair_count, unique_shell_pair_count},
                      "OVOV shell-pair transform quartets");
  if(transform_quartet_count >
     static_cast<std::size_t>(
         std::numeric_limits<std::uint64_t>::max())) {
    throw std::overflow_error(
        "OVOV shell-pair transform quartet count exceeds uint64");
  }
  const std::size_t left_pair_count = checked_product(
      {options.left_occupied_count, options.left_virtual_count},
      "OVOV left pair tile");
  const std::size_t right_pair_count = checked_product(
      {options.right_occupied_count, options.right_virtual_count},
      "OVOV right pair tile");
  const std::size_t right_pair_elements = checked_product(
      {unique_ao_pair_count, right_pair_count},
      "OVOV right pair coefficients");
  const std::size_t intermediate_elements = right_pair_elements;
  const std::size_t result_elements = checked_product(
      {left_pair_count, right_pair_count}, "OVOV result tile");
  std::size_t maximum_shell_pair_functions = 0;
  for(std::size_t shell_1 = 0; shell_1 < shell_count; ++shell_1) {
    for(std::size_t shell_2 = 0; shell_2 <= shell_1; ++shell_2) {
      const std::size_t pair_functions =
          shell_1 == shell_2
              ? checked_product({basis[shell_1].size(),
                                 basis[shell_1].size() + 1U},
                                "OVOV diagonal shell-pair functions") /
                    2U
              : checked_product({basis[shell_1].size(),
                                 basis[shell_2].size()},
                                "OVOV shell-pair functions");
      maximum_shell_pair_functions =
          std::max(maximum_shell_pair_functions, pair_functions);
    }
  }
  std::size_t resident_bytes = 0;
  checked_add(resident_bytes,
              checked_bytes(right_pair_elements,
                            "OVOV right pair coefficients"),
              "OVOV transform peak");
  checked_add(resident_bytes,
              checked_bytes(intermediate_elements,
                            "OVOV AO-pair intermediate"),
              "OVOV transform peak");
  checked_add(resident_bytes,
              checked_bytes(result_elements, "OVOV result tile"),
              "OVOV transform peak");
  checked_add(
      resident_bytes,
      checked_product({2U, unique_ao_pair_count, sizeof(std::size_t)},
                      "OVOV unique AO-pair indices"),
      "OVOV transform peak");
  checked_add(
      resident_bytes,
      checked_product({unique_shell_pair_count + 1U, sizeof(std::size_t)},
                      "OVOV unique shell-pair offsets"),
      "OVOV transform peak");
  checked_add(
      resident_bytes,
      checked_bytes(
          checked_product({options.threads, maximum_shell_pair_functions,
                           maximum_shell_pair_functions},
                          "OVOV thread shell-quartet blocks"),
          "OVOV thread shell-quartet blocks"),
      "OVOV transform peak");
  std::size_t workspace_elements_per_left_pair = unique_ao_pair_count;
  checked_add(workspace_elements_per_left_pair, right_pair_count,
              "OVOV streamed-left workspace");
  const std::size_t workspace_bytes_per_left_pair = checked_bytes(
      workspace_elements_per_left_pair,
      "OVOV streamed-left workspace");
  if(resident_bytes >= options.maximum_additional_memory_bytes ||
     workspace_bytes_per_left_pair >
         options.maximum_additional_memory_bytes - resident_bytes) {
    throw std::runtime_error(
        "direct OVOV tile estimates " +
        std::to_string(resident_bytes + workspace_bytes_per_left_pair) +
        " additional bytes, exceeding the explicit limit of " +
        std::to_string(options.maximum_additional_memory_bytes));
  }
  // Do not consume the caller's entire explicit allowance merely because it
  // is available. A bounded left-pair panel leaves headroom for MPI, BLAS,
  // Libint2 engine state, and the selected-integral layer above this backend.
  constexpr std::size_t maximum_streamed_left_pairs = 1024U;
  const std::size_t left_pair_block_size = std::min(
      {left_pair_count, maximum_streamed_left_pairs,
       (options.maximum_additional_memory_bytes - resident_bytes) /
           workspace_bytes_per_left_pair});
  const std::size_t estimated_peak_bytes =
      resident_bytes + left_pair_block_size * workspace_bytes_per_left_pair;

  // Compress the symmetric AO products into mu>=nu pairs. For an off-diagonal
  // pair, its orbital coefficient is C_mu,p C_nu,q + C_nu,p C_mu,q. The
  // resulting pair-space Coulomb matrix reproduces the complete ordered AO
  // sum exactly while removing redundant within-pair permutations. Shell-pair
  // blocks remain contiguous so each OpenMP worker can own disjoint rows of
  // the intermediate without atomics or a thread-private full tensor.
  struct UniqueShellPair {
    std::size_t shell_1;
    std::size_t shell_2;
    std::size_t offset;
    std::size_t count;
  };
  std::vector<UniqueShellPair> shell_pairs;
  shell_pairs.reserve(unique_shell_pair_count);
  std::vector<std::size_t> ao_pair_first(unique_ao_pair_count, 0U);
  std::vector<std::size_t> ao_pair_second(unique_ao_pair_count, 0U);
  std::size_t unique_pair = 0;
  for(std::size_t shell_1 = 0; shell_1 < shell_count; ++shell_1) {
    const std::size_t first_1 = shell_to_ao[shell_1];
    const std::size_t count_1 = basis[shell_1].size();
    for(std::size_t shell_2 = 0; shell_2 <= shell_1; ++shell_2) {
      const std::size_t first_2 = shell_to_ao[shell_2];
      const std::size_t count_2 = basis[shell_2].size();
      const std::size_t block_offset = unique_pair;
      for(std::size_t function_1 = 0; function_1 < count_1;
          ++function_1) {
        const std::size_t function_2_count =
            shell_1 == shell_2 ? function_1 + 1U : count_2;
        for(std::size_t function_2 = 0;
            function_2 < function_2_count;
            ++function_2) {
          if(unique_pair >= unique_ao_pair_count) {
            throw std::runtime_error(
                "OVOV unique AO-pair map exceeds its extent");
          }
          ao_pair_first[unique_pair] = first_1 + function_1;
          ao_pair_second[unique_pair] = first_2 + function_2;
          ++unique_pair;
        }
      }
      shell_pairs.push_back(UniqueShellPair{
          .shell_1 = shell_1,
          .shell_2 = shell_2,
          .offset = block_offset,
          .count = unique_pair - block_offset,
      });
    }
  }
  if(shell_pairs.size() != unique_shell_pair_count ||
     unique_pair != unique_ao_pair_count) {
    throw std::runtime_error(
        "OVOV unique AO-pair map has an invalid size");
  }

  linalg::Matrix right_pairs{unique_ao_pair_count, right_pair_count};
#ifdef MODERNQC_HAS_OPENMP
#pragma omp parallel for schedule(static) \
    num_threads(static_cast<int>(options.threads))
#endif
  for(std::size_t pair = 0; pair < right_pair_count; ++pair) {
    const std::size_t j = pair / options.right_virtual_count;
    const std::size_t b = pair % options.right_virtual_count;
    const std::size_t occupied = options.right_occupied_first + j;
    const std::size_t virtual_orbital = options.right_virtual_first + b;
    for(std::size_t ao_pair = 0; ao_pair < unique_ao_pair_count;
        ++ao_pair) {
      const std::size_t first = ao_pair_first[ao_pair];
      const std::size_t second = ao_pair_second[ao_pair];
      double coefficient =
          occupied_coefficients(first, occupied) *
          virtual_coefficients(second, virtual_orbital);
      if(first != second) {
        coefficient +=
            occupied_coefficients(second, occupied) *
            virtual_coefficients(first, virtual_orbital);
      }
      right_pairs(ao_pair, pair) = coefficient;
    }
  }

  linalg::Matrix intermediate{unique_ao_pair_count, right_pair_count};
  const auto blas_dimension = [](std::size_t value,
                                 const char* description) {
    if(value == 0 ||
       value >
           static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error(
          std::string{description} + " exceeds the BLAS integer range");
    }
    return static_cast<int>(value);
  };
  const int pair_leading_dimension =
      blas_dimension(unique_ao_pair_count,
                     "OVOV unique AO-pair dimension");
  const int right_pair_dimension =
      blas_dimension(right_pair_count, "OVOV right-pair dimension");
  constexpr char no_transpose = 'N';
  constexpr double alpha = 1.0;
  constexpr double beta = 1.0;
  std::vector<std::exception_ptr> thread_errors(options.threads);
  std::vector<std::uint64_t> thread_evaluated(options.threads, 0);
  std::vector<std::uint64_t> thread_screened(options.threads, 0);
  const auto transform_owned_bra_pairs =
      [&](std::size_t thread, std::size_t actual_threads,
          libint2::Engine& engine) {
        std::vector<double> integral_block;
        for(std::size_t bra_pair_index = thread;
            bra_pair_index < shell_pairs.size();
            bra_pair_index += actual_threads) {
          const UniqueShellPair& bra = shell_pairs[bra_pair_index];
          const std::size_t count_2 = basis[bra.shell_2].size();
          const std::size_t first_1 = shell_to_ao[bra.shell_1];
          const std::size_t first_2 = shell_to_ao[bra.shell_2];
          const int bra_dimension = blas_dimension(
              bra.count, "OVOV unique bra shell-pair dimension");
          for(const UniqueShellPair& ket : shell_pairs) {
            const double schwarz_bound =
                implementation_->shell_pair_schwarz_bounds
                    [bra.shell_1 * shell_count + bra.shell_2] *
                implementation_->shell_pair_schwarz_bounds
                    [ket.shell_1 * shell_count + ket.shell_2];
            if(options.schwarz_threshold > 0.0 &&
               schwarz_bound < options.schwarz_threshold) {
              ++thread_screened[thread];
              continue;
            }
            ++thread_evaluated[thread];
            engine.compute(basis[bra.shell_1], basis[bra.shell_2],
                           basis[ket.shell_1], basis[ket.shell_2]);
            const double* buffer = engine.results()[0];
            if(buffer == nullptr) {
              throw std::runtime_error(
                  "no-screen direct OVOV engine returned a null quartet");
            }
            const std::size_t count_3 = basis[ket.shell_1].size();
            const std::size_t count_4 = basis[ket.shell_2].size();
            const std::size_t first_3 = shell_to_ao[ket.shell_1];
            const std::size_t first_4 = shell_to_ao[ket.shell_2];
            const int ket_dimension = blas_dimension(
                ket.count, "OVOV unique ket shell-pair dimension");
            integral_block.resize(checked_product(
                {bra.count, ket.count},
                "OVOV unique shell-quartet block"));
            for(std::size_t local_ket = 0; local_ket < ket.count;
                ++local_ket) {
              const std::size_t function_3 =
                  ao_pair_first[ket.offset + local_ket] - first_3;
              const std::size_t function_4 =
                  ao_pair_second[ket.offset + local_ket] - first_4;
              for(std::size_t local_bra = 0; local_bra < bra.count;
                  ++local_bra) {
                const std::size_t function_1 =
                    ao_pair_first[bra.offset + local_bra] - first_1;
                const std::size_t function_2 =
                    ao_pair_second[bra.offset + local_bra] - first_2;
                const std::size_t buffer_index =
                    (((function_1 * count_2) + function_2) * count_3 +
                     function_3) *
                        count_4 +
                    function_4;
                const double value = buffer[buffer_index];
                if(!std::isfinite(value)) {
                  throw std::runtime_error(
                      "direct OVOV shell quartet contains a nonfinite value");
                }
                integral_block[local_ket * bra.count + local_bra] = value;
              }
            }
            dgemm_(&no_transpose, &no_transpose, &bra_dimension,
                   &right_pair_dimension, &ket_dimension, &alpha,
                   integral_block.data(), &bra_dimension,
                   right_pairs.data() + ket.offset,
                   &pair_leading_dimension, &beta,
                   intermediate.data() + bra.offset,
                   &pair_leading_dimension);
          }
        }
      };
#ifdef MODERNQC_HAS_OPENMP
#pragma omp parallel num_threads(static_cast<int>(options.threads))
  {
    const std::size_t thread =
        static_cast<std::size_t>(omp_get_thread_num());
    const std::size_t actual_threads =
        static_cast<std::size_t>(omp_get_num_threads());
    try {
      libint2::Engine engine{
          libint2::Operator::coulomb, basis.max_nprim(),
          static_cast<int>(basis.max_l()), 0, 0.0};
      transform_owned_bra_pairs(thread, actual_threads, engine);
    } catch(...) {
      thread_errors[thread] = std::current_exception();
    }
  }
#else
  try {
    libint2::Engine engine{
        libint2::Operator::coulomb, basis.max_nprim(),
        static_cast<int>(basis.max_l()), 0, 0.0};
    transform_owned_bra_pairs(0U, 1U, engine);
  } catch(...) {
    thread_errors[0] = std::current_exception();
  }
#endif
  for(const std::exception_ptr& error : thread_errors) {
    if(error) {
      std::rethrow_exception(error);
    }
  }
  linalg::require_finite(intermediate,
                         "direct OVOV AO-pair intermediate");
  std::vector<double> transformed_values(result_elements, 0.0);
  for(std::size_t first_pair = 0; first_pair < left_pair_count;
      first_pair += left_pair_block_size) {
    const std::size_t block_count =
        std::min(left_pair_block_size, left_pair_count - first_pair);
    linalg::Matrix left_pairs{block_count, unique_ao_pair_count};
#ifdef MODERNQC_HAS_OPENMP
#pragma omp parallel for schedule(static) \
    num_threads(static_cast<int>(options.threads))
#endif
    for(std::size_t local_pair = 0; local_pair < block_count;
        ++local_pair) {
      const std::size_t pair = first_pair + local_pair;
      const std::size_t i = pair / options.left_virtual_count;
      const std::size_t a = pair % options.left_virtual_count;
      const std::size_t occupied = options.left_occupied_first + i;
      const std::size_t virtual_orbital =
          options.left_virtual_first + a;
      for(std::size_t ao_pair = 0; ao_pair < unique_ao_pair_count;
          ++ao_pair) {
        const std::size_t first = ao_pair_first[ao_pair];
        const std::size_t second = ao_pair_second[ao_pair];
        double coefficient =
            occupied_coefficients(first, occupied) *
            virtual_coefficients(second, virtual_orbital);
        if(first != second) {
          coefficient +=
              occupied_coefficients(second, occupied) *
              virtual_coefficients(first, virtual_orbital);
        }
        left_pairs(local_pair, ao_pair) = coefficient;
      }
    }
    const linalg::Matrix transformed_block =
        linalg::multiply(left_pairs, intermediate);
    for(std::size_t right_pair = 0; right_pair < right_pair_count;
        ++right_pair) {
      std::copy_n(
          transformed_block.data() + right_pair * block_count,
          block_count,
          transformed_values.data() + right_pair * left_pair_count +
              first_pair);
    }
  }
  std::uint64_t evaluated_quartets = 0;
  std::uint64_t screened_quartets = 0;
  for(std::size_t thread = 0; thread < options.threads; ++thread) {
    evaluated_quartets += thread_evaluated[thread];
    screened_quartets += thread_screened[thread];
  }
  if(evaluated_quartets + screened_quartets !=
     static_cast<std::uint64_t>(transform_quartet_count)) {
    throw std::runtime_error(
        "direct OVOV shell-quartet screening counters are inconsistent");
  }
  return OvovTileResult{
      .left_occupied_count = options.left_occupied_count,
      .left_virtual_count = options.left_virtual_count,
      .right_occupied_count = options.right_occupied_count,
      .right_virtual_count = options.right_virtual_count,
      .values = std::move(transformed_values),
      .left_pair_ao_intermediate = {},
      .ao_count = ao_count,
      .estimated_peak_additional_bytes = estimated_peak_bytes,
      .ordered_shell_quartets =
          static_cast<std::uint64_t>(transform_quartet_count),
      .evaluated_shell_quartets = evaluated_quartets,
      .schwarz_screened_shell_quartets = screened_quartets,
  };
}

DirectFockResult Libint2IntegralProvider::build_rhf_two_electron(
    const linalg::Matrix& spin_summed_density,
    ScreeningOptions screening, ParallelBuildOptions parallel) const {
  if(!std::isfinite(screening.schwarz_threshold) ||
     screening.schwarz_threshold < 0.0 ||
     !std::isfinite(screening.fock_contribution_threshold) ||
     screening.fock_contribution_threshold < 0.0) {
    throw std::invalid_argument(
        "integral screening thresholds must be finite and nonnegative");
  }
  if(parallel.ranks == 0 || parallel.rank >= parallel.ranks ||
     parallel.threads == 0) {
    throw std::invalid_argument(
        "parallel direct-Fock topology is invalid");
  }
#ifndef MODERNQC_HAS_OPENMP
  if(parallel.threads != 1) {
    throw std::invalid_argument(
        "multiple direct-Fock threads require OpenMP support");
  }
#else
  if(parallel.threads >
     static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "direct-Fock thread count exceeds the OpenMP integer range");
  }
#endif
  const std::size_t nbf = implementation_->metadata.number_of_aos;
  if(spin_summed_density.rows() != nbf ||
     spin_summed_density.columns() != nbf) {
    throw std::invalid_argument(
        "RHF density dimensions do not match the AO basis");
  }
  linalg::require_finite(spin_summed_density, "RHF density");
  if(linalg::maximum_asymmetry(spin_summed_density) > 1.0e-12) {
    throw std::invalid_argument(
        "RHF density exceeds the symmetry tolerance");
  }

  const libint2::BasisSet& basis = implementation_->backend_basis;
  const std::vector<std::size_t>& shell_to_ao = basis.shell2bf();
  const std::size_t shell_count = basis.size();
  const std::vector<ShellQuartet> quartets =
      enumerate_unique_shell_quartets(shell_count);
  if(quartets.size() >
     static_cast<std::size_t>(
         std::numeric_limits<std::uint64_t>::max())) {
    throw std::overflow_error("unique shell-quartet count exceeds uint64");
  }
  const std::size_t owned_quartet_count =
      parallel.rank < quartets.size()
          ? 1 + (quartets.size() - 1 - parallel.rank) / parallel.ranks
          : 0;

  struct ThreadResult {
    linalg::Matrix fock;
    std::uint64_t owned{0};
    std::uint64_t evaluated{0};
    std::uint64_t schwarz_screened{0};
    std::uint64_t density_screened{0};
    double error_bound{0.0};
  };
  std::vector<ThreadResult> thread_results;
  thread_results.reserve(parallel.threads);
  for(std::size_t thread = 0; thread < parallel.threads; ++thread) {
    thread_results.push_back(
        ThreadResult{.fock = linalg::Matrix{nbf, nbf}});
  }
  std::vector<std::exception_ptr> thread_errors(parallel.threads);

  const auto process_quartet =
      [&](const ShellQuartet& quartet, libint2::Engine& engine,
          ThreadResult& thread_result) {
    const std::size_t shell_1 = quartet.shell_1;
    const std::size_t shell_2 = quartet.shell_2;
    const std::size_t shell_3 = quartet.shell_3;
    const std::size_t shell_4 = quartet.shell_4;
    const std::size_t ao_1 = shell_to_ao[shell_1];
    const std::size_t functions_1 = basis[shell_1].size();
    const std::size_t ao_2 = shell_to_ao[shell_2];
    const std::size_t functions_2 = basis[shell_2].size();
    const std::size_t ao_3 = shell_to_ao[shell_3];
    const std::size_t functions_3 = basis[shell_3].size();
    const std::size_t ao_4 = shell_to_ao[shell_4];
    const std::size_t functions_4 = basis[shell_4].size();
    const double pair_12_degeneracy =
        shell_1 == shell_2 ? 1.0 : 2.0;
    const double pair_34_degeneracy =
        shell_3 == shell_4 ? 1.0 : 2.0;
    const double pair_exchange_degeneracy =
        shell_1 == shell_3
            ? (shell_2 == shell_4 ? 1.0 : 2.0)
            : 2.0;
    const double degeneracy = pair_12_degeneracy * pair_34_degeneracy *
                              pair_exchange_degeneracy;

    const double schwarz_bound =
        implementation_->shell_pair_schwarz_bounds
            [shell_1 * shell_count + shell_2] *
        implementation_->shell_pair_schwarz_bounds
            [shell_3 * shell_count + shell_4];
    const bool schwarz_screened =
        screening.schwarz_threshold > 0.0 &&
        schwarz_bound < screening.schwarz_threshold;
    const bool needs_fock_bound =
        schwarz_screened ||
        (screening.density_aware &&
         screening.fock_contribution_threshold > 0.0);
    const double fock_bound =
        needs_fock_bound
            ? density_aware_fock_bound(
                  spin_summed_density, schwarz_bound, degeneracy, ao_1,
                  functions_1, ao_2, functions_2, ao_3, functions_3, ao_4,
                  functions_4)
            : 0.0;
    if(schwarz_screened) {
      ++thread_result.schwarz_screened;
      thread_result.error_bound += fock_bound;
      if(!std::isfinite(thread_result.error_bound)) {
        throw std::overflow_error(
            "screened Fock error-bound accumulation overflow");
      }
      return;
    }
    if(screening.density_aware &&
       screening.fock_contribution_threshold > 0.0 &&
       fock_bound < screening.fock_contribution_threshold) {
      ++thread_result.density_screened;
      thread_result.error_bound += fock_bound;
      if(!std::isfinite(thread_result.error_bound)) {
        throw std::overflow_error(
            "screened Fock error-bound accumulation overflow");
      }
      return;
    }

    engine.compute(basis[shell_1], basis[shell_2], basis[shell_3],
                   basis[shell_4]);
    const double* buffer = engine.results()[0];
    if(buffer == nullptr) {
      throw std::runtime_error(
          "no-screen Libint2 ERI engine returned a null shell quartet");
    }
    ++thread_result.evaluated;

    std::size_t integral_index = 0;
    for(std::size_t function_1 = 0; function_1 < functions_1;
        ++function_1) {
      const std::size_t basis_1 = ao_1 + function_1;
      for(std::size_t function_2 = 0; function_2 < functions_2;
          ++function_2) {
        const std::size_t basis_2 = ao_2 + function_2;
        for(std::size_t function_3 = 0; function_3 < functions_3;
            ++function_3) {
          const std::size_t basis_3 = ao_3 + function_3;
          for(std::size_t function_4 = 0; function_4 < functions_4;
              ++function_4, ++integral_index) {
            const std::size_t basis_4 = ao_4 + function_4;
            const double value = buffer[integral_index] * degeneracy;
            const double density_34 =
                0.5 * spin_summed_density(basis_3, basis_4);
            const double density_12 =
                0.5 * spin_summed_density(basis_1, basis_2);
            const double density_24 =
                0.5 * spin_summed_density(basis_2, basis_4);
            const double density_13 =
                0.5 * spin_summed_density(basis_1, basis_3);
            const double density_23 =
                0.5 * spin_summed_density(basis_2, basis_3);
            const double density_14 =
                0.5 * spin_summed_density(basis_1, basis_4);
            thread_result.fock(basis_1, basis_2) += density_34 * value;
            thread_result.fock(basis_3, basis_4) += density_12 * value;
            thread_result.fock(basis_1, basis_3) -=
                0.25 * density_24 * value;
            thread_result.fock(basis_2, basis_4) -=
                0.25 * density_13 * value;
            thread_result.fock(basis_1, basis_4) -=
                0.25 * density_23 * value;
            thread_result.fock(basis_2, basis_3) -=
                0.25 * density_14 * value;
          }
        }
      }
    }
  };

#ifdef MODERNQC_HAS_OPENMP
#pragma omp parallel num_threads(static_cast<int>(parallel.threads))
  {
    const std::size_t thread =
        static_cast<std::size_t>(omp_get_thread_num());
    const std::size_t actual_threads =
        static_cast<std::size_t>(omp_get_num_threads());
    try {
      libint2::Engine engine{libint2::Operator::coulomb, basis.max_nprim(),
                             static_cast<int>(basis.max_l()), 0, 0.0};
      for(std::size_t local_index = thread;
          local_index < owned_quartet_count;
          local_index += actual_threads) {
        const std::size_t global_index =
            parallel.rank + local_index * parallel.ranks;
        ++thread_results[thread].owned;
        process_quartet(quartets[global_index], engine,
                        thread_results[thread]);
      }
    } catch(...) {
      thread_errors[thread] = std::current_exception();
    }
  }
#else
  try {
    libint2::Engine engine{libint2::Operator::coulomb, basis.max_nprim(),
                           static_cast<int>(basis.max_l()), 0, 0.0};
    for(std::size_t local_index = 0;
        local_index < owned_quartet_count; ++local_index) {
      const std::size_t global_index =
          parallel.rank + local_index * parallel.ranks;
      ++thread_results[0].owned;
      process_quartet(quartets[global_index], engine, thread_results[0]);
    }
  } catch(...) {
    thread_errors[0] = std::current_exception();
  }
#endif

  for(const std::exception_ptr& error : thread_errors) {
    if(error) {
      std::rethrow_exception(error);
    }
  }

  linalg::Matrix two_electron{nbf, nbf};
  std::uint64_t owned_quartets = 0;
  std::uint64_t evaluated_quartets = 0;
  std::uint64_t schwarz_screened_quartets = 0;
  std::uint64_t density_screened_quartets = 0;
  double screened_fock_error_bound = 0.0;
  for(const ThreadResult& thread_result : thread_results) {
    for(std::size_t index = 0; index < two_electron.size(); ++index) {
      two_electron.values()[index] += thread_result.fock.values()[index];
    }
    owned_quartets += thread_result.owned;
    evaluated_quartets += thread_result.evaluated;
    schwarz_screened_quartets += thread_result.schwarz_screened;
    density_screened_quartets += thread_result.density_screened;
    screened_fock_error_bound += thread_result.error_bound;
  }
  if(owned_quartets != static_cast<std::uint64_t>(owned_quartet_count) ||
     evaluated_quartets + schwarz_screened_quartets +
             density_screened_quartets !=
         owned_quartets) {
    throw std::runtime_error(
        "parallel direct-Fock quartet accounting is inconsistent");
  }
  linalg::symmetrize_in_place(two_electron);
  linalg::require_finite(two_electron, "direct RHF two-electron matrix");
  return DirectFockResult{
      .two_electron = std::move(two_electron),
      .unique_shell_quartets =
          static_cast<std::uint64_t>(quartets.size()),
      .owned_shell_quartets = owned_quartets,
      .evaluated_shell_quartets = evaluated_quartets,
      .screened_shell_quartets =
          schwarz_screened_quartets + density_screened_quartets,
      .schwarz_screened_shell_quartets = schwarz_screened_quartets,
      .density_screened_shell_quartets = density_screened_quartets,
      .screened_fock_maximum_error_bound = screened_fock_error_bound,
  };
}

}  // namespace modernqc::integrals
