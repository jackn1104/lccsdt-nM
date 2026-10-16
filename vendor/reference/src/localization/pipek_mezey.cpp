#include "modernqc/localization/pipek_mezey.hpp"

#include "pipek_mezey_ciah.hpp"

#include "modernqc/linalg/eigensolver.hpp"
#include "modernqc/linalg/matrix.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace modernqc::localization {
namespace {

template <typename Operation>
[[nodiscard]] linalg::Matrix root_matrix_operation(
    std::size_t rows, std::size_t columns,
    const LocalizationParallelOptions& parallel, Operation&& operation,
    const char* description) {
  if(parallel.ranks == 1) {
    return operation();
  }
  linalg::Matrix result{rows, columns};
  std::exception_ptr root_error;
  if(parallel.rank == 0) {
    try {
      result = operation();
      if(result.rows() != rows || result.columns() != columns) {
        throw std::logic_error(std::string{description} +
                               " returned incompatible dimensions");
      }
    } catch(...) {
      root_error = std::current_exception();
    }
  }
  std::vector<double> failed{root_error ? 1.0 : 0.0};
  parallel.sum_in_place(failed);
  if(failed.front() != 0.0) {
    if(root_error) {
      std::rethrow_exception(root_error);
    }
    throw std::runtime_error(std::string{"rank zero failed during "} +
                             description);
  }
  parallel.broadcast_from_root(result.values());
  linalg::require_finite(result, description);
  return result;
}

template <typename Operation>
[[nodiscard]] double root_scalar_operation(
    const LocalizationParallelOptions& parallel, Operation&& operation,
    const char* description) {
  if(parallel.ranks == 1) {
    return operation();
  }
  double result = 0.0;
  std::exception_ptr root_error;
  if(parallel.rank == 0) {
    try {
      result = operation();
      if(!std::isfinite(result)) {
        throw std::runtime_error(std::string{description} +
                                 " returned a nonfinite value");
      }
    } catch(...) {
      root_error = std::current_exception();
    }
  }
  std::vector<double> reduced{root_error ? 1.0 : 0.0, result};
  parallel.sum_in_place(reduced);
  if(reduced.front() != 0.0) {
    if(root_error) {
      std::rethrow_exception(root_error);
    }
    throw std::runtime_error(std::string{"rank zero failed during "} +
                             description);
  }
  return reduced.back();
}

[[nodiscard]] linalg::Matrix select_columns(
    const linalg::Matrix& matrix,
    const std::vector<std::size_t>& indices) {
  if(indices.empty()) {
    throw std::invalid_argument(
        "Pipek-Mezey orbital space may not be empty");
  }
  std::set<std::size_t> unique;
  linalg::Matrix result{matrix.rows(), indices.size()};
  for(std::size_t column = 0; column < indices.size(); ++column) {
    const std::size_t source = indices[column];
    if(source >= matrix.columns()) {
      throw std::invalid_argument(
          "Pipek-Mezey orbital index is out of range");
    }
    if(!unique.insert(source).second) {
      throw std::invalid_argument(
          "Pipek-Mezey orbital indices contain a duplicate");
    }
    for(std::size_t row = 0; row < matrix.rows(); ++row) {
      result(row, column) = matrix(row, source);
    }
  }
  return result;
}

[[nodiscard]] std::filesystem::path diagnostic_path(
    const std::filesystem::path& prefix, const char* suffix) {
  return std::filesystem::path{prefix.string() + suffix};
}

void require_diagnostic_outputs(
    const PipekMezeyDiagnosticPaths& paths, bool overwrite) {
  const std::array<std::filesystem::path, 5> all{
      paths.rotation, paths.coefficients, paths.monomer_populations,
      paths.objective_history, paths.summary};
  for(const auto& path : all) {
    if(std::filesystem::exists(path) && !overwrite) {
      throw std::runtime_error(
          "Pipek-Mezey diagnostic output already exists: " +
          path.string());
    }
    const std::filesystem::path parent = path.parent_path();
    if(!parent.empty() && !std::filesystem::is_directory(parent)) {
      throw std::runtime_error(
          "Pipek-Mezey diagnostic output directory does not exist: " +
          parent.string());
    }
  }
}

void write_matrix_csv(const std::filesystem::path& path,
                      const linalg::Matrix& matrix) {
  if(matrix.empty()) {
    throw std::invalid_argument(
        "Pipek-Mezey diagnostic matrix may not be empty");
  }
  std::ofstream stream{path, std::ios::trunc};
  if(!stream) {
    throw std::runtime_error(
        "unable to open Pipek-Mezey diagnostic matrix: " + path.string());
  }
  stream << "row";
  for(std::size_t column = 0; column < matrix.columns(); ++column) {
    stream << ",column_" << column;
  }
  stream << '\n' << std::setprecision(17);
  for(std::size_t row = 0; row < matrix.rows(); ++row) {
    stream << row;
    for(std::size_t column = 0; column < matrix.columns(); ++column) {
      stream << ',' << matrix(row, column);
    }
    stream << '\n';
  }
  if(!stream) {
    throw std::runtime_error(
        "failed while writing Pipek-Mezey diagnostic matrix: " +
        path.string());
  }
}

[[nodiscard]] linalg::Matrix symmetric_square_root(
    const linalg::Matrix& overlap) {
  const linalg::SymmetricEigendecomposition eigensystem =
      linalg::diagonalize_symmetric(overlap);
  linalg::Matrix scaled_vectors = eigensystem.eigenvectors;
  constexpr double negative_tolerance = 1.0e-12;
  for(std::size_t eigenvector = 0;
      eigenvector < eigensystem.eigenvalues.size(); ++eigenvector) {
    const double eigenvalue = eigensystem.eigenvalues[eigenvector];
    if(eigenvalue < -negative_tolerance) {
      throw std::runtime_error(
          "overlap has a negative eigenvalue beyond the Löwdin tolerance");
    }
    const double factor = std::sqrt(std::max(0.0, eigenvalue));
    for(std::size_t row = 0; row < scaled_vectors.rows(); ++row) {
      scaled_vectors(row, eigenvector) *= factor;
    }
  }
  linalg::Matrix result =
      linalg::multiply(scaled_vectors,
                       linalg::transpose(eigensystem.eigenvectors));
  linalg::symmetrize_in_place(result);
  return result;
}

[[nodiscard]] linalg::Matrix inverse_symmetric_square_root(
    const linalg::Matrix& matrix, const char* description) {
  const linalg::SymmetricEigendecomposition eigensystem =
      linalg::diagonalize_symmetric(matrix);
  const double largest = eigensystem.eigenvalues.back();
  if(!std::isfinite(largest) || largest <= 0.0) {
    throw std::runtime_error(std::string{description} +
                             " is not positive definite");
  }
  linalg::Matrix scaled = eigensystem.eigenvectors;
  for(std::size_t column = 0; column < matrix.columns(); ++column) {
    const double eigenvalue = eigensystem.eigenvalues[column];
    if(eigenvalue <= 1.0e-14 * largest) {
      throw std::runtime_error(std::string{description} +
                               " is rank deficient");
    }
    const double factor = 1.0 / std::sqrt(eigenvalue);
    for(std::size_t row = 0; row < matrix.rows(); ++row) {
      scaled(row, column) *= factor;
    }
  }
  return linalg::multiply(scaled,
                          linalg::transpose(eigensystem.eigenvectors));
}

[[nodiscard]] linalg::Matrix submatrix(
    const linalg::Matrix& matrix, const std::vector<std::size_t>& rows,
    const std::vector<std::size_t>& columns) {
  linalg::Matrix result{rows.size(), columns.size()};
  for(std::size_t column = 0; column < columns.size(); ++column) {
    for(std::size_t row = 0; row < rows.size(); ++row) {
      result(row, column) = matrix(rows[row], columns[column]);
    }
  }
  return result;
}

void set_submatrix(linalg::Matrix& matrix,
                   const std::vector<std::size_t>& rows,
                   const std::vector<std::size_t>& columns,
                   const linalg::Matrix& values) {
  if(values.rows() != rows.size() || values.columns() != columns.size()) {
    throw std::invalid_argument("submatrix assignment dimensions disagree");
  }
  for(std::size_t column = 0; column < columns.size(); ++column) {
    for(std::size_t row = 0; row < rows.size(); ++row) {
      matrix(rows[row], columns[column]) = values(row, column);
    }
  }
}

[[nodiscard]] std::vector<std::size_t> atom_ao_indices(
    const basis::BasisSet& basis, std::size_t atom) {
  std::vector<std::size_t> result;
  for(const basis::Shell& shell : basis.shells) {
    if(shell.atom_index != atom) {
      continue;
    }
    for(std::size_t offset = 0; offset < shell.function_count; ++offset) {
      result.push_back(shell.first_ao + offset);
    }
  }
  return result;
}

[[nodiscard]] std::vector<std::size_t> angular_positions_on_atom(
    const basis::BasisSet& basis, std::size_t atom, int angular_momentum,
    const std::vector<std::size_t>& atom_indices) {
  std::vector<std::size_t> result;
  for(const basis::Shell& shell : basis.shells) {
    if(shell.atom_index != atom ||
       shell.angular_momentum != angular_momentum) {
      continue;
    }
    for(std::size_t offset = 0; offset < shell.function_count; ++offset) {
      const std::size_t global = shell.first_ao + offset;
      const auto found = std::lower_bound(atom_indices.begin(),
                                          atom_indices.end(), global);
      if(found == atom_indices.end() || *found != global) {
        throw std::logic_error("atom AO position lookup failed");
      }
      result.push_back(static_cast<std::size_t>(
          std::distance(atom_indices.begin(), found)));
    }
  }
  return result;
}

[[nodiscard]] linalg::Matrix metric_orthonormalize(
    const linalg::Matrix& coefficients, const linalg::Matrix& overlap,
    const char* description) {
  if(coefficients.columns() == 0) {
    return coefficients;
  }
  const linalg::Matrix gram = linalg::multiply(
      linalg::transpose(coefficients),
      linalg::multiply(overlap, coefficients));
  return linalg::multiply(
      coefficients, inverse_symmetric_square_root(gram, description));
}

[[nodiscard]] linalg::Matrix project_out_metric_space(
    const linalg::Matrix& coefficients, const linalg::Matrix& space,
    const linalg::Matrix& overlap) {
  if(space.columns() == 0 || coefficients.columns() == 0) {
    return coefficients;
  }
  return linalg::subtract(
      coefficients,
      linalg::multiply(
          space,
          linalg::multiply(linalg::transpose(space),
                           linalg::multiply(overlap, coefficients))));
}

[[nodiscard]] std::array<int, 4> core_shell_counts(int atomic_number) {
  if(atomic_number == 1) {
    return {0, 0, 0, 0};
  }
  if(atomic_number == 8) {
    return {1, 0, 0, 0};
  }
  throw std::invalid_argument(
      "meta-Lowdin shell partition currently supports only H and O");
}

[[nodiscard]] std::array<int, 4> core_valence_shell_counts(
    int atomic_number) {
  if(atomic_number == 1) {
    return {1, 0, 0, 0};
  }
  if(atomic_number == 8) {
    return {2, 1, 0, 0};
  }
  throw std::invalid_argument(
      "meta-Lowdin shell partition currently supports only H and O");
}

[[nodiscard]] linalg::Matrix atomic_initial_rotation(
    const linalg::Matrix& canonical_space, const linalg::Matrix& overlap,
    const linalg::Matrix& orthogonal_ao) {
  const linalg::Matrix projections = linalg::multiply(
      linalg::transpose(orthogonal_ao),
      linalg::multiply(overlap, canonical_space));
  std::vector<std::size_t> rows(projections.rows());
  if(projections.rows() < projections.columns()) {
    throw std::invalid_argument(
        "atomic PM initial guess has fewer AO projections than orbitals");
  }
  std::iota(rows.begin(), rows.end(), std::size_t{0});
  std::stable_sort(rows.begin(), rows.end(), [&](std::size_t left,
                                                 std::size_t right) {
    long double left_norm = 0.0L;
    long double right_norm = 0.0L;
    for(std::size_t column = 0; column < projections.columns(); ++column) {
      left_norm += static_cast<long double>(projections(left, column)) *
                   projections(left, column);
      right_norm += static_cast<long double>(projections(right, column)) *
                    projections(right, column);
    }
    return left_norm < right_norm;
  });
  const std::size_t n = canonical_space.columns();
  rows.erase(rows.begin(), rows.end() - static_cast<std::ptrdiff_t>(n));
  std::sort(rows.begin(), rows.end());
  std::vector<std::size_t> columns(n);
  std::iota(columns.begin(), columns.end(), std::size_t{0});
  const linalg::Matrix selected = submatrix(projections, rows, columns);
  const linalg::Matrix gram = linalg::multiply(
      linalg::transpose(selected), selected);
  return linalg::multiply(
      inverse_symmetric_square_root(gram,
                                    "atomic PM initial-guess Gram matrix"),
      linalg::transpose(selected));
}

[[nodiscard]] std::vector<linalg::Matrix> population_matrices(
    const linalg::Matrix& lowdin_coefficients,
    const basis::BasisSet& basis, std::size_t atom_count,
    const LocalizationParallelOptions& parallel = {}) {
  if(parallel.ranks == 0 || parallel.rank >= parallel.ranks ||
     parallel.ranks > atom_count) {
    throw std::invalid_argument(
        "Pipek-Mezey population rank/count are outside their domains");
  }
  const std::size_t orbital_count = lowdin_coefficients.columns();
  std::vector<linalg::Matrix> result;
  const std::size_t local_atom_count =
      (atom_count + parallel.ranks - 1U - parallel.rank) /
      parallel.ranks;
  result.reserve(local_atom_count);
  for(std::size_t atom = parallel.rank; atom < atom_count;
      atom += parallel.ranks) {
    result.emplace_back(orbital_count, orbital_count);
  }
  for(std::size_t ao = 0; ao < lowdin_coefficients.rows(); ++ao) {
    const std::size_t shell_index = basis.ao_to_shell.at(ao);
    const std::size_t atom = basis.shells.at(shell_index).atom_index;
    if(atom >= atom_count) {
      throw std::runtime_error(
          "basis AO maps outside the molecule during Löwdin populations");
    }
    if(atom % parallel.ranks != parallel.rank) {
      continue;
    }
    const std::size_t local_atom = atom / parallel.ranks;
    for(std::size_t right = 0; right < orbital_count; ++right) {
      for(std::size_t left = 0; left < orbital_count; ++left) {
        result[local_atom](left, right) +=
            lowdin_coefficients(ao, left) *
            lowdin_coefficients(ao, right);
      }
    }
  }
  return result;
}

[[nodiscard]] double objective(
    const std::vector<linalg::Matrix>& populations,
    const LocalizationParallelOptions& parallel = {}) {
  double result = 0.0;
  for(const linalg::Matrix& atom : populations) {
    for(std::size_t orbital = 0; orbital < atom.rows(); ++orbital) {
      result += atom(orbital, orbital) * atom(orbital, orbital);
    }
  }
  std::vector<double> reduced{result};
  if(parallel.ranks > 1) {
    parallel.sum_in_place(reduced);
  }
  result = reduced.front();
  if(!std::isfinite(result)) {
    throw std::runtime_error(
        "Pipek-Mezey objective became nonfinite");
  }
  return result;
}

void rotate_pair(linalg::Matrix& matrix, std::size_t first,
                 std::size_t second, double cosine, double sine) {
  for(std::size_t row = 0; row < matrix.rows(); ++row) {
    const double old_first = matrix(row, first);
    const double old_second = matrix(row, second);
    matrix(row, first) = cosine * old_first + sine * old_second;
    matrix(row, second) = -sine * old_first + cosine * old_second;
  }
}

void rotate_population_pair(linalg::Matrix& population, std::size_t first,
                            std::size_t second, double cosine,
                            double sine) {
  const double first_diagonal = population(first, first);
  const double second_diagonal = population(second, second);
  const double off_diagonal = population(first, second);
  for(std::size_t orbital = 0; orbital < population.rows(); ++orbital) {
    if(orbital == first || orbital == second) {
      continue;
    }
    const double old_first = population(orbital, first);
    const double old_second = population(orbital, second);
    const double new_first = cosine * old_first + sine * old_second;
    const double new_second = -sine * old_first + cosine * old_second;
    population(orbital, first) = new_first;
    population(first, orbital) = new_first;
    population(orbital, second) = new_second;
    population(second, orbital) = new_second;
  }
  const double cosine_squared = cosine * cosine;
  const double sine_squared = sine * sine;
  const double twice_sine_cosine = 2.0 * sine * cosine;
  population(first, first) =
      cosine_squared * first_diagonal +
      sine_squared * second_diagonal +
      twice_sine_cosine * off_diagonal;
  population(second, second) =
      sine_squared * first_diagonal +
      cosine_squared * second_diagonal -
      twice_sine_cosine * off_diagonal;
  population(first, second) =
      (cosine_squared - sine_squared) * off_diagonal +
      sine * cosine * (second_diagonal - first_diagonal);
  population(second, first) = population(first, second);
}

[[nodiscard]] linalg::Matrix diagonal_populations(
    const std::vector<linalg::Matrix>& populations) {
  const std::size_t orbital_count = populations.front().rows();
  linalg::Matrix result{orbital_count, populations.size()};
  for(std::size_t atom = 0; atom < populations.size(); ++atom) {
    for(std::size_t orbital = 0; orbital < orbital_count; ++orbital) {
      result(orbital, atom) = populations[atom](orbital, orbital);
    }
  }
  return result;
}

[[nodiscard]] linalg::Matrix distributed_diagonal_populations(
    const std::vector<linalg::Matrix>& local_populations,
    std::size_t atom_count,
    const LocalizationParallelOptions& parallel) {
  if(local_populations.empty() || parallel.ranks == 0 ||
     parallel.rank >= parallel.ranks) {
    throw std::invalid_argument(
        "distributed population dimensions are invalid");
  }
  const std::size_t orbital_count = local_populations.front().rows();
  linalg::Matrix result{orbital_count, atom_count};
  for(std::size_t local_atom = 0;
      local_atom < local_populations.size(); ++local_atom) {
    const std::size_t atom = parallel.rank + local_atom * parallel.ranks;
    if(atom >= atom_count) {
      throw std::logic_error(
          "distributed Pipek-Mezey atom ownership is inconsistent");
    }
    const linalg::Matrix& population = local_populations[local_atom];
    if(population.rows() != orbital_count ||
       population.columns() != orbital_count) {
      throw std::invalid_argument(
          "distributed Pipek-Mezey population matrices disagree");
    }
    for(std::size_t orbital = 0; orbital < orbital_count; ++orbital) {
      result(orbital, atom) = population(orbital, orbital);
    }
  }
  if(parallel.ranks > 1) {
    parallel.sum_in_place(result.values());
  }
  linalg::require_finite(result,
                         "distributed final Lowdin populations");
  return result;
}

struct BestTwo {
  std::size_t best_index{0};
  double best{-std::numeric_limits<double>::infinity()};
  double second{-std::numeric_limits<double>::infinity()};
};

[[nodiscard]] BestTwo best_two(const std::vector<double>& values) {
  if(values.empty()) {
    throw std::invalid_argument("assignment population list is empty");
  }
  BestTwo result;
  for(std::size_t index = 0; index < values.size(); ++index) {
    const double value = values[index];
    if(value > result.best) {
      result.second = result.best;
      result.best = value;
      result.best_index = index;
    } else if(value > result.second) {
      result.second = value;
    }
  }
  if(values.size() == 1) {
    result.second = 0.0;
  }
  return result;
}

[[nodiscard]] std::vector<OrbitalAssignment> assignments(
    const linalg::Matrix& populations,
    const molecule::Molecule& molecule, double ambiguity_margin) {
  std::map<int, std::size_t> monomer_to_column;
  for(const int monomer : molecule.monomer_ids()) {
    if(monomer >= 0 && !monomer_to_column.contains(monomer)) {
      monomer_to_column.emplace(monomer, monomer_to_column.size());
    }
  }
  std::vector<int> monomer_labels(monomer_to_column.size(), -1);
  for(const auto& [label, column] : monomer_to_column) {
    monomer_labels[column] = label;
  }

  std::vector<OrbitalAssignment> result;
  result.reserve(populations.rows());
  for(std::size_t orbital = 0; orbital < populations.rows(); ++orbital) {
    std::vector<double> atom_values(populations.columns());
    for(std::size_t atom = 0; atom < populations.columns(); ++atom) {
      atom_values[atom] = populations(orbital, atom);
    }
    const BestTwo atom_best = best_two(atom_values);

    int monomer_id = -1;
    double monomer_population = 0.0;
    double second_monomer_population = 0.0;
    double monomer_margin = 0.0;
    bool monomer_ambiguous = false;
    if(!monomer_to_column.empty()) {
      std::vector<double> monomer_values(monomer_to_column.size(), 0.0);
      for(std::size_t atom = 0; atom < atom_values.size(); ++atom) {
        const int label = molecule.monomer_ids()[atom];
        if(label >= 0) {
          monomer_values[monomer_to_column.at(label)] += atom_values[atom];
        }
      }
      const BestTwo monomer_best = best_two(monomer_values);
      monomer_id = monomer_labels[monomer_best.best_index];
      monomer_population = monomer_best.best;
      second_monomer_population = monomer_best.second;
      monomer_margin = monomer_best.best - monomer_best.second;
      monomer_ambiguous = monomer_margin < ambiguity_margin;
    }

    const double atom_margin = atom_best.best - atom_best.second;
    result.push_back(OrbitalAssignment{
        .atom_index = atom_best.best_index,
        .atom_population = atom_best.best,
        .second_atom_population = atom_best.second,
        .atom_population_margin = atom_margin,
        .atom_ambiguous = atom_margin < ambiguity_margin,
        .monomer_id = monomer_id,
        .monomer_population = monomer_population,
        .second_monomer_population = second_monomer_population,
        .monomer_population_margin = monomer_margin,
        .monomer_ambiguous = monomer_ambiguous,
    });
  }
  return result;
}

[[nodiscard]] double orthogonality_error(const linalg::Matrix& matrix) {
  const linalg::Matrix metric =
      linalg::multiply(linalg::transpose(matrix), matrix);
  return linalg::maximum_absolute_value(
      linalg::subtract(metric, linalg::Matrix::identity(metric.rows())));
}

[[nodiscard]] double metric_orthonormality_error(
    const linalg::Matrix& coefficients, const linalg::Matrix& overlap) {
  const linalg::Matrix metric =
      linalg::multiply(linalg::transpose(coefficients),
                       linalg::multiply(overlap, coefficients));
  return linalg::maximum_absolute_value(
      linalg::subtract(metric, linalg::Matrix::identity(metric.rows())));
}

[[nodiscard]] double projector_error(
    const linalg::Matrix& before, const linalg::Matrix& after) {
  const linalg::Matrix before_projector =
      linalg::multiply(before, linalg::transpose(before));
  const linalg::Matrix after_projector =
      linalg::multiply(after, linalg::transpose(after));
  return linalg::maximum_absolute_value(
      linalg::subtract(before_projector, after_projector));
}

}  // namespace

linalg::Matrix build_meta_lowdin_orthogonal_ao(
    const molecule::Molecule& molecule,
    const basis::BasisSet& target_basis,
    const linalg::Matrix& target_overlap,
    const basis::BasisSet& ano_reference_basis,
    const linalg::Matrix& target_ano_overlap) {
  const std::size_t nao = target_basis.number_of_aos;
  if(target_overlap.rows() != nao || target_overlap.columns() != nao ||
     target_ano_overlap.rows() != nao ||
     target_ano_overlap.columns() !=
         ano_reference_basis.number_of_aos) {
    throw std::invalid_argument(
        "meta-Lowdin basis or overlap dimensions are incompatible");
  }
  if(!target_basis.spherical || !ano_reference_basis.spherical) {
    throw std::invalid_argument(
        "meta-Lowdin compatibility requires spherical AO bases");
  }
  linalg::require_finite(target_overlap, "meta-Lowdin target overlap");
  linalg::require_finite(target_ano_overlap,
                         "meta-Lowdin target-ANO overlap");

  linalg::Matrix preorthogonal_ao{nao, nao};
  for(std::size_t atom = 0; atom < molecule.atoms().size(); ++atom) {
    const std::vector<std::size_t> target_atom =
        atom_ao_indices(target_basis, atom);
    const std::vector<std::size_t> ano_atom =
        atom_ao_indices(ano_reference_basis, atom);
    if(target_atom.empty() || ano_atom.empty()) {
      throw std::runtime_error(
          "meta-Lowdin atom is missing target or ANO functions");
    }
    const linalg::Matrix atom_overlap =
        submatrix(target_overlap, target_atom, target_atom);
    const linalg::Matrix atom_cross =
        submatrix(target_ano_overlap, target_atom, ano_atom);
    const linalg::Matrix projected_ano =
        linalg::solve_symmetric_positive_definite(atom_overlap, atom_cross);
    const linalg::Matrix removed_ano = linalg::subtract(
        linalg::Matrix::identity(target_atom.size()),
        linalg::multiply(
            linalg::multiply(projected_ano,
                             linalg::transpose(projected_ano)),
            atom_overlap));
    linalg::Matrix restored = removed_ano;

    const int maximum_l = std::max(target_basis.maximum_angular_momentum,
                                   ano_reference_basis.maximum_angular_momentum);
    for(int angular = 0; angular <= maximum_l; ++angular) {
      const std::vector<std::size_t> target_positions =
          angular_positions_on_atom(target_basis, atom, angular,
                                    target_atom);
      const std::vector<std::size_t> ano_positions =
          angular_positions_on_atom(ano_reference_basis, atom, angular,
                                    ano_atom);
      if(target_positions.empty() || ano_positions.empty()) {
        continue;
      }
      if(ano_positions.size() >= target_positions.size()) {
        for(std::size_t column = 0; column < target_positions.size();
            ++column) {
          for(std::size_t row = 0; row < target_atom.size(); ++row) {
            restored(row, target_positions[column]) =
                projected_ano(row, ano_positions[column]);
          }
        }
        continue;
      }

      const std::size_t degeneracy =
          static_cast<std::size_t>(2 * angular + 1);
      if(target_positions.size() % degeneracy != 0 ||
         ano_positions.size() % degeneracy != 0 ||
         (target_positions.size() - ano_positions.size()) % degeneracy != 0) {
        throw std::runtime_error(
            "meta-Lowdin angular blocks do not contain complete shells");
      }
      for(std::size_t column = 0; column < ano_positions.size(); ++column) {
        for(std::size_t row = 0; row < target_atom.size(); ++row) {
          restored(row, target_positions[column]) =
              projected_ano(row, ano_positions[column]);
        }
      }

      struct ShellScore {
        std::size_t shell;
        double score;
      };
      const std::size_t target_shells =
          target_positions.size() / degeneracy;
      std::vector<ShellScore> scores;
      scores.reserve(target_shells);
      for(std::size_t shell = 0; shell < target_shells; ++shell) {
        double score = 0.0;
        for(std::size_t component = 0; component < degeneracy; ++component) {
          const std::size_t column =
              target_positions[shell * degeneracy + component];
          long double norm = 0.0L;
          for(std::size_t right = 0; right < target_atom.size(); ++right) {
            long double row_value = 0.0L;
            for(std::size_t left = 0; left < target_atom.size(); ++left) {
              row_value += static_cast<long double>(
                               atom_overlap(right, left)) *
                           removed_ano(left, column);
            }
            norm += static_cast<long double>(removed_ano(right, column)) *
                    row_value;
          }
          score += static_cast<double>(norm);
        }
        scores.push_back(ShellScore{shell, score});
      }
      std::stable_sort(scores.begin(), scores.end(),
                       [](const ShellScore& left, const ShellScore& right) {
        return left.score > right.score;
      });
      const std::size_t remaining_shells =
          (target_positions.size() - ano_positions.size()) / degeneracy;
      std::size_t destination = ano_positions.size();
      for(std::size_t selected = 0; selected < remaining_shells; ++selected) {
        const std::size_t shell = scores[selected].shell;
        for(std::size_t component = 0; component < degeneracy; ++component) {
          const std::size_t source =
              target_positions[shell * degeneracy + component];
          for(std::size_t row = 0; row < target_atom.size(); ++row) {
            restored(row, target_positions[destination]) =
                removed_ano(row, source);
          }
          ++destination;
        }
      }
    }

    for(std::size_t column = 0; column < restored.columns(); ++column) {
      long double norm = 0.0L;
      for(std::size_t right = 0; right < restored.rows(); ++right) {
        long double row_value = 0.0L;
        for(std::size_t left = 0; left < restored.rows(); ++left) {
          row_value += static_cast<long double>(atom_overlap(right, left)) *
                       restored(left, column);
        }
        norm += static_cast<long double>(restored(right, column)) *
                row_value;
      }
      if(!std::isfinite(static_cast<double>(norm)) || norm <= 0.0L) {
        throw std::runtime_error(
            "projected ANO column has a nonpositive norm");
      }
      const double scale = 1.0 / std::sqrt(static_cast<double>(norm));
      for(std::size_t row = 0; row < restored.rows(); ++row) {
        restored(row, column) *= scale;
      }
    }
    set_submatrix(preorthogonal_ao, target_atom, target_atom, restored);
  }

  std::vector<std::size_t> core;
  std::vector<std::size_t> valence;
  std::vector<std::size_t> rydberg;
  std::vector<std::array<int, 4>> shell_counts(molecule.atoms().size());
  for(const basis::Shell& shell : target_basis.shells) {
    const int angular = shell.angular_momentum;
    const int atomic_number =
        molecule.atoms().at(shell.atom_index).atomic_number;
    const auto core_limit = core_shell_counts(atomic_number);
    const auto valence_limit = core_valence_shell_counts(atomic_number);
    const bool high_angular = angular < 0 || angular > 3;
    const int ordinal = high_angular
                            ? 0
                            : shell_counts[shell.atom_index]
                                              [static_cast<std::size_t>(angular)]++;
    std::vector<std::size_t>* destination = &rydberg;
    if(!high_angular &&
       ordinal < core_limit[static_cast<std::size_t>(angular)]) {
      destination = &core;
    } else if(!high_angular &&
              ordinal < valence_limit[static_cast<std::size_t>(angular)]) {
      destination = &valence;
    }
    for(std::size_t offset = 0; offset < shell.function_count; ++offset) {
      destination->push_back(shell.first_ao + offset);
    }
  }
  if(core.size() + valence.size() + rydberg.size() != nao ||
     valence.empty()) {
    throw std::runtime_error(
        "meta-Lowdin core/valence/Rydberg partition is incomplete");
  }

  linalg::Matrix orthogonal_ao{nao, nao};
  const std::vector<std::size_t> all_rows = [&]() {
    std::vector<std::size_t> rows(nao);
    std::iota(rows.begin(), rows.end(), std::size_t{0});
    return rows;
  }();
  linalg::Matrix core_space;
  if(!core.empty()) {
    core_space = metric_orthonormalize(
        submatrix(preorthogonal_ao, all_rows, core), target_overlap,
        "meta-Lowdin core overlap");
    set_submatrix(orthogonal_ao, all_rows, core, core_space);
  }
  linalg::Matrix valence_space = project_out_metric_space(
      submatrix(preorthogonal_ao, all_rows, valence), core_space,
      target_overlap);
  valence_space = metric_orthonormalize(
      valence_space, target_overlap, "meta-Lowdin valence overlap");
  set_submatrix(orthogonal_ao, all_rows, valence, valence_space);

  std::vector<std::size_t> core_valence = core;
  core_valence.insert(core_valence.end(), valence.begin(), valence.end());
  const linalg::Matrix core_valence_space =
      submatrix(orthogonal_ao, all_rows, core_valence);
  linalg::Matrix rydberg_space = project_out_metric_space(
      submatrix(preorthogonal_ao, all_rows, rydberg),
      core_valence_space, target_overlap);
  rydberg_space = metric_orthonormalize(
      rydberg_space, target_overlap, "meta-Lowdin Rydberg overlap");
  set_submatrix(orthogonal_ao, all_rows, rydberg, rydberg_space);

  for(std::size_t column = 0; column < nao; ++column) {
    if(orthogonal_ao(column, column) < 0.0) {
      for(std::size_t row = 0; row < nao; ++row) {
        orthogonal_ao(row, column) *= -1.0;
      }
    }
  }
  if(metric_orthonormality_error(orthogonal_ao, target_overlap) > 1.0e-9) {
    throw std::runtime_error(
        "meta-Lowdin AO basis violates metric orthonormality");
  }
  return orthogonal_ao;
}

PipekMezeyResult localize_pipek_mezey(
    const molecule::Molecule& molecule, const basis::BasisSet& basis,
    const linalg::Matrix& overlap,
    const linalg::Matrix& canonical_coefficients,
    const std::vector<std::size_t>& orbital_indices,
    const PipekMezeyOptions& options) {
  if(options.maximum_sweeps <= 0 ||
     !std::isfinite(options.objective_tolerance) ||
     options.objective_tolerance <= 0.0 ||
     !std::isfinite(options.ambiguity_population_margin) ||
     options.ambiguity_population_margin < 0.0 ||
     options.ambiguity_population_margin > 1.0) {
    throw std::invalid_argument(
        "Pipek-Mezey options are outside their physical domains");
  }
  if(options.restart_state && !options.initial_rotation.empty()) {
    throw std::invalid_argument(
        "Pipek-Mezey initial_rotation and restart_state are mutually exclusive");
  }
  if(options.optimizer == PipekMezeyOptimizer::pyscf_ciah &&
     options.restart_state) {
    throw std::invalid_argument(
        "PySCF CIAH PM cannot read the Jacobi-only restart state");
  }
  if(options.atomic_initial_guess &&
     (options.restart_state || !options.initial_rotation.empty())) {
    throw std::invalid_argument(
        "Pipek-Mezey atomic initial guess cannot be combined with an explicit or restart rotation");
  }
  if(options.parallel.ranks == 0 ||
     options.parallel.rank >= options.parallel.ranks ||
     options.parallel.ranks > molecule.atoms().size() ||
     (options.parallel.ranks > 1 &&
      (!options.parallel.sum_in_place ||
       !options.parallel.broadcast_from_root))) {
    throw std::invalid_argument(
        "Pipek-Mezey distributed rank/count or reduction callback is invalid");
  }
  if(options.parallel.ranks > 1 &&
     options.optimizer != PipekMezeyOptimizer::pyscf_ciah) {
    throw std::invalid_argument(
        "distributed Pipek-Mezey currently requires the CIAH optimizer");
  }
  if(overlap.rows() != basis.number_of_aos ||
     overlap.columns() != basis.number_of_aos ||
     canonical_coefficients.rows() != basis.number_of_aos) {
    throw std::invalid_argument(
        "Pipek-Mezey AO dimensions are incompatible");
  }
  linalg::require_finite(overlap, "Pipek-Mezey overlap");
  linalg::require_finite(canonical_coefficients,
                         "Pipek-Mezey canonical coefficients");
  if(linalg::maximum_asymmetry(overlap) > 1.0e-12) {
    throw std::invalid_argument(
        "Pipek-Mezey overlap exceeds the symmetry tolerance");
  }

  const linalg::Matrix canonical_space =
      select_columns(canonical_coefficients, orbital_indices);
  const linalg::Matrix lowdin_coefficients = root_matrix_operation(
      basis.number_of_aos, orbital_indices.size(), options.parallel,
      [&]() {
        return linalg::multiply(symmetric_square_root(overlap),
                                canonical_space);
      },
      "symmetric-Lowdin orbital construction");
  linalg::Matrix objective_coefficients = lowdin_coefficients;
  if(options.population_method ==
     PipekMezeyPopulationMethod::meta_lowdin) {
    if(options.meta_lowdin_orthogonal_ao.rows() != basis.number_of_aos ||
       options.meta_lowdin_orthogonal_ao.columns() !=
           basis.number_of_aos) {
      throw std::invalid_argument(
          "meta-Lowdin PM requires a square target-AO orthogonalization");
    }
    linalg::require_finite(options.meta_lowdin_orthogonal_ao,
                           "meta-Lowdin orthogonal AO coefficients");
    const double meta_lowdin_error = root_scalar_operation(
        options.parallel,
        [&]() {
          return metric_orthonormality_error(
              options.meta_lowdin_orthogonal_ao, overlap);
        },
        "meta-Lowdin metric-orthonormality validation");
    if(meta_lowdin_error > 1.0e-9) {
      throw std::invalid_argument(
          "meta-Lowdin AO coefficients are not orthonormal in the AO metric");
    }
    objective_coefficients = root_matrix_operation(
        basis.number_of_aos, orbital_indices.size(), options.parallel,
        [&]() {
          return linalg::multiply(
              linalg::transpose(options.meta_lowdin_orthogonal_ao),
              linalg::multiply(overlap, canonical_space));
        },
        "meta-Lowdin objective-orbital construction");
  } else if(!options.meta_lowdin_orthogonal_ao.empty()) {
    throw std::invalid_argument(
        "ordinary Lowdin PM rejects meta-Lowdin AO coefficients");
  }

  linalg::Matrix rotation =
      linalg::Matrix::identity(orbital_indices.size());
  int completed_sweeps = 0;
  std::vector<double> objective_history;
  const linalg::Matrix* requested_rotation = nullptr;
  if(options.restart_state) {
    const PipekMezeyRestartState& state = *options.restart_state;
    if(state.completed_sweeps < 0 ||
       state.completed_sweeps > options.maximum_sweeps ||
       state.objective_history.size() !=
           static_cast<std::size_t>(state.completed_sweeps) + 1 ||
       std::any_of(state.objective_history.begin(),
                   state.objective_history.end(),
                   [](double value) { return !std::isfinite(value); })) {
      throw std::invalid_argument(
          "Pipek-Mezey restart counters or objective history are invalid");
    }
    requested_rotation = &state.rotation;
    completed_sweeps = state.completed_sweeps;
    objective_history = state.objective_history;
  } else if(!options.initial_rotation.empty()) {
    requested_rotation = &options.initial_rotation;
  } else if(options.atomic_initial_guess) {
    rotation = root_matrix_operation(
        orbital_indices.size(), orbital_indices.size(), options.parallel,
        [&]() {
          return atomic_initial_rotation(
              canonical_space, overlap,
              options.population_method ==
                      PipekMezeyPopulationMethod::meta_lowdin
                  ? options.meta_lowdin_orthogonal_ao
                  : linalg::multiply(
                        inverse_symmetric_square_root(
                            overlap, "Lowdin AO overlap"),
                        linalg::Matrix::identity(overlap.rows())));
        },
        "Pipek-Mezey atomic initial rotation");
  }
  if(requested_rotation != nullptr) {
    if(requested_rotation->rows() != orbital_indices.size() ||
       requested_rotation->columns() != orbital_indices.size()) {
      throw std::invalid_argument(
          "Pipek-Mezey initial rotation has incompatible dimensions");
    }
    linalg::require_finite(*requested_rotation,
                           "Pipek-Mezey initial rotation");
    const double requested_rotation_error = root_scalar_operation(
        options.parallel,
        [&]() { return orthogonality_error(*requested_rotation); },
        "Pipek-Mezey initial-rotation validation");
    if(requested_rotation_error > 2.0e-10) {
      throw std::invalid_argument(
          "Pipek-Mezey initial rotation is not orthogonal");
    }
    rotation = *requested_rotation;
  }
  const std::vector<linalg::Matrix> canonical_populations =
      population_matrices(objective_coefficients, basis,
                          molecule.atoms().size(), options.parallel);
  const auto rotated_objective_coefficients = [&]() {
    return root_matrix_operation(
        basis.number_of_aos, orbital_indices.size(), options.parallel,
        [&]() {
          return linalg::multiply(objective_coefficients, rotation);
        },
        "initial Pipek-Mezey objective-orbital rotation");
  };
  std::vector<linalg::Matrix> populations = population_matrices(
      rotated_objective_coefficients(), basis, molecule.atoms().size(),
      options.parallel);

  const double starting_objective = objective(populations, options.parallel);
  if(options.restart_state) {
    const double saved_objective = objective_history.back();
    const double tolerance =
        1.0e-9 * std::max(1.0, std::abs(saved_objective));
    if(std::abs(starting_objective - saved_objective) > tolerance) {
      throw std::runtime_error(
          "Pipek-Mezey restart objective does not match its saved rotation");
    }
    objective_history.back() = starting_objective;
  } else {
    objective_history = {starting_objective};
  }
  double final_sweep_gain =
      objective_history.size() > 1
          ? objective_history.back() -
                objective_history[objective_history.size() - 2]
          : 0.0;
  bool converged = orbital_indices.size() == 1 ||
                   (options.restart_state &&
                    std::abs(final_sweep_gain) <=
                        options.objective_tolerance);
  double final_gradient_norm = 0.0;
  int total_keyframes = 0;
  int total_hessian_actions = 0;
  const auto report = [&]() {
    if(options.progress) {
      options.progress(PipekMezeySweep{
                           .completed_sweeps = completed_sweeps,
                           .converged = converged,
                           .objective = objective_history.back(),
                           .sweep_gain = final_sweep_gain,
                           .gradient_norm = final_gradient_norm,
                           .keyframes = total_keyframes,
                           .hessian_actions = total_hessian_actions,
                       },
                       rotation, objective_history);
    }
  };
  if(options.optimizer == PipekMezeyOptimizer::pyscf_ciah) {
    const detail::PipekMezeyCiahResult ciah =
        detail::optimize_pipek_mezey_ciah(canonical_populations, rotation,
                                          options);
    rotation = ciah.rotation;
    objective_history = ciah.objective_history;
    completed_sweeps = ciah.completed_macro_iterations;
    final_sweep_gain = ciah.final_objective_gain;
    final_gradient_norm = ciah.final_gradient_norm;
    total_keyframes = ciah.total_keyframes;
    total_hessian_actions = ciah.total_hessian_actions;
    populations = population_matrices(
        rotated_objective_coefficients(), basis, molecule.atoms().size(),
        options.parallel);
    converged = true;
  } else {
    report();
  }
  for(int sweep = completed_sweeps + 1;
      sweep <= options.maximum_sweeps && !converged; ++sweep) {
    const double objective_before = objective_history.back();
    for(std::size_t first = 0; first < orbital_indices.size(); ++first) {
      for(std::size_t second = first + 1;
          second < orbital_indices.size(); ++second) {
        double a = 0.0;
        double b = 0.0;
        double cross = 0.0;
        for(const linalg::Matrix& atom : populations) {
          const double difference =
              0.5 * (atom(first, first) - atom(second, second));
          const double coupling = atom(first, second);
          a += difference * difference;
          b += coupling * coupling;
          cross += difference * coupling;
        }
        const double largest =
            0.5 * (a + b + std::hypot(a - b, 2.0 * cross));
        const double predicted_gain = 2.0 * (largest - a);
        const double resolved_gain =
            64.0 * std::numeric_limits<double>::epsilon() *
            std::max(1.0, objective_before);
        if(predicted_gain <= resolved_gain) {
          continue;
        }
        const double angle =
            0.25 * std::atan2(2.0 * cross, a - b);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        for(linalg::Matrix& atom : populations) {
          rotate_population_pair(atom, first, second, cosine, sine);
        }
        rotate_pair(rotation, first, second, cosine, sine);
      }
    }
    const double objective_after = objective(populations, options.parallel);
    final_sweep_gain = objective_after - objective_before;
    const double decrease_tolerance =
        256.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, objective_before);
    if(final_sweep_gain < -decrease_tolerance) {
      throw std::runtime_error(
          "Pipek-Mezey objective decreased beyond roundoff");
    }
    objective_history.push_back(objective_after);
    completed_sweeps = sweep;
    converged = std::abs(final_sweep_gain) <=
                options.objective_tolerance;
    report();
  }
  if(!converged) {
    throw std::runtime_error(
        "Pipek-Mezey localization did not converge within maximum_sweeps");
  }

  linalg::Matrix localized = root_matrix_operation(
      basis.number_of_aos, orbital_indices.size(), options.parallel,
      [&]() { return linalg::multiply(canonical_space, rotation); },
      "localized-orbital coefficient construction");
  const linalg::Matrix final_lowdin_coefficients = root_matrix_operation(
      basis.number_of_aos, orbital_indices.size(), options.parallel,
      [&]() { return linalg::multiply(lowdin_coefficients, rotation); },
      "final symmetric-Lowdin orbital construction");
  const std::vector<linalg::Matrix> final_population_matrices =
      population_matrices(final_lowdin_coefficients, basis,
                          molecule.atoms().size(), options.parallel);
  const linalg::Matrix final_populations =
      options.parallel.ranks > 1
          ? distributed_diagonal_populations(
                final_population_matrices, molecule.atoms().size(),
                options.parallel)
          : diagonal_populations(final_population_matrices);
  const double rotation_error = root_scalar_operation(
      options.parallel, [&]() { return orthogonality_error(rotation); },
      "Pipek-Mezey rotation invariant");
  const double orbital_error = root_scalar_operation(
      options.parallel,
      [&]() { return metric_orthonormality_error(localized, overlap); },
      "Pipek-Mezey orbital invariant");
  const double preserved_projector_error = root_scalar_operation(
      options.parallel,
      [&]() { return projector_error(canonical_space, localized); },
      "Pipek-Mezey projector invariant");
  constexpr double invariant_tolerance = 2.0e-10;
  if(rotation_error > invariant_tolerance ||
     orbital_error > invariant_tolerance ||
     preserved_projector_error > invariant_tolerance) {
    throw std::runtime_error(
        "Pipek-Mezey result violates orthogonality or projector invariants");
  }
  for(std::size_t orbital = 0; orbital < final_populations.rows();
      ++orbital) {
    double population_sum = 0.0;
    for(std::size_t atom = 0; atom < final_populations.columns(); ++atom) {
      population_sum += final_populations(orbital, atom);
    }
    if(std::abs(population_sum - 1.0) > invariant_tolerance) {
      throw std::runtime_error(
          "Löwdin atomic populations do not sum to one");
    }
  }

  return PipekMezeyResult{
      .coefficients = std::move(localized),
      .rotation = std::move(rotation),
      .lowdin_populations = final_populations,
      .assignments =
          assignments(final_populations, molecule,
                      options.ambiguity_population_margin),
      .objective_history = std::move(objective_history),
      .completed_sweeps = completed_sweeps,
      .final_sweep_gain = final_sweep_gain,
      .rotation_orthogonality_maximum_error = rotation_error,
      .orbital_orthonormality_maximum_error = orbital_error,
      .projector_maximum_error = preserved_projector_error,
      .population_method = options.population_method,
      .optimizer = options.optimizer,
      .final_gradient_norm = final_gradient_norm,
      .total_keyframes = total_keyframes,
      .total_hessian_actions = total_hessian_actions,
  };
}

PipekMezeyDiagnosticPaths write_pipek_mezey_diagnostics(
    const PipekMezeyResult& result,
    const molecule::Molecule& molecule,
    const std::vector<std::size_t>& orbital_indices,
    const std::filesystem::path& prefix, bool overwrite) {
  if(prefix.empty()) {
    throw std::invalid_argument(
        "Pipek-Mezey diagnostic prefix may not be empty");
  }
  const std::size_t orbitals = result.rotation.rows();
  if(orbitals == 0 || result.rotation.columns() != orbitals ||
     result.coefficients.columns() != orbitals ||
     result.lowdin_populations.rows() != orbitals ||
     result.assignments.size() != orbitals ||
     orbital_indices.size() != orbitals) {
    throw std::invalid_argument(
        "Pipek-Mezey diagnostic dimensions are inconsistent");
  }
  if(result.lowdin_populations.columns() != molecule.atoms().size() ||
     molecule.monomer_ids().size() != molecule.atoms().size()) {
    throw std::invalid_argument(
        "Pipek-Mezey diagnostic atom and monomer dimensions disagree");
  }
  if(result.objective_history.empty() ||
     result.objective_history.size() !=
         static_cast<std::size_t>(result.completed_sweeps) + 1U) {
    throw std::invalid_argument(
        "Pipek-Mezey diagnostic objective history is inconsistent");
  }
  linalg::require_finite(result.rotation,
                         "Pipek-Mezey diagnostic rotation");
  linalg::require_finite(result.coefficients,
                         "Pipek-Mezey diagnostic coefficients");
  linalg::require_finite(result.lowdin_populations,
                         "Pipek-Mezey diagnostic populations");
  if(std::any_of(result.objective_history.begin(),
                 result.objective_history.end(),
                 [](double value) { return !std::isfinite(value); })) {
    throw std::invalid_argument(
        "Pipek-Mezey diagnostic objective history is not finite");
  }

  const PipekMezeyDiagnosticPaths paths{
      .rotation = diagnostic_path(prefix, "_rotation.csv"),
      .coefficients = diagnostic_path(prefix, "_coefficients.csv"),
      .monomer_populations =
          diagnostic_path(prefix, "_monomer_populations.csv"),
      .objective_history =
          diagnostic_path(prefix, "_objective_history.csv"),
      .summary = diagnostic_path(prefix, "_summary.json"),
  };
  require_diagnostic_outputs(paths, overwrite);

  std::vector<int> monomer_ids = molecule.monomer_ids();
  std::sort(monomer_ids.begin(), monomer_ids.end());
  monomer_ids.erase(std::unique(monomer_ids.begin(), monomer_ids.end()),
                    monomer_ids.end());
  if(monomer_ids.empty()) {
    throw std::invalid_argument(
        "Pipek-Mezey diagnostics require at least one monomer");
  }

  write_matrix_csv(paths.rotation, result.rotation);
  write_matrix_csv(paths.coefficients, result.coefficients);

  {
    std::ofstream stream{paths.monomer_populations, std::ios::trunc};
    if(!stream) {
      throw std::runtime_error(
          "unable to open Pipek-Mezey population diagnostics: " +
          paths.monomer_populations.string());
    }
    stream << "localized_orbital,canonical_orbital,assigned_atom,"
              "assigned_monomer,atom_population,second_atom_population,"
              "atom_population_margin,atom_ambiguous,monomer_population,"
              "second_monomer_population,monomer_population_margin,"
              "monomer_ambiguous";
    for(const int monomer : monomer_ids) {
      stream << ",pop_monomer_" << monomer;
    }
    stream << '\n' << std::setprecision(17);
    for(std::size_t orbital = 0; orbital < orbitals; ++orbital) {
      const OrbitalAssignment& assignment = result.assignments[orbital];
      if(assignment.atom_index >= molecule.atoms().size()) {
        throw std::invalid_argument(
            "Pipek-Mezey diagnostic assignment atom is invalid");
      }
      std::vector<double> populations(monomer_ids.size(), 0.0);
      for(std::size_t atom = 0; atom < molecule.atoms().size(); ++atom) {
        const int atom_monomer = molecule.monomer_ids()[atom];
        const auto position = std::lower_bound(
            monomer_ids.begin(), monomer_ids.end(), atom_monomer);
        if(position == monomer_ids.end() || *position != atom_monomer) {
          throw std::logic_error(
              "Pipek-Mezey diagnostic monomer lookup failed");
        }
        const std::size_t monomer_offset = static_cast<std::size_t>(
            std::distance(monomer_ids.begin(), position));
        populations[monomer_offset] +=
            result.lowdin_populations(orbital, atom);
      }
      stream << orbital << ',' << orbital_indices[orbital] << ','
             << assignment.atom_index << ',' << assignment.monomer_id << ','
             << assignment.atom_population << ','
             << assignment.second_atom_population << ','
             << assignment.atom_population_margin << ','
             << (assignment.atom_ambiguous ? 1 : 0) << ','
             << assignment.monomer_population << ','
             << assignment.second_monomer_population << ','
             << assignment.monomer_population_margin << ','
             << (assignment.monomer_ambiguous ? 1 : 0);
      for(const double population : populations) {
        stream << ',' << population;
      }
      stream << '\n';
    }
    if(!stream) {
      throw std::runtime_error(
          "failed while writing Pipek-Mezey population diagnostics: " +
          paths.monomer_populations.string());
    }
  }

  {
    std::ofstream stream{paths.objective_history, std::ios::trunc};
    if(!stream) {
      throw std::runtime_error(
          "unable to open Pipek-Mezey objective diagnostics: " +
          paths.objective_history.string());
    }
    stream << "sweep,objective,gain\n" << std::setprecision(17);
    for(std::size_t sweep = 0; sweep < result.objective_history.size();
        ++sweep) {
      const double gain = sweep == 0
                              ? 0.0
                              : result.objective_history[sweep] -
                                    result.objective_history[sweep - 1];
      stream << sweep << ',' << result.objective_history[sweep] << ','
             << gain << '\n';
    }
    if(!stream) {
      throw std::runtime_error(
          "failed while writing Pipek-Mezey objective diagnostics: " +
          paths.objective_history.string());
    }
  }

  {
    std::ofstream stream{paths.summary, std::ios::trunc};
    if(!stream) {
      throw std::runtime_error(
          "unable to open Pipek-Mezey diagnostic summary: " +
          paths.summary.string());
    }
    stream << std::setprecision(17)
           << "{\n"
           << "  \"index_base\": 0,\n"
           << "  \"population_method\": \""
           << (result.population_method ==
                       PipekMezeyPopulationMethod::meta_lowdin
                   ? "meta_lowdin_objective_symmetric_lowdin_assignment"
                   : "symmetric_lowdin_objective_and_assignment")
           << "\",\n"
           << "  \"optimizer\": \""
           << (result.optimizer == PipekMezeyOptimizer::pyscf_ciah
                   ? "pyscf_2_11_ciah"
                   : "deterministic_jacobi")
           << "\",\n"
           << "  \"ao_count\": " << result.coefficients.rows() << ",\n"
           << "  \"orbital_count\": " << orbitals << ",\n"
           << "  \"atom_count\": " << molecule.atoms().size() << ",\n"
           << "  \"monomer_count\": " << monomer_ids.size() << ",\n"
           << "  \"completed_sweeps\": " << result.completed_sweeps
           << ",\n"
           << "  \"objective_initial\": "
           << result.objective_history.front() << ",\n"
           << "  \"objective_final\": "
           << result.objective_history.back() << ",\n"
           << "  \"final_sweep_gain\": " << result.final_sweep_gain
           << ",\n"
           << "  \"final_gradient_norm\": "
           << result.final_gradient_norm << ",\n"
           << "  \"total_keyframes\": " << result.total_keyframes
           << ",\n"
           << "  \"total_hessian_actions\": "
           << result.total_hessian_actions << ",\n"
           << "  \"rotation_orthogonality_maximum_error\": "
           << result.rotation_orthogonality_maximum_error << ",\n"
           << "  \"orbital_orthonormality_maximum_error\": "
           << result.orbital_orthonormality_maximum_error << ",\n"
           << "  \"projector_maximum_error\": "
           << result.projector_maximum_error << "\n"
           << "}\n";
    if(!stream) {
      throw std::runtime_error(
          "failed while writing Pipek-Mezey diagnostic summary: " +
          paths.summary.string());
    }
  }

  return paths;
}

}  // namespace modernqc::localization
