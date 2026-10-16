#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace modernqc::mp2 {

enum class LaplaceQuadratureScheme {
  wa_continuous,
  gauss_laguerre,
  minimax,
};

struct DenominatorHistogram {
  double denominator_minimum;
  double denominator_maximum;
  std::uint64_t total_ordered_denominators;
  std::vector<double> edges;
  std::vector<double> midpoints;
  std::vector<std::uint64_t> counts;
  std::vector<double> frequencies;
};

struct LaplaceFitOptions {
  std::size_t histogram_bins{1200};
  std::size_t number_of_points{20};
  std::size_t gap_block_size{256};
  std::size_t validation_grid_size{4800};
  std::size_t maximum_optimizer_iterations{12000};
  std::size_t threads{1};
  std::size_t mpi_rank{0};
  std::size_t mpi_ranks{1};
  double relative_singular_value_cutoff{1.0e-12};
  double normal_equation_condition_threshold{1.0e8};
  double initial_simplex_step{0.08};
  double logarithmic_node_tolerance{1.0e-8};
  double objective_tolerance{1.0e-18};
  LaplaceQuadratureScheme scheme{LaplaceQuadratureScheme::wa_continuous};
};

struct LaplaceFitResult {
  DenominatorHistogram histogram;
  std::vector<double> nodes;
  std::vector<double> weights;
  std::vector<double> optimization_history;
  std::vector<double> condition_number_history;
  std::size_t optimizer_iterations;
  std::size_t weight_numerical_rank;
  double weighted_objective;
  double weighted_rmse;
  double histogram_maximum_absolute_error;
  double histogram_maximum_relative_error;
  double validation_maximum_absolute_error;
  double validation_maximum_relative_error;
};

[[nodiscard]] DenominatorHistogram build_denominator_histogram(
    const std::vector<double>& orbital_energies,
    const std::vector<std::size_t>& active_occupied_indices,
    const std::vector<std::size_t>& virtual_indices,
    const LaplaceFitOptions& options = {});

[[nodiscard]] LaplaceFitResult fit_laplace_quadrature(
    const DenominatorHistogram& histogram,
    const LaplaceFitOptions& options = {});

[[nodiscard]] LaplaceFitResult build_and_fit_laplace_quadrature(
    const std::vector<double>& orbital_energies,
    const std::vector<std::size_t>& active_occupied_indices,
    const std::vector<std::size_t>& virtual_indices,
    const LaplaceFitOptions& options = {});

[[nodiscard]] double evaluate_laplace_reciprocal(
    double denominator, const std::vector<double>& nodes,
    const std::vector<double>& weights);

}  // namespace modernqc::mp2
