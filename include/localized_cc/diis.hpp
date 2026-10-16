#pragma once

#include "localized_cc/diis_solver.hpp"
#include "localized_cc/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace localized_cc {

enum class DiisGramBackend { recomputed, incremental };

struct DiisWork {
  std::size_t error_dot_products{};
  std::size_t history_vector_accumulations{};
};

class Diis {
public:
  explicit Diis(
      std::size_t capacity,
      DiisGramBackend gram_backend = DiisGramBackend::recomputed)
      : capacity_(capacity), gram_backend_(gram_backend) {
    if (capacity < 2)
      throw std::invalid_argument("DIIS capacity must be at least two");
  }
  std::vector<double> extrapolate(const std::vector<double>& vector,
                                  const std::vector<double>& error) {
    if (vector.size() != error.size())
      throw std::invalid_argument("DIIS vector/error size mismatch");
    const auto finite = [](const double value) { return std::isfinite(value); };
    if (!std::all_of(vector.begin(), vector.end(), finite) ||
        !std::all_of(error.begin(), error.end(), finite))
      throw std::invalid_argument(
          "DIIS vector/error contains a non-finite value");
    last_work_ = {};
    if (vectors_.size() == capacity_) remove_oldest(1);
    if (gram_backend_ == DiisGramBackend::incremental) {
      const auto old_count = errors_.size();
      for (auto& row : error_products_) row.push_back(0.0);
      std::vector<double> new_row(old_count + 1, 0.0);
      for (std::size_t i = 0; i < old_count; ++i) {
        const auto product = dot(errors_[i], error);
        error_products_[i].back() = product;
        new_row[i] = product;
        ++last_work_.error_dot_products;
      }
      new_row.back() = dot(error, error);
      ++last_work_.error_dot_products;
      error_products_.push_back(std::move(new_row));
    }
    vectors_.push_back(vector);
    errors_.push_back(error);
    const auto count = vectors_.size();
    if (count < 2)
      return vector;
    std::vector<std::vector<double>> recomputed_products;
    const std::vector<std::vector<double>>* products = &error_products_;
    if (gram_backend_ == DiisGramBackend::recomputed) {
      recomputed_products.assign(count, std::vector<double>(count));
      for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = 0; j < count; ++j) {
          recomputed_products[i][j] = dot(errors_[i], errors_[j]);
          ++last_work_.error_dot_products;
        }
      products = &recomputed_products;
    }
    const auto solution = solve_diis_coefficients(*products);
    if (!solution)
      return vector;
    remove_oldest(solution->first_history);
    std::vector<double> out(vector.size());
    last_work_.history_vector_accumulations = solution->coefficients.size();
    if (out.size() >
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
      throw std::overflow_error("DIIS vector extent exceeds ptrdiff_t");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(out.size() >= 16384)
#endif
    for (std::ptrdiff_t element = 0;
         element < static_cast<std::ptrdiff_t>(out.size()); ++element) {
      const auto index = static_cast<std::size_t>(element);
      double value = 0.0;
      for (std::size_t history = 0;
           history < solution->coefficients.size(); ++history)
        value += solution->coefficients[history] * vectors_[history][index];
      out[index] = value;
    }
    return out;
  }

  [[nodiscard]] DiisWork last_work() const noexcept { return last_work_; }
  [[nodiscard]] std::size_t history_size() const noexcept {
    return vectors_.size();
  }

private:
  void remove_oldest(const std::size_t count) {
    if (count > vectors_.size() || count > errors_.size())
      throw std::logic_error("DIIS history removal exceeds stored history");
    for (std::size_t i = 0; i < count; ++i) {
      vectors_.pop_front();
      errors_.pop_front();
      if (gram_backend_ == DiisGramBackend::incremental) {
        error_products_.erase(error_products_.begin());
        for (auto& row : error_products_) row.erase(row.begin());
      }
    }
  }

  std::size_t capacity_;
  DiisGramBackend gram_backend_;
  std::deque<std::vector<double>> vectors_;
  std::deque<std::vector<double>> errors_;
  std::vector<std::vector<double>> error_products_;
  DiisWork last_work_;
};

} // namespace localized_cc
