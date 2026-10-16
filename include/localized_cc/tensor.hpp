#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace localized_cc {

template <std::size_t Rank> class Tensor {
public:
  Tensor() = default;
  explicit Tensor(const std::array<std::size_t, Rank>& dimensions, double value = 0.0)
      : dimensions_(dimensions) {
    std::size_t count = 1;
    for (std::size_t axis = Rank; axis-- > 0;) {
      strides_[axis] = count;
      if (dimensions_[axis] == 0 || count > max_size() / dimensions_[axis])
        throw std::overflow_error("invalid or overflowing tensor dimensions");
      count *= dimensions_[axis];
    }
    data_.assign(count, value);
  }

  template <typename... Index>
  double& operator()(Index... index) {
    static_assert(sizeof...(Index) == Rank);
    return data_.at(offset({static_cast<std::size_t>(index)...}));
  }
  template <typename... Index>
  const double& operator()(Index... index) const {
    static_assert(sizeof...(Index) == Rank);
    return data_.at(offset({static_cast<std::size_t>(index)...}));
  }
  [[nodiscard]] const std::array<std::size_t, Rank>& dimensions() const noexcept { return dimensions_; }
  [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
  [[nodiscard]] std::vector<double>& data() noexcept { return data_; }
  [[nodiscard]] const std::vector<double>& data() const noexcept { return data_; }
  void fill(double value) { std::fill(data_.begin(), data_.end(), value); }

private:
  static constexpr std::size_t max_size() { return static_cast<std::size_t>(-1); }
  std::size_t offset(const std::array<std::size_t, Rank>& index) const {
    std::size_t result = 0;
    for (std::size_t axis = 0; axis < Rank; ++axis) {
      if (index[axis] >= dimensions_[axis]) throw std::out_of_range("tensor index out of range");
      result += index[axis] * strides_[axis];
    }
    return result;
  }
  std::array<std::size_t, Rank> dimensions_{};
  std::array<std::size_t, Rank> strides_{};
  std::vector<double> data_;
};

using Matrix = Tensor<2>;
using Tensor4 = Tensor<4>;
using Tensor6 = Tensor<6>;

// Exact dense four-center chemists'-notation ERIs stored with
// (pq|rs)=(qp|rs)=(pq|sr)=(rs|pq) permutation symmetry. Every logical
// four-index element remains addressable; only symmetry-equivalent copies are
// removed from storage.
class SymmetricEri4 {
public:
  SymmetricEri4() = default;
  explicit SymmetricEri4(const std::size_t orbitals)
      : dimensions_{orbitals, orbitals, orbitals, orbitals},
        data_(stored_elements(orbitals), 0.0),
        pair_indices_(checked_product(orbitals, orbitals), 0),
        pair_offsets_(triangular(orbitals), 0) {
    if (orbitals == 0)
      throw std::invalid_argument(
          "symmetric ERI orbital dimension must be positive");
    for (std::size_t first = 0; first < orbitals; ++first)
      for (std::size_t second = 0; second < orbitals; ++second) {
        const auto high = std::max(first, second);
        const auto low = std::min(first, second);
        pair_indices_[first * orbitals + second] = triangular(high) + low;
      }
    for (std::size_t pair = 0; pair < pair_offsets_.size(); ++pair)
      pair_offsets_[pair] = triangular(pair);
  }

  explicit SymmetricEri4(const std::array<std::size_t, 4>& dimensions)
      : SymmetricEri4(validated_orbitals(dimensions)) {}

  explicit SymmetricEri4(const Tensor4& dense)
      : SymmetricEri4(validated_orbitals(dense.dimensions())) {
    std::size_t first_pair = 0;
    for (std::size_t p = 0; p < orbitals(); ++p)
      for (std::size_t q = 0; q <= p; ++q, ++first_pair) {
        std::size_t second_pair = 0;
        for (std::size_t r = 0; r < orbitals(); ++r)
          for (std::size_t s = 0; s <= r; ++s, ++second_pair)
            if (second_pair <= first_pair)
              (*this)(p, q, r, s) = dense(p, q, r, s);
      }
  }

  double& operator()(const std::size_t p, const std::size_t q,
                     const std::size_t r, const std::size_t s) {
    return data_.at(offset(p, q, r, s));
  }
  const double& operator()(const std::size_t p, const std::size_t q,
                           const std::size_t r,
                           const std::size_t s) const {
    return data_.at(offset(p, q, r, s));
  }

  [[nodiscard]] const std::array<std::size_t, 4>& dimensions() const noexcept {
    return dimensions_;
  }
  [[nodiscard]] std::size_t orbitals() const noexcept {
    return dimensions_[0];
  }
  [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
  [[nodiscard]] std::size_t logical_size() const {
    return checked_product(checked_product(orbitals(), orbitals()),
                           checked_product(orbitals(), orbitals()));
  }
  [[nodiscard]] std::vector<double>& data() noexcept { return data_; }
  [[nodiscard]] const std::vector<double>& data() const noexcept {
    return data_;
  }
  void fill(const double value) { std::fill(data_.begin(), data_.end(), value); }

  [[nodiscard]] static std::size_t stored_elements(
      const std::size_t orbitals) {
    const auto pairs = triangular(orbitals);
    return triangular(pairs);
  }

private:
  [[nodiscard]] static std::size_t checked_product(const std::size_t left,
                                                   const std::size_t right) {
    if (left != 0 &&
        right > std::numeric_limits<std::size_t>::max() / left)
      throw std::overflow_error("symmetric ERI logical size overflows");
    return left * right;
  }

  [[nodiscard]] static std::size_t triangular(const std::size_t value) {
    if (value == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("symmetric ERI pair count overflows");
    return value % 2 == 0
               ? checked_product(value / 2, value + 1)
               : checked_product(value, (value + 1) / 2);
  }

  [[nodiscard]] static std::size_t validated_orbitals(
      const std::array<std::size_t, 4>& dimensions) {
    if (dimensions[0] == 0 || dimensions[1] != dimensions[0] ||
        dimensions[2] != dimensions[0] || dimensions[3] != dimensions[0])
      throw std::invalid_argument(
          "symmetric ERI dimensions must be nonzero and equal");
    return dimensions[0];
  }

  [[nodiscard]] std::size_t pair_index(const std::size_t first,
                                       const std::size_t second) const {
    if (first >= orbitals() || second >= orbitals())
      throw std::out_of_range("symmetric ERI index out of range");
    return pair_indices_[first * orbitals() + second];
  }

  [[nodiscard]] std::size_t offset(const std::size_t p, const std::size_t q,
                                   const std::size_t r,
                                   const std::size_t s) const {
    const auto first_pair = pair_index(p, q);
    const auto second_pair = pair_index(r, s);
    const auto high = std::max(first_pair, second_pair);
    const auto low = std::min(first_pair, second_pair);
    return pair_offsets_[high] + low;
  }

  std::array<std::size_t, 4> dimensions_{};
  std::vector<double> data_;
  std::vector<std::size_t> pair_indices_;
  std::vector<std::size_t> pair_offsets_;
};

inline double dot(const std::vector<double>& x, const std::vector<double>& y) {
  if (x.size() != y.size()) throw std::invalid_argument("dot product size mismatch");
  if (x.size() >
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::overflow_error("dot product extent exceeds ptrdiff_t");
  double result = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : result) if(x.size() >= 16384)
#endif
  for (std::ptrdiff_t index = 0;
       index < static_cast<std::ptrdiff_t>(x.size()); ++index)
    result += x[static_cast<std::size_t>(index)] *
              y[static_cast<std::size_t>(index)];
  return result;
}
inline double norm(const std::vector<double>& x) {
  return std::sqrt(dot(x, x));
}

} // namespace localized_cc
