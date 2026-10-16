#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace localized_cc::parallel {

struct World {
  std::size_t rank{};
  std::size_t size{1};
};

class Runtime {
public:
  Runtime(int& argc, char**& argv);
  ~Runtime();

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  [[nodiscard]] World world() const;
  [[noreturn]] void abort(int error_code) const;

private:
  [[maybe_unused]] bool owns_runtime_{};
};

[[nodiscard]] World world();
void sum_in_place(std::vector<double>& values);
// Ordered, main-thread tile transfer. All ranks call in the same order; only
// source and destination communicate. Buffer sizes must agree on those ranks.
void transfer_tile(std::vector<double>& values, std::size_t source,
                   std::size_t destination);
// Collective sum over ranks that supply the same color. Every world rank must
// call this operation in the same order, although colors may differ.
void sum_in_place_group(std::vector<double>& values, std::size_t color);
[[nodiscard]] std::uint64_t sum_group(std::uint64_t value,
                                      std::size_t color);
void broadcast_in_place(std::vector<double>& values, std::size_t root = 0);
void broadcast_in_place(std::vector<int>& values, std::size_t root = 0);
void barrier();

} // namespace localized_cc::parallel
