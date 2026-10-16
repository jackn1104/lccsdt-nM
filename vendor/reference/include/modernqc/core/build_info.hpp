#pragma once

#include <string>

namespace modernqc::core {

struct BuildInfo {
  std::string version;
  std::string compiler;
  std::string build_type;
  bool mpi_enabled;
  bool openmp_enabled;
  bool hdf5_enabled;
  bool libint2_enabled;
};

[[nodiscard]] BuildInfo build_info();
[[nodiscard]] std::string format_build_info(const BuildInfo& info);

}  // namespace modernqc::core
