#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string_view>

namespace localized_cc {

enum class SpatialDiisMetric { raw, rms, preconditioned_rms };

inline SpatialDiisMetric spatial_diis_metric(std::string_view name) {
  if (name == "raw") return SpatialDiisMetric::raw;
  if (name == "rms") return SpatialDiisMetric::rms;
  if (name == "preconditioned-rms") return SpatialDiisMetric::preconditioned_rms;
  throw std::invalid_argument("CCSD_SPATIAL_DIIS_METRIC must be raw, rms, or preconditioned-rms");
}

// Fixed, iteration-independent block weights. Never normalize by the current
// residual norm: that would distort the relative magnitudes of history vectors.
// Denominators have the same sign/order as the Jacobi update, not damping.
inline void weight_spatial_diis_error(
    std::span<double> error, std::span<const double> d1,
    std::span<const double> d2, SpatialDiisMetric metric) {
  if (d1.empty() || d2.empty() || d1.size() > error.size() ||
      d2.size() != error.size() - d1.size())
    throw std::invalid_argument("spatial DIIS error/denominator dimensions mismatch");
  if (metric != SpatialDiisMetric::raw && metric != SpatialDiisMetric::rms &&
      metric != SpatialDiisMetric::preconditioned_rms)
    throw std::invalid_argument("unknown spatial DIIS error metric");
  const double s1 = std::sqrt(static_cast<double>(d1.size()));
  const double s2 = std::sqrt(static_cast<double>(d2.size()));
  for (std::size_t k = 0; k < error.size(); ++k) {
    if (!std::isfinite(error[k]))
      throw std::invalid_argument("nonfinite spatial DIIS residual");
    if (metric == SpatialDiisMetric::raw) continue;
    if (metric == SpatialDiisMetric::preconditioned_rms) {
      const double d = k < d1.size() ? d1[k] : d2[k-d1.size()];
      if (!std::isfinite(d) || d == 0.0)
        throw std::invalid_argument("invalid spatial DIIS preconditioner");
      error[k] /= d;
    }
    error[k] /= (k < d1.size() ? s1 : s2);
    if (!std::isfinite(error[k]))
      throw std::overflow_error("nonfinite weighted spatial DIIS error");
  }
}
} // namespace localized_cc
