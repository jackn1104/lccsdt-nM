#pragma once
#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace localized_cc::detail {
extern "C" void dgemm_(const char*, const char*, const int*, const int*, const int*,
    const double*, const double*, const int*, const double*, const int*,
    const double*, double*, const int*);

// One occupied target, bounded contiguous virtual panel. Source layout is
// [occupied source][virtual triple]. No tensor/model truncation is performed.
inline void contract_disk_occupied_panel(std::span<const double> source,
    std::size_t virtual_count, std::span<const double> coefficients,
    std::size_t first, std::span<double> output, std::vector<double>& packed) {
  const auto count=output.size(), ns=coefficients.size();
  if(!virtual_count||!ns||!count||first>virtual_count||count>virtual_count-first||
      ns>std::numeric_limits<std::size_t>::max()/virtual_count||
      source.size()!=ns*virtual_count||
      ns>static_cast<std::size_t>(std::numeric_limits<int>::max())||
      count>static_cast<std::size_t>(std::numeric_limits<int>::max())||
      ns>std::numeric_limits<std::size_t>::max()/count)
    throw std::invalid_argument("disk occupied panel extent/BLAS mismatch");
  packed.resize(ns*count);
  for(std::size_t k=0;k<ns;++k)
    std::copy_n(source.data()+k*virtual_count+first,count,packed.data()+k*count);
  const int m=static_cast<int>(count),n=1,k=static_cast<int>(ns);
  constexpr char trans='N';constexpr double alpha=1.,beta=0.;
  dgemm_(&trans,&trans,&m,&n,&k,&alpha,packed.data(),&m,
      coefficients.data(),&k,&beta,output.data(),&m);
}
} // namespace localized_cc::detail
