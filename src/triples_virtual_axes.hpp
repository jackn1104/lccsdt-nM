#pragma once
#include "triples_occupied_panel.hpp"

namespace localized_cc::detail {
// Column-major tensor x + n*(y+n*z). Each BLAS call is bounded by n x n;
// the last axis is explicitly gathered, avoiding a cube-sized leading dimension.
inline void transform_virtual_axes(std::span<const double> source,
    std::span<const double> propagator, std::size_t extent,
    std::span<double> first, std::span<double> second) {
  if (!extent || extent > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      extent > std::numeric_limits<std::size_t>::max()/extent)
    throw std::invalid_argument("virtual axes invalid extent");
  const auto square=extent*extent;
  if (square>std::numeric_limits<std::size_t>::max()/extent ||
      propagator.size()!=square || source.size()!=square*extent ||
      first.size()!=source.size() || second.size()!=source.size())
    throw std::invalid_argument("virtual axes buffer dimensions disagree");
  const int n=static_cast<int>(extent);
  constexpr char normal='N', transpose='T';
  constexpr double alpha=1., beta=0.;
  for(std::size_t z=0;z<extent;++z) {
    dgemm_(&normal,&normal,&n,&n,&n,&alpha,propagator.data(),&n,
        source.data()+z*square,&n,&beta,first.data()+z*square,&n);
    dgemm_(&normal,&transpose,&n,&n,&n,&alpha,first.data()+z*square,&n,
        propagator.data(),&n,&beta,second.data()+z*square,&n);
  }
  std::vector<double> packed(square), transformed(square);
  for(std::size_t y=0;y<extent;++y) {
    for(std::size_t z=0;z<extent;++z)
      std::copy_n(second.data()+extent*y+square*z,extent,packed.data()+extent*z);
    dgemm_(&normal,&transpose,&n,&n,&n,&alpha,packed.data(),&n,
        propagator.data(),&n,&beta,transformed.data(),&n);
    for(std::size_t z=0;z<extent;++z)
      std::copy_n(transformed.data()+extent*z,extent,first.data()+extent*y+square*z);
  }
}
}
