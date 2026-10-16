#pragma once
#include "localized_cc/pm_symmetry.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace localized_cc::detail {
// Signed, real C3/C4 action on external tensor indices. Internal sums remain full.
// A null group is the unchanged identity executor; no representative list.
template<std::size_t N> class PmOrbitPlan {
 public:
  PmOrbitPlan(const PmSymmetry* group, std::array<std::size_t,N> dimensions,
              std::array<std::size_t,N> offsets, bool spin=false,
              std::size_t maximum_bytes=std::numeric_limits<std::size_t>::max())
      : group_(group), dimensions_(dimensions), offsets_(offsets), spin_(spin) {
    if (group_) validate_pm_symmetry_group(*group_);
    for (std::size_t a=0;a<N;++a) {
      if (!dimensions_[a] || size_>std::numeric_limits<std::size_t>::max()/dimensions_[a])
        throw std::overflow_error("PM orbit tensor extent overflow");
      size_*=dimensions_[a];
      if (group_) {
        const auto extent=offsets_[a]==0 ? group_->occupied : group_->virtuals;
        if (spin_ && extent>std::numeric_limits<std::size_t>::max()/2)
          throw std::overflow_error("PM spin-axis extent overflow");
        if ((offsets_[a]!=0 && offsets_[a]!=group_->occupied) ||
            dimensions_[a]!=(spin_?2:1)*extent)
          throw std::invalid_argument("PM orbit axis does not cover an orbital space");
      }
    }
    if (group_) for (std::size_t p=0;p<size_;++p) {
      std::size_t representative=p;
      for (std::size_t g=1;g<group_->operations.size();++g)
        representative=std::min(representative,map(g,p).first);
      if (representative==p) {
        if (representatives_.size()>=maximum_bytes/sizeof(std::size_t))
          throw std::runtime_error("PM representative metadata exceeds memory cap");
        if (representatives_.size()==representatives_.capacity()) {
          const auto cap=maximum_bytes/sizeof(std::size_t);
          const auto old=representatives_.capacity();
          representatives_.reserve(old>cap/2?cap:std::min(cap,std::max(std::size_t{1},2*old)));
        }
        representatives_.push_back(p);
      }
    }
  }
  std::size_t size() const { return size_; }
  std::size_t tasks() const { return group_?representatives_.size():size_; }
  std::size_t representative(std::size_t task) const { return group_?representatives_[task]:task; }
  bool reduced() const { return group_!=nullptr; }
  std::pair<std::size_t,int> map(std::size_t g,std::size_t flat) const {
    if (!group_) return {flat,1};
    const auto& operation=group_->operations[g];
    std::array<std::size_t,N> mapped{};
    int sign=1;
    for (std::size_t axis=N;axis-- >0;) {
      const auto x=flat%dimensions_[axis]; flat/=dimensions_[axis];
      const auto p=offsets_[axis]+(spin_?x/2:x);
      mapped[axis]=(operation.image[p]-offsets_[axis])*(spin_?2:1)+(spin_?x%2:0);
      sign*=operation.sign[p];
    }
    std::size_t index=0;
    for (std::size_t axis=0;axis<N;++axis) index=index*dimensions_[axis]+mapped[axis];
    return {index,sign};
  }
  template<class F> void images(std::size_t flat,const F& visit) const {
    if (!group_) { visit(0,flat,1); return; }
    std::array<std::size_t,4> seen{}; // Validated group order is at most four.
    std::size_t count=0;
    for (std::size_t g=0;g<group_->operations.size();++g) {
      const auto [index,sign]=map(g,flat);
      if (std::find(seen.begin(),seen.begin()+count,index)!=seen.begin()+count) continue;
      seen[count++]=index;
      visit(g,index,sign);
    }
  }
  void scatter(std::size_t flat,double value,std::span<double> output) const {
    images(flat,[&](std::size_t,std::size_t index,int sign) { output[index]=sign*value; });
  }
 private:
  const PmSymmetry* group_;
  std::array<std::size_t,N> dimensions_,offsets_;
  bool spin_;
  std::size_t size_{1};
  std::vector<std::size_t> representatives_;
};
} // namespace localized_cc::detail
