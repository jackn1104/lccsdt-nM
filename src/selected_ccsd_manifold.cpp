#include "localized_cc/selected_ccsd_manifold.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace localized_cc {
namespace {

std::size_t checked_product(const std::size_t left,
                            const std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error("selected CCSD manifold size overflows size_t");
  return left * right;
}

std::size_t checked_add(const std::size_t left, const std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error("selected CCSD manifold count overflows size_t");
  return left + right;
}

std::size_t monomer_count(const std::vector<int>& occupied_labels,
                          const std::vector<int>& virtual_labels) {
  int maximum = -1;
  const auto inspect = [&maximum](const std::vector<int>& labels) {
    for (const auto label : labels) {
      if (label < 0)
        throw std::invalid_argument(
            "selected CCSD manifold requires confident monomer labels");
      maximum = std::max(maximum, label);
    }
  };
  inspect(occupied_labels);
  inspect(virtual_labels);
  return static_cast<std::size_t>(maximum) + 1;
}

std::vector<std::vector<std::size_t>> group_orbitals(
    const std::vector<int>& labels, const std::size_t count) {
  std::vector<std::vector<std::size_t>> result(count);
  for (std::size_t orbital = 0; orbital < labels.size(); ++orbital)
    result[static_cast<std::size_t>(labels[orbital])].push_back(orbital);
  return result;
}

class InteractionGraph {
public:
  InteractionGraph(const std::size_t monomers,
                   const SelectedCcsdManifoldOptions& options)
      : monomers_(monomers), complete_(options.complete_interaction_graph) {
    for (const auto& [left, right] : options.interacting_monomer_pairs) {
      if (left >= monomers_ || right >= monomers_)
        throw std::invalid_argument(
            "selected CCSD interaction edge contains an unknown monomer");
      edges_.insert(canonical_edge(left, right));
    }
  }

  [[nodiscard]] bool contains(const std::size_t left,
                              const std::size_t right) const {
    return left == right || complete_ ||
           edges_.contains(canonical_edge(left, right));
  }

private:
  static std::pair<std::size_t, std::size_t>
  canonical_edge(const std::size_t left, const std::size_t right) {
    return std::minmax(left, right);
  }

  std::size_t monomers_{};
  bool complete_{};
  std::set<std::pair<std::size_t, std::size_t>> edges_;
};

template <std::size_t Size>
std::vector<std::size_t> participating_monomers(
    const std::array<std::size_t, Size>& labels) {
  auto ordered = labels;
  std::sort(ordered.begin(), ordered.end());
  const auto end = std::unique(ordered.begin(), ordered.end());
  return {ordered.begin(), end};
}

bool retained(const std::vector<std::size_t>& participating,
              const SelectedCcsdManifoldOptions& options,
              const InteractionGraph& graph) {
  if (participating.size() > options.maximum_monomers)
    return false;
  for (std::size_t left = 0; left < participating.size(); ++left)
    for (std::size_t right = left + 1; right < participating.size(); ++right)
      if (!graph.contains(participating[left], participating[right]))
        return false;
  return true;
}

using Space = CcsdOrbitalSpace;
constexpr Space O = Space::occupied;
constexpr Space V = Space::virtual_space;

std::uint32_t use_bit(const CcsdDependencyUse use) {
  return static_cast<std::uint32_t>(use);
}

struct FockKey {
  std::array<Space, 2> spaces{};
  std::array<std::size_t, 2> monomers{};

  friend bool operator<(const FockKey& left, const FockKey& right) {
    return std::tie(left.spaces, left.monomers) <
           std::tie(right.spaces, right.monomers);
  }
};

struct IntegralKey {
  std::array<Space, 4> spaces{};
  std::array<std::size_t, 4> monomers{};

  friend bool operator<(const IntegralKey& left,
                        const IntegralKey& right) {
    return std::tie(left.spaces, left.monomers) <
           std::tie(right.spaces, right.monomers);
  }
};

std::size_t block_extent(const SelectedCcsdManifoldPlan& plan,
                         const Space space, const std::size_t monomer) {
  return space == O ? plan.occupied_orbitals_by_monomer.at(monomer).size()
                    : plan.virtual_orbitals_by_monomer.at(monomer).size();
}

class MonomerSupport {
public:
  explicit MonomerSupport(const SelectedCcsdManifoldPlan& plan)
      : monomers_(plan.monomer_count),
        singles_(checked_product(monomers_, monomers_), 0U),
        doubles_(checked_product(checked_product(monomers_, monomers_),
                                 checked_product(monomers_, monomers_)),
                 0U),
        tau_(doubles_.size(), 0U) {
    for (const auto& block : plan.singles_blocks)
      singles_[index2(block.occupied_monomer,
                      block.virtual_monomer)] = 1U;
    for (const auto& block : plan.doubles_blocks)
      doubles_[index4(block.monomers)] = 1U;
    for (std::size_t i = 0; i < monomers_; ++i)
      for (std::size_t j = 0; j < monomers_; ++j)
        for (std::size_t a = 0; a < monomers_; ++a)
          for (std::size_t b = 0; b < monomers_; ++b) {
            const std::array<std::size_t, 4> labels{i, j, a, b};
            tau_[index4(labels)] =
                doubles(labels) ||
                (singles(i, a) && singles(j, b)) ||
                (singles(i, b) && singles(j, a));
          }
  }

  [[nodiscard]] bool singles(const std::size_t i,
                             const std::size_t a) const {
    return singles_[index2(i, a)] != 0U;
  }

  [[nodiscard]] bool doubles(
      const std::array<std::size_t, 4>& labels) const {
    return doubles_[index4(labels)] != 0U;
  }

  [[nodiscard]] bool doubles(const std::size_t i, const std::size_t j,
                             const std::size_t a,
                             const std::size_t b) const {
    return doubles({i, j, a, b});
  }

  [[nodiscard]] bool tau(const std::size_t i, const std::size_t j,
                         const std::size_t a,
                         const std::size_t b) const {
    return tau_[index4({i, j, a, b})] != 0U;
  }

private:
  [[nodiscard]] std::size_t index2(const std::size_t first,
                                   const std::size_t second) const {
    return first * monomers_ + second;
  }

  [[nodiscard]] std::size_t index4(
      const std::array<std::size_t, 4>& labels) const {
    return ((labels[0] * monomers_ + labels[1]) * monomers_ + labels[2]) *
               monomers_ +
           labels[3];
  }

  std::size_t monomers_{};
  std::vector<std::uint8_t> singles_;
  std::vector<std::uint8_t> doubles_;
  std::vector<std::uint8_t> tau_;
};

void audit_ccsd_dependencies(SelectedCcsdManifoldPlan& plan) {
  const auto monomers = plan.monomer_count;
  const MonomerSupport support(plan);
  std::map<FockKey, std::uint32_t> fock;
  std::map<IntegralKey, std::uint32_t> integral;
  const auto require_fock = [&](const std::array<Space, 2> spaces,
                                const std::array<std::size_t, 2> labels,
                                const CcsdDependencyUse use) {
    fock[FockKey{spaces, labels}] |= use_bit(use);
  };
  const auto require_integral =
      [&](const std::array<Space, 4> spaces,
          const std::array<std::size_t, 4> labels,
          const CcsdDependencyUse use) {
        integral[IntegralKey{spaces, labels}] |= use_bit(use);
      };

  // The diagonal denominators require every occupied and virtual Fock block.
  // The small F_ae/F_mi/F_me matrices are retained completely; their source
  // integral blocks below are nevertheless restricted by the selected T1,
  // T2, tau, and tilde-tau support.
  for (std::size_t first = 0; first < monomers; ++first) {
    for (std::size_t second = 0; second < monomers; ++second) {
      require_fock({O, O}, {first, second}, CcsdDependencyUse::fmi);
      require_fock({V, V}, {first, second}, CcsdDependencyUse::fae);
      require_fock({O, V}, {first, second}, CcsdDependencyUse::fme);
    }
    require_fock({O, O}, {first, first},
                 CcsdDependencyUse::denominator);
    require_fock({V, V}, {first, first},
                 CcsdDependencyUse::denominator);
  }
  plan.required_intermediate_blocks.fae_blocks =
      checked_product(monomers, monomers);
  plan.required_intermediate_blocks.fmi_blocks =
      checked_product(monomers, monomers);
  plan.required_intermediate_blocks.fme_blocks =
      checked_product(monomers, monomers);

  // Exact source closure for the three F intermediates.
  for (std::size_t a = 0; a < monomers; ++a)
    for (std::size_t e = 0; e < monomers; ++e)
      for (std::size_t m = 0; m < monomers; ++m) {
        if (support.singles(m, a))
          require_fock({O, V}, {m, e}, CcsdDependencyUse::fae);
        for (std::size_t q = 0; q < monomers; ++q)
          if (support.singles(m, q))
            require_integral({O, V, V, V}, {m, a, q, e},
                             CcsdDependencyUse::fae);
        for (std::size_t n = 0; n < monomers; ++n)
          for (std::size_t q = 0; q < monomers; ++q)
            if (support.tau(m, n, a, q))
              require_integral({O, O, V, V}, {m, n, e, q},
                               CcsdDependencyUse::fae);
      }

  for (std::size_t m = 0; m < monomers; ++m)
    for (std::size_t i = 0; i < monomers; ++i) {
      for (std::size_t e = 0; e < monomers; ++e)
        if (support.singles(i, e))
          require_fock({O, V}, {m, e}, CcsdDependencyUse::fmi);
      for (std::size_t n = 0; n < monomers; ++n)
        for (std::size_t e = 0; e < monomers; ++e) {
          if (support.singles(n, e))
            require_integral({O, O, O, V}, {m, n, i, e},
                             CcsdDependencyUse::fmi);
          for (std::size_t q = 0; q < monomers; ++q)
            if (support.tau(i, n, e, q))
              require_integral({O, O, V, V}, {m, n, e, q},
                               CcsdDependencyUse::fmi);
        }
    }

  for (std::size_t m = 0; m < monomers; ++m)
    for (std::size_t e = 0; e < monomers; ++e)
      for (std::size_t n = 0; n < monomers; ++n)
        for (std::size_t q = 0; q < monomers; ++q)
          if (support.singles(n, q))
            require_integral({O, O, V, V}, {m, n, e, q},
                             CcsdDependencyUse::fme);

  std::set<std::array<std::size_t, 4>> needed_wmnij;
  std::set<std::array<std::size_t, 4>> needed_wmbej;
  std::set<std::array<std::size_t, 4>> needed_wabef;
  const auto require_wm_term =
      [&](const std::size_t i, const std::size_t j,
          const std::size_t a, const std::size_t b) {
        for (std::size_t m = 0; m < monomers; ++m)
          for (std::size_t e = 0; e < monomers; ++e)
            if (support.doubles(i, m, a, e))
              needed_wmbej.insert({m, b, e, j});
      };
  for (const auto& external : plan.doubles_blocks) {
    const auto i = external.monomers[0];
    const auto j = external.monomers[1];
    const auto a = external.monomers[2];
    const auto b = external.monomers[3];
    for (std::size_t m = 0; m < monomers; ++m)
      for (std::size_t n = 0; n < monomers; ++n)
        if (support.tau(m, n, a, b))
          needed_wmnij.insert({m, n, i, j});
    require_wm_term(i, j, a, b);
    require_wm_term(j, i, a, b);
    require_wm_term(i, j, b, a);
    require_wm_term(j, i, b, a);
    for (std::size_t e = 0; e < monomers; ++e)
      for (std::size_t q = 0; q < monomers; ++q)
        if (support.tau(i, j, e, q))
          needed_wabef.insert({a, b, e, q});
  }
  plan.required_intermediate_blocks.wmnij_blocks = needed_wmnij.size();
  plan.required_intermediate_blocks.wmbej_blocks = needed_wmbej.size();
  plan.required_intermediate_blocks.wabef_blocks = needed_wabef.size();
  const auto append_intermediate =
      [&](std::vector<CcsdIntermediateMonomerBlock>& output,
          const std::array<Space, 4> spaces,
          const std::array<std::size_t, 4> labels) {
        CcsdIntermediateMonomerBlock block;
        block.spaces = spaces;
        block.monomers = labels;
        block.element_count = 1;
        for (std::size_t axis = 0; axis < 4; ++axis) {
          block.extents[axis] =
              block_extent(plan, spaces[axis], labels[axis]);
          block.element_count =
              checked_product(block.element_count, block.extents[axis]);
        }
        output.push_back(block);
      };
  for (std::size_t i = 0; i < monomers; ++i)
    for (std::size_t j = 0; j < monomers; ++j)
      for (std::size_t a = 0; a < monomers; ++a)
        for (std::size_t b = 0; b < monomers; ++b)
          if (support.tau(i, j, a, b))
            append_intermediate(plan.required_tau_blocks, {O, O, V, V},
                                {i, j, a, b});
  for (const auto& labels : needed_wmnij)
    append_intermediate(plan.required_wmnij_blocks, {O, O, O, O}, labels);
  for (const auto& labels : needed_wmbej)
    append_intermediate(plan.required_wmbej_blocks, {O, V, V, O}, labels);
  for (const auto& labels : needed_wabef)
    append_intermediate(plan.required_wabef_blocks, {V, V, V, V}, labels);

  for (const auto& labels : needed_wmnij) {
    const auto [m, n, i, j] = labels;
    require_integral({O, O, O, O}, labels, CcsdDependencyUse::wmnij);
    for (std::size_t e = 0; e < monomers; ++e) {
      if (support.singles(j, e))
        require_integral({O, O, O, V}, {m, n, i, e},
                         CcsdDependencyUse::wmnij);
      if (support.singles(i, e))
        require_integral({O, O, O, V}, {m, n, j, e},
                         CcsdDependencyUse::wmnij);
      for (std::size_t q = 0; q < monomers; ++q)
        if (support.tau(i, j, e, q))
          require_integral({O, O, V, V}, {m, n, e, q},
                           CcsdDependencyUse::wmnij);
    }
  }

  for (const auto& labels : needed_wmbej) {
    const auto [m, b, e, j] = labels;
    require_integral({O, V, V, O}, labels, CcsdDependencyUse::wmbej);
    for (std::size_t q = 0; q < monomers; ++q)
      if (support.singles(j, q))
        require_integral({O, V, V, V}, {m, b, e, q},
                         CcsdDependencyUse::wmbej);
    for (std::size_t n = 0; n < monomers; ++n) {
      if (support.singles(n, b))
        require_integral({O, O, V, O}, {m, n, e, j},
                         CcsdDependencyUse::wmbej);
      for (std::size_t q = 0; q < monomers; ++q)
        if (support.doubles(j, n, q, b) ||
            (support.singles(j, q) && support.singles(n, b)))
          require_integral({O, O, V, V}, {m, n, e, q},
                           CcsdDependencyUse::wmbej);
    }
  }

  for (const auto& labels : needed_wabef) {
    const auto [a, b, e, q] = labels;
    require_integral({V, V, V, V}, labels, CcsdDependencyUse::wabef);
    for (std::size_t m = 0; m < monomers; ++m) {
      if (support.singles(m, b))
        require_integral({V, O, V, V}, {a, m, e, q},
                         CcsdDependencyUse::wabef);
      if (support.singles(m, a))
        require_integral({V, O, V, V}, {b, m, e, q},
                         CcsdDependencyUse::wabef);
      for (std::size_t n = 0; n < monomers; ++n)
        if (support.tau(m, n, a, b))
          require_integral({O, O, V, V}, {m, n, e, q},
                           CcsdDependencyUse::wabef);
    }
  }

  // Direct residual and energy dependencies that are not represented by an
  // F/W intermediate. Only retained external blocks are traversed.
  for (const auto& external : plan.singles_blocks) {
    const auto i = external.occupied_monomer;
    const auto a = external.virtual_monomer;
    require_fock({O, V}, {i, a}, CcsdDependencyUse::singles_residual);
    for (std::size_t m = 0; m < monomers; ++m)
      for (std::size_t e = 0; e < monomers; ++e) {
        if (support.singles(m, e))
          require_integral({O, V, O, V}, {m, a, i, e},
                           CcsdDependencyUse::singles_residual);
        for (std::size_t q = 0; q < monomers; ++q)
          if (support.doubles(i, m, e, q))
            require_integral({O, V, V, V}, {m, a, e, q},
                             CcsdDependencyUse::singles_residual);
        for (std::size_t n = 0; n < monomers; ++n)
          if (support.doubles(m, n, a, e))
            require_integral({O, O, V, O}, {n, m, e, i},
                             CcsdDependencyUse::singles_residual);
      }
  }

  const auto require_doubles_direct =
      [&](const std::size_t i, const std::size_t j,
          const std::size_t a, const std::size_t b) {
        require_integral({O, O, V, V}, {i, j, a, b},
                         CcsdDependencyUse::doubles_residual);
        require_integral({O, O, V, V}, {i, j, a, b},
                         CcsdDependencyUse::energy);
        for (std::size_t e = 0; e < monomers; ++e) {
          if (support.singles(i, e))
            require_integral({V, V, V, O}, {a, b, e, j},
                             CcsdDependencyUse::doubles_residual);
          for (std::size_t m = 0; m < monomers; ++m)
            if (support.singles(i, e) && support.singles(m, a))
              require_integral({O, V, V, O}, {m, b, e, j},
                               CcsdDependencyUse::doubles_residual);
        }
        for (std::size_t m = 0; m < monomers; ++m)
          if (support.singles(m, a))
            require_integral({O, V, O, O}, {m, b, i, j},
                             CcsdDependencyUse::doubles_residual);
      };
  for (const auto& external : plan.doubles_blocks) {
    const auto [i, j, a, b] = external.monomers;
    require_doubles_direct(i, j, a, b);
    require_doubles_direct(j, i, a, b);
    require_doubles_direct(i, j, b, a);
    require_doubles_direct(j, i, b, a);
  }

  // The CCSD energy contains the disconnected T1*T1 contribution even when
  // the corresponding T2 external block is not retained.  Its ERI support
  // is therefore the union of the selected-doubles support and every pair of
  // selected singles blocks.  Omitting this closure would make a finite-span
  // projected model silently request an undeclared integral during energy
  // evaluation.
  for (const auto& left : plan.singles_blocks)
    for (const auto& right : plan.singles_blocks)
      require_integral(
          {O, O, V, V},
          {left.occupied_monomer, right.occupied_monomer,
           left.virtual_monomer, right.virtual_monomer},
          CcsdDependencyUse::energy);

  plan.required_fock_blocks.reserve(fock.size());
  for (const auto& [key, uses] : fock) {
    CcsdFockMonomerBlock block;
    block.spaces = key.spaces;
    block.monomers = key.monomers;
    block.uses = uses;
    block.element_count = 1;
    for (std::size_t axis = 0; axis < 2; ++axis) {
      block.extents[axis] =
          block_extent(plan, block.spaces[axis], block.monomers[axis]);
      block.element_count =
          checked_product(block.element_count, block.extents[axis]);
    }
    plan.required_fock_blocks.push_back(block);
  }
  plan.required_integral_blocks.reserve(integral.size());
  for (const auto& [key, uses] : integral) {
    CcsdIntegralMonomerBlock block;
    block.spaces = key.spaces;
    block.monomers = key.monomers;
    block.uses = uses;
    block.element_count = 1;
    for (std::size_t axis = 0; axis < 4; ++axis) {
      block.extents[axis] =
          block_extent(plan, block.spaces[axis], block.monomers[axis]);
      block.element_count =
          checked_product(block.element_count, block.extents[axis]);
    }
    plan.required_integral_blocks.push_back(block);
  }
  if (plan.required_fock_blocks.empty() ||
      plan.required_integral_blocks.empty() ||
      plan.required_intermediate_blocks.wmnij_blocks == 0 ||
      plan.required_intermediate_blocks.wmbej_blocks == 0 ||
      plan.required_intermediate_blocks.wabef_blocks == 0)
    throw std::logic_error(
        "selected CCSD equation dependency audit produced an empty closure");
  plan.dependency_status.integral_tile_closure_complete = true;
  plan.dependency_status.factorized_intermediate_closure_complete = true;
}

} // namespace

SelectedCcsdManifoldPlan plan_selected_ccsd_manifold(
    const std::vector<int>& occupied_labels,
    const std::vector<int>& virtual_labels,
    const SelectedCcsdManifoldOptions& options) {
  if (occupied_labels.empty() || virtual_labels.empty())
    throw std::invalid_argument(
        "selected CCSD manifold requires occupied and virtual orbitals");
  if (options.maximum_monomers == 0)
    throw std::invalid_argument(
        "selected CCSD maximum monomer span must be positive");

  SelectedCcsdManifoldPlan result;
  result.monomer_count = monomer_count(occupied_labels, virtual_labels);
  result.occupied_orbitals_by_monomer =
      group_orbitals(occupied_labels, result.monomer_count);
  result.virtual_orbitals_by_monomer =
      group_orbitals(virtual_labels, result.monomer_count);
  for (std::size_t monomer = 0; monomer < result.monomer_count; ++monomer) {
    if (result.occupied_orbitals_by_monomer[monomer].empty() ||
        result.virtual_orbitals_by_monomer[monomer].empty())
      throw std::invalid_argument(
          "selected CCSD manifold requires occupied and virtual orbitals "
          "on every labeled monomer");
  }

  const InteractionGraph graph(result.monomer_count, options);
  const auto occupied = occupied_labels.size();
  const auto virtuals = virtual_labels.size();
  result.full_singles_elements = checked_product(occupied, virtuals);
  result.full_doubles_elements =
      checked_product(checked_product(occupied, occupied),
                      checked_product(virtuals, virtuals));

  for (std::size_t hole = 0; hole < result.monomer_count; ++hole) {
    for (std::size_t particle = 0; particle < result.monomer_count;
         ++particle) {
      const std::array<std::size_t, 2> labels{hole, particle};
      if (!retained(participating_monomers(labels), options, graph))
        continue;
      const auto elements = checked_product(
          result.occupied_orbitals_by_monomer[hole].size(),
          result.virtual_orbitals_by_monomer[particle].size());
      result.singles_blocks.push_back(
          SinglesMonomerBlock{hole, particle,
                              result.occupied_orbitals_by_monomer[hole].size(),
                              result.virtual_orbitals_by_monomer[particle].size(),
                              elements});
      result.selected_singles_elements =
          checked_add(result.selected_singles_elements, elements);
    }
  }

  for (std::size_t first_hole = 0; first_hole < result.monomer_count;
       ++first_hole) {
    for (std::size_t second_hole = 0; second_hole < result.monomer_count;
         ++second_hole) {
      for (std::size_t first_particle = 0;
           first_particle < result.monomer_count; ++first_particle) {
        for (std::size_t second_particle = 0;
             second_particle < result.monomer_count; ++second_particle) {
          const std::array<std::size_t, 4> labels{
              first_hole, second_hole, first_particle, second_particle};
          const auto participating = participating_monomers(labels);
          if (!retained(participating, options, graph))
            continue;
          const std::array<std::size_t, 4> extents{
              result.occupied_orbitals_by_monomer[first_hole].size(),
              result.occupied_orbitals_by_monomer[second_hole].size(),
              result.virtual_orbitals_by_monomer[first_particle].size(),
              result.virtual_orbitals_by_monomer[second_particle].size()};
          auto elements = std::size_t{1};
          for (const auto extent : extents)
            elements = checked_product(elements, extent);
          result.doubles_blocks.push_back(DoublesMonomerBlock{
              labels, extents, participating.size(), elements});
          result.selected_doubles_elements =
              checked_add(result.selected_doubles_elements, elements);
        }
      }
    }
  }

  if (result.selected_singles_elements > result.full_singles_elements ||
      result.selected_doubles_elements > result.full_doubles_elements)
    throw std::logic_error(
        "selected CCSD manifold exceeds the complete amplitude space");

  result.dependency_status.projected_amplitude_manifold_closed = true;
  audit_ccsd_dependencies(result);
  return result;
}

CcsdProjectionMask make_selected_ccsd_projection_mask(
    const std::vector<int>& occupied_labels,
    const std::vector<int>& virtual_labels,
    const SelectedCcsdManifoldOptions& options) {
  const auto plan = plan_selected_ccsd_manifold(
      occupied_labels, virtual_labels, options);
  const auto occupied = occupied_labels.size();
  const auto virtuals = virtual_labels.size();
  CcsdProjectionMask mask;
  mask.singles.assign(plan.full_singles_elements, 0U);
  mask.doubles.assign(plan.full_doubles_elements, 0U);
  for (const auto& block : plan.singles_blocks)
    for (const auto i :
         plan.occupied_orbitals_by_monomer[block.occupied_monomer])
      for (const auto a :
           plan.virtual_orbitals_by_monomer[block.virtual_monomer])
        mask.singles[i * virtuals + a] = 1U;
  for (const auto& block : plan.doubles_blocks)
    for (const auto i : plan.occupied_orbitals_by_monomer[block.monomers[0]])
      for (const auto j :
           plan.occupied_orbitals_by_monomer[block.monomers[1]])
        for (const auto a :
             plan.virtual_orbitals_by_monomer[block.monomers[2]])
          for (const auto b :
               plan.virtual_orbitals_by_monomer[block.monomers[3]]) {
            const auto flat =
                ((i * occupied + j) * virtuals + a) * virtuals + b;
            mask.doubles[flat] = 1U;
          }
  mask.selected_singles = static_cast<std::size_t>(
      std::count(mask.singles.begin(), mask.singles.end(), 1U));
  mask.selected_doubles = static_cast<std::size_t>(
      std::count(mask.doubles.begin(), mask.doubles.end(), 1U));
  if (mask.selected_singles != plan.selected_singles_elements ||
      mask.selected_doubles != plan.selected_doubles_elements)
    throw std::logic_error(
        "selected CCSD dense projection disagrees with the block plan");
  return mask;
}

} // namespace localized_cc
