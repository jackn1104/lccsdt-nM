#include "localized_cc/selected_ccsd_storage.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace localized_cc {
namespace {

constexpr std::array<std::array<std::size_t, 4>, 8>
    chemists_eri_permutations{{
        {0, 1, 2, 3},
        {1, 0, 2, 3},
        {0, 1, 3, 2},
        {1, 0, 3, 2},
        {2, 3, 0, 1},
        {3, 2, 0, 1},
        {2, 3, 1, 0},
        {3, 2, 1, 0},
    }};

std::size_t orbital_space_token(const CcsdOrbitalSpace space,
                                const std::size_t monomer,
                                const std::size_t monomers) {
  if (monomer >= monomers)
    throw std::out_of_range(
        "selected integral monomer label is out of range");
  switch (space) {
  case CcsdOrbitalSpace::occupied:
    return monomer;
  case CcsdOrbitalSpace::virtual_space:
    if (monomers > std::numeric_limits<std::size_t>::max() - monomer)
      throw std::overflow_error("selected integral orbital token overflows");
    return monomers + monomer;
  }
  throw std::invalid_argument("selected integral orbital space is invalid");
}

std::array<std::size_t, 4> descriptor_tokens(
    const CcsdIntegralMonomerBlock& descriptor,
    const std::size_t monomers) {
  std::array<std::size_t, 4> result{};
  for (std::size_t axis = 0; axis < 4; ++axis)
    result[axis] = orbital_space_token(
        descriptor.spaces[axis], descriptor.monomers[axis], monomers);
  return result;
}

constexpr std::size_t canonical_eri_permutation_index(
    const std::array<std::size_t, 4>& tokens) {
  std::size_t best_index = 0;
  auto best = chemists_eri_permutations[best_index];
  std::array<std::size_t, 4> best_tokens{
      tokens[best[0]], tokens[best[1]], tokens[best[2]], tokens[best[3]]};
  for (std::size_t index = 1; index < chemists_eri_permutations.size();
       ++index) {
    const auto& permutation = chemists_eri_permutations[index];
    const std::array<std::size_t, 4> candidate{
        tokens[permutation[0]], tokens[permutation[1]],
        tokens[permutation[2]], tokens[permutation[3]]};
    if (candidate < best_tokens) {
      best_index = index;
      best = permutation;
      best_tokens = candidate;
    }
  }
  return best_index;
}

constexpr std::array<std::uint8_t, 16>
canonical_space_permutation_indices() {
  std::array<std::uint8_t, 16> result{};
  for (std::size_t pattern = 0; pattern < result.size(); ++pattern) {
    std::array<std::size_t, 4> tokens{};
    for (std::size_t axis = 0; axis < 4; ++axis)
      tokens[axis] = (pattern >> (3U - axis)) & 1U;
    result[pattern] = static_cast<std::uint8_t>(
        canonical_eri_permutation_index(tokens));
  }
  return result;
}

constexpr auto canonical_space_permutation_by_pattern =
    canonical_space_permutation_indices();

std::array<std::size_t, 4> canonical_eri_permutation(
    const std::array<std::size_t, 4>& tokens) {
  return chemists_eri_permutations[canonical_eri_permutation_index(tokens)];
}

CcsdIntegralMonomerBlock canonical_spatial_descriptor(
    const CcsdIntegralMonomerBlock& descriptor,
    const std::size_t monomers) {
  const auto permutation =
      canonical_eri_permutation(descriptor_tokens(descriptor, monomers));
  CcsdIntegralMonomerBlock result;
  result.uses = descriptor.uses;
  result.element_count = descriptor.element_count;
  for (std::size_t axis = 0; axis < 4; ++axis) {
    const auto source = permutation[axis];
    result.spaces[axis] = descriptor.spaces[source];
    result.monomers[axis] = descriptor.monomers[source];
    result.extents[axis] = descriptor.extents[source];
  }
  return result;
}

std::size_t checked_product(const std::size_t left,
                            const std::size_t right,
                            const char* message) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(message);
  return left * right;
}

std::vector<std::size_t> block_sizes(
    const std::vector<DoublesMonomerBlock>& blocks) {
  std::vector<std::size_t> result;
  result.reserve(blocks.size());
  for (const auto& block : blocks)
    result.push_back(block.element_count);
  return result;
}

std::vector<std::size_t> intermediate_block_sizes(
    const std::vector<CcsdIntermediateMonomerBlock>& blocks) {
  std::vector<std::size_t> result;
  result.reserve(blocks.size());
  for (const auto& block : blocks)
    result.push_back(block.element_count);
  return result;
}

std::vector<std::vector<std::size_t>> collapse_spin_groups(
    const std::vector<std::vector<std::size_t>>& spin_groups,
    const std::size_t spin_extent, const char* space_name) {
  if (spin_extent == 0 || spin_extent % 2 != 0)
    throw std::invalid_argument(
        std::string("selected integral ") + space_name +
        " spin extent must contain complete alpha/beta pairs");
  std::vector<std::size_t> monomer_by_spin(spin_extent,
                                           spin_groups.size());
  for (std::size_t monomer = 0; monomer < spin_groups.size(); ++monomer)
    for (const auto orbital : spin_groups[monomer]) {
      if (orbital >= spin_extent ||
          monomer_by_spin[orbital] != spin_groups.size())
        throw std::invalid_argument(
            std::string("selected integral ") + space_name +
            " spin-orbital groups are invalid");
      monomer_by_spin[orbital] = monomer;
    }
  std::vector<std::vector<std::size_t>> result(spin_groups.size());
  for (std::size_t spatial = 0; spatial < spin_extent / 2; ++spatial) {
    const auto alpha = 2 * spatial;
    const auto beta = alpha + 1;
    if (monomer_by_spin[alpha] == spin_groups.size() ||
        monomer_by_spin[alpha] != monomer_by_spin[beta])
      throw std::invalid_argument(
          std::string("selected integral ") + space_name +
          " alpha/beta partners must have the same monomer label");
    result[monomer_by_spin[alpha]].push_back(spatial);
  }
  return result;
}

std::vector<std::vector<double>> dense_oracle_blocks(
    const std::vector<CcsdIntegralMonomerBlock>& descriptors,
    const std::vector<std::vector<std::size_t>>& occupied_groups,
    const std::vector<std::vector<std::size_t>>& virtual_groups,
    const SpatialRhfHamiltonian& dense_oracle) {
  std::vector<std::vector<double>> result;
  result.reserve(descriptors.size());
  for (const auto& descriptor : descriptors) {
    std::vector<double> values(descriptor.element_count);
    for (std::size_t flat = 0; flat < descriptor.element_count; ++flat) {
      auto remainder = flat;
      std::array<std::size_t, 4> orbitals{};
      for (std::size_t reversed = 0; reversed < 4; ++reversed) {
        const auto axis = 3 - reversed;
        const auto& groups =
            descriptor.spaces[axis] == CcsdOrbitalSpace::occupied
                ? occupied_groups
                : virtual_groups;
        const auto& group = groups.at(descriptor.monomers[axis]);
        const auto relative = group.at(remainder % group.size());
        remainder /= group.size();
        orbitals[axis] =
            descriptor.spaces[axis] == CcsdOrbitalSpace::occupied
                ? relative
                : dense_oracle.nocc + relative;
      }
      if (remainder != 0)
        throw std::logic_error(
            "selected spatial integral block offset did not fully decode");
      values[flat] = dense_oracle.eri_chemist(
          orbitals[0], orbitals[1], orbitals[2], orbitals[3]);
    }
    result.push_back(std::move(values));
  }
  return result;
}

} // namespace

SelectedIntegralStore::SelectedIntegralStore(
    const SelectedCcsdManifoldPlan& plan,
    const SpatialRhfHamiltonian& dense_oracle)
    : SelectedIntegralStore(
          plan, dense_oracle.nocc, dense_oracle.nvirt,
          [&dense_oracle](const auto& descriptors,
                          const auto& occupied_groups,
                          const auto& virtual_groups) {
            return dense_oracle_blocks(descriptors, occupied_groups,
                                       virtual_groups, dense_oracle);
          }) {}

SelectedIntegralStore::SelectedIntegralStore(
    const SelectedCcsdManifoldPlan& plan,
    const std::size_t spatial_occupied,
    const std::size_t spatial_virtuals,
    const SelectedSpatialIntegralBlockProducer& producer)
    : monomer_count_(plan.monomer_count),
      occupied_(checked_product(
          2, spatial_occupied,
          "selected integral occupied spin-orbital count overflows")),
      virtuals_(checked_product(
          2, spatial_virtuals,
          "selected integral virtual spin-orbital count overflows")),
      spatial_occupied_(spatial_occupied),
      spatial_virtuals_(spatial_virtuals) {
  if (!plan.dependency_status.ready_for_sparse_ccsd() ||
      monomer_count_ == 0 || plan.required_integral_blocks.empty() ||
      !producer)
    throw std::invalid_argument(
        "selected integral storage requires a closed nonempty plan");
  std::size_t planned_occupied = 0;
  std::size_t planned_virtuals = 0;
  for (const auto& group : plan.occupied_orbitals_by_monomer)
    planned_occupied += group.size();
  for (const auto& group : plan.virtual_orbitals_by_monomer)
    planned_virtuals += group.size();
  if (planned_occupied != occupied_ || planned_virtuals != virtuals_)
    throw std::invalid_argument(
        "selected integral orbital spaces differ from the dense oracle");
  if (occupied_ > std::numeric_limits<std::size_t>::max() - virtuals_)
    throw std::overflow_error("selected integral orbital count overflows");

  spin_orbitals_.assign(occupied_ + virtuals_,
                        OrbitalAddress{CcsdOrbitalSpace::occupied, missing,
                                       missing, missing});
  const auto assign_space = [&](const CcsdOrbitalSpace space,
                                const auto& groups,
                                const std::size_t offset,
                                const std::size_t extent,
                                auto& orbitals) {
    for (std::size_t monomer = 0; monomer < groups.size(); ++monomer)
      for (std::size_t local = 0; local < groups[monomer].size(); ++local) {
        const auto relative = groups[monomer][local];
        if (relative >= extent ||
            orbitals[offset + relative].monomer != missing)
          throw std::invalid_argument(
              "selected integral orbital map is invalid");
        orbitals[offset + relative] = OrbitalAddress{
            space, monomer, local,
            orbital_space_token(space, monomer, monomer_count_)};
      }
  };
  assign_space(CcsdOrbitalSpace::occupied,
               plan.occupied_orbitals_by_monomer, 0, occupied_,
               spin_orbitals_);
  assign_space(CcsdOrbitalSpace::virtual_space,
               plan.virtual_orbitals_by_monomer, occupied_, virtuals_,
               spin_orbitals_);
  for (const auto& address : spin_orbitals_)
    if (address.monomer == missing || address.local == missing)
      throw std::invalid_argument(
          "selected integral orbital map is incomplete");

  const auto spatial_occupied_groups = collapse_spin_groups(
      plan.occupied_orbitals_by_monomer, occupied_, "occupied");
  const auto spatial_virtual_groups = collapse_spin_groups(
      plan.virtual_orbitals_by_monomer, virtuals_, "virtual");
  spatial_orbitals_.assign(
      spatial_occupied_ + spatial_virtuals_,
      OrbitalAddress{CcsdOrbitalSpace::occupied, missing, missing, missing});
  assign_space(CcsdOrbitalSpace::occupied, spatial_occupied_groups, 0,
               spatial_occupied_, spatial_orbitals_);
  assign_space(CcsdOrbitalSpace::virtual_space, spatial_virtual_groups,
               spatial_occupied_, spatial_virtuals_, spatial_orbitals_);
  for (const auto& address : spatial_orbitals_)
    if (address.monomer == missing || address.local == missing)
      throw std::invalid_argument(
          "selected spatial integral orbital map is incomplete");
  spin_to_spatial_.resize(spin_orbitals_.size());
  for (std::size_t orbital = 0; orbital < spin_orbitals_.size(); ++orbital)
    spin_to_spatial_[orbital] = spatial_orbital(orbital);

  spin_block_by_tuple_.reserve(plan.required_integral_blocks.size());
  spatial_block_by_tuple_.reserve(2 * plan.required_integral_blocks.size());
  const auto add_spatial_component =
      [&](const CcsdIntegralMonomerBlock& spin_descriptor,
          const std::array<std::size_t, 4>& order) {
        ++spatial_component_requests_;
        CcsdIntegralMonomerBlock spatial_descriptor;
        spatial_descriptor.uses = spin_descriptor.uses;
        spatial_descriptor.element_count = 1;
        for (std::size_t axis = 0; axis < 4; ++axis) {
          const auto source = order[axis];
          spatial_descriptor.spaces[axis] = spin_descriptor.spaces[source];
          spatial_descriptor.monomers[axis] =
              spin_descriptor.monomers[source];
          const auto& groups =
              spatial_descriptor.spaces[axis] ==
                      CcsdOrbitalSpace::occupied
                  ? spatial_occupied_groups
                  : spatial_virtual_groups;
          spatial_descriptor.extents[axis] =
              groups.at(spatial_descriptor.monomers[axis]).size();
          spatial_descriptor.element_count = checked_product(
              spatial_descriptor.element_count,
              spatial_descriptor.extents[axis],
              "selected spatial integral block size overflows");
        }
        spatial_descriptor = canonical_spatial_descriptor(
            spatial_descriptor, monomer_count_);
        const auto key = tuple_key(spatial_descriptor.spaces,
                                   spatial_descriptor.monomers);
        const auto found = spatial_block_by_tuple_.find(key);
        if (found != spatial_block_by_tuple_.end()) {
          spatial_descriptors_.at(found->second).uses |=
              spin_descriptor.uses;
          return;
        }
        const auto index = spatial_descriptors_.size();
        spatial_block_by_tuple_.emplace(key, index);
        spatial_descriptors_.push_back(spatial_descriptor);
      };

  for (const auto& descriptor : plan.required_integral_blocks) {
    const auto key = tuple_key(descriptor.spaces, descriptor.monomers);
    if (!spin_block_by_tuple_
             .emplace(key, spin_block_by_tuple_.size())
             .second)
      throw std::invalid_argument(
          "selected integral dependency closure contains a duplicate block");
    std::size_t expected = 1;
    for (std::size_t axis = 0; axis < 4; ++axis) {
      const auto& groups =
          descriptor.spaces[axis] == CcsdOrbitalSpace::occupied
              ? plan.occupied_orbitals_by_monomer
              : plan.virtual_orbitals_by_monomer;
      const auto extent = groups.at(descriptor.monomers[axis]).size();
      if (descriptor.extents[axis] != extent)
        throw std::invalid_argument(
            "selected integral block extent disagrees with its orbital group");
      expected = checked_product(
          expected, extent,
          "selected integral block element count overflows");
    }
    if (expected != descriptor.element_count)
      throw std::invalid_argument(
          "selected integral block element count is invalid");
    if (expected > std::numeric_limits<std::size_t>::max() -
                       declared_spin_elements_)
      throw std::overflow_error(
          "selected integral declared spin element count overflows");
    declared_spin_elements_ += expected;
    add_spatial_component(descriptor, {0, 2, 1, 3});
    add_spatial_component(descriptor, {0, 3, 1, 2});
  }
  declared_spin_blocks_ = spin_block_by_tuple_.size();

  blocks_ = producer(spatial_descriptors_, spatial_occupied_groups,
                     spatial_virtual_groups);
  if (blocks_.size() != spatial_descriptors_.size())
    throw std::invalid_argument(
        "selected integral producer returned the wrong block count");
  for (std::size_t block = 0; block < spatial_descriptors_.size(); ++block) {
    const auto& descriptor = spatial_descriptors_[block];
    if (blocks_[block].size() != descriptor.element_count)
      throw std::invalid_argument(
          "selected integral producer returned a block with the wrong "
          "extent");
    if (descriptor.element_count >
        std::numeric_limits<std::size_t>::max() - selected_elements_)
      throw std::overflow_error(
          "selected spatial integral total element count overflows");
    selected_elements_ += descriptor.element_count;
  }
  if (selected_elements_ >
      std::numeric_limits<std::size_t>::max() / sizeof(double))
    throw std::overflow_error(
        "selected spatial integral total byte count overflows");
  if (!all_finite())
    throw std::invalid_argument(
        "selected integral storage contains a non-finite value");

  // Build hot-path tuple metadata only after the producer releases its AO/MO
  // transform workspaces. This keeps the tuple table out of the transform
  // peak while preserving the constructor's fail-closed hash maps as the
  // fallback for larger monomer counts.
  const auto base = checked_product(
      2, monomer_count_, "selected integral tuple base overflows");
  const auto tuple_count = checked_product(
      checked_product(base, base,
                      "selected integral dense tuple index overflows"),
      checked_product(base, base,
                      "selected integral dense tuple index overflows"),
      "selected integral dense tuple index overflows");
  constexpr std::size_t maximum_dense_tuple_index_bytes =
      64U * 1024U * 1024U;
  constexpr std::size_t dense_tuple_bytes =
      2 * sizeof(std::size_t) + sizeof(std::uint8_t);
  if (tuple_count <= maximum_dense_tuple_index_bytes / dense_tuple_bytes) {
    spin_block_by_tuple_dense_.assign(tuple_count, missing);
    spatial_block_by_tuple_dense_.assign(tuple_count, missing);
    canonical_permutation_by_tuple_.resize(tuple_count);
    for (const auto& [key, block] : spin_block_by_tuple_)
      spin_block_by_tuple_dense_.at(key) = block;
    for (const auto& [key, block] : spatial_block_by_tuple_)
      spatial_block_by_tuple_dense_.at(key) = block;
    for (std::size_t key = 0; key < tuple_count; ++key) {
      auto remainder = key;
      std::array<std::size_t, 4> tokens{};
      for (std::size_t reversed = 0; reversed < 4; ++reversed) {
        const auto axis = 3 - reversed;
        tokens[axis] = remainder % base;
        remainder /= base;
      }
      canonical_permutation_by_tuple_[key] = static_cast<std::uint8_t>(
          canonical_eri_permutation_index(tokens));
    }
    spin_block_by_tuple_.clear();
    spin_block_by_tuple_.rehash(0);
    spatial_block_by_tuple_.clear();
    spatial_block_by_tuple_.rehash(0);
  }
}

std::array<std::size_t, 4>
SelectedIntegralStore::dimensions() const noexcept {
  const auto orbitals = occupied_ + virtuals_;
  return {orbitals, orbitals, orbitals, orbitals};
}

std::size_t
SelectedIntegralStore::declared_spin_block_count() const noexcept {
  return declared_spin_blocks_;
}

std::size_t
SelectedIntegralStore::declared_spin_element_count() const noexcept {
  return declared_spin_elements_;
}

std::size_t SelectedIntegralStore::block_count() const noexcept {
  return blocks_.size();
}

std::size_t
SelectedIntegralStore::spatial_component_request_count() const noexcept {
  return spatial_component_requests_;
}

std::size_t SelectedIntegralStore::selected_element_count() const noexcept {
  return selected_elements_;
}

std::size_t SelectedIntegralStore::selected_bytes() const noexcept {
  return selected_elements_ * sizeof(double);
}

std::size_t SelectedIntegralStore::tuple_lookup_bytes() const noexcept {
  return (spin_block_by_tuple_dense_.size() +
          spatial_block_by_tuple_dense_.size()) *
             sizeof(std::size_t) +
         canonical_permutation_by_tuple_.size() * sizeof(std::uint8_t);
}

bool SelectedIntegralStore::contains(const std::size_t p,
                                     const std::size_t q,
                                     const std::size_t r,
                                     const std::size_t s) const {
  const auto resolved = spin_addresses(p, q, r, s);
  return spin_block_lookup(address_tuple_key(resolved)) != missing;
}

std::array<std::size_t, 2>
SelectedIntegralStore::spatial_dimensions() const noexcept {
  return {spatial_occupied_, spatial_virtuals_};
}

bool SelectedIntegralStore::contains_spatial_chemist(
    const std::size_t p, const std::size_t q, const std::size_t r,
    const std::size_t s) const {
  const auto resolved =
      canonical_spatial_addresses(spatial_addresses({p, q, r, s}));
  return spatial_block_lookup(address_tuple_key(resolved)) != missing;
}

double SelectedIntegralStore::spatial_chemist(
    const std::size_t p, const std::size_t q, const std::size_t r,
    const std::size_t s) const {
  return spatial_value({p, q, r, s});
}

double SelectedIntegralStore::operator()(const std::size_t p,
                                         const std::size_t q,
                                         const std::size_t r,
                                         const std::size_t s) const {
  if (p >= spin_orbitals_.size() || q >= spin_orbitals_.size() ||
      r >= spin_orbitals_.size() || s >= spin_orbitals_.size())
    throw std::out_of_range(
        "selected integral spin orbital is out of range");
  const std::array<OrbitalAddress, 4> resolved{
      spin_orbitals_[p], spin_orbitals_[q], spin_orbitals_[r],
      spin_orbitals_[s]};
  static_cast<void>(spin_block_index(resolved));
  const auto& spatial_p = spatial_orbitals_[spin_to_spatial_[p]];
  const auto& spatial_q = spatial_orbitals_[spin_to_spatial_[q]];
  const auto& spatial_r = spatial_orbitals_[spin_to_spatial_[r]];
  const auto& spatial_s = spatial_orbitals_[spin_to_spatial_[s]];
  double value = 0.0;
  if (p % 2 == r % 2 && q % 2 == s % 2)
    value += spatial_value_unchecked(
        {spatial_p, spatial_r, spatial_q, spatial_s});
  if (p % 2 == s % 2 && q % 2 == r % 2)
    value -= spatial_value_unchecked(
        {spatial_p, spatial_s, spatial_q, spatial_r});
  return value;
}

bool SelectedIntegralStore::all_finite() const noexcept {
  for (const auto& block : blocks_)
    for (const auto value : block)
      if (!std::isfinite(value)) return false;
  return true;
}

std::size_t SelectedIntegralStore::tuple_key(
    const std::array<CcsdOrbitalSpace, 4>& spaces,
    const std::array<std::size_t, 4>& monomers) const {
  if (monomer_count_ > std::numeric_limits<std::size_t>::max() / 2)
    throw std::overflow_error("selected integral tuple base overflows");
  const auto base = 2 * monomer_count_;
  std::size_t result = 0;
  for (std::size_t axis = 0; axis < 4; ++axis) {
    if (monomers[axis] >= monomer_count_)
      throw std::out_of_range(
          "selected integral monomer label is out of range");
    std::size_t space = 0;
    switch (spaces[axis]) {
    case CcsdOrbitalSpace::occupied:
      space = 0;
      break;
    case CcsdOrbitalSpace::virtual_space:
      space = 1;
      break;
    default:
      throw std::invalid_argument(
          "selected integral orbital space is invalid");
    }
    const auto token = space * monomer_count_ + monomers[axis];
    result = checked_product(result, base,
                             "selected integral tuple key overflows");
    if (token > std::numeric_limits<std::size_t>::max() - result)
      throw std::overflow_error("selected integral tuple key overflows");
    result += token;
  }
  return result;
}

std::array<SelectedIntegralStore::OrbitalAddress, 4>
SelectedIntegralStore::spin_addresses(const std::size_t p,
                                      const std::size_t q,
                                      const std::size_t r,
                                      const std::size_t s) const {
  const std::array<std::size_t, 4> indices{p, q, r, s};
  std::array<OrbitalAddress, 4> result{};
  for (std::size_t axis = 0; axis < 4; ++axis) {
    if (indices[axis] >= spin_orbitals_.size())
      throw std::out_of_range(
          "selected integral spin orbital is out of range");
    result[axis] = spin_orbitals_[indices[axis]];
  }
  return result;
}

std::array<SelectedIntegralStore::OrbitalAddress, 4>
SelectedIntegralStore::spatial_addresses(
    const std::array<std::size_t, 4>& orbitals) const {
  std::array<OrbitalAddress, 4> result{};
  for (std::size_t axis = 0; axis < 4; ++axis) {
    if (orbitals[axis] >= spatial_orbitals_.size())
      throw std::out_of_range(
          "selected integral spatial orbital is out of range");
    result[axis] = spatial_orbitals_[orbitals[axis]];
  }
  return result;
}

std::array<SelectedIntegralStore::OrbitalAddress, 4>
SelectedIntegralStore::canonical_spatial_addresses(
    const std::array<OrbitalAddress, 4>& addresses) const {
  std::array<std::size_t, 4> permutation{};
  if (!canonical_permutation_by_tuple_.empty()) {
    const auto key = address_tuple_key(addresses);
    if (key >= canonical_permutation_by_tuple_.size())
      throw std::out_of_range("selected integral spatial tuple is invalid");
    permutation = chemists_eri_permutations[
        canonical_permutation_by_tuple_[key]];
  } else {
    std::array<std::size_t, 4> tokens{};
    for (std::size_t axis = 0; axis < 4; ++axis)
      tokens[axis] = addresses[axis].token;
    permutation = canonical_eri_permutation(tokens);
  }
  std::array<OrbitalAddress, 4> result{};
  for (std::size_t axis = 0; axis < 4; ++axis)
    result[axis] = addresses[permutation[axis]];
  return result;
}

std::size_t SelectedIntegralStore::address_tuple_key(
    const std::array<OrbitalAddress, 4>& addresses) const {
  const auto base = 2 * monomer_count_;
  std::size_t result = addresses[0].token;
  for (std::size_t axis = 1; axis < 4; ++axis)
    result = result * base + addresses[axis].token;
  return result;
}

std::size_t
SelectedIntegralStore::spin_block_lookup(const std::size_t key) const {
  if (!spin_block_by_tuple_dense_.empty()) {
    if (key >= spin_block_by_tuple_dense_.size())
      throw std::out_of_range("selected integral spin tuple is invalid");
    return spin_block_by_tuple_dense_[key];
  }
  const auto found = spin_block_by_tuple_.find(key);
  return found == spin_block_by_tuple_.end() ? missing : found->second;
}

std::size_t
SelectedIntegralStore::spatial_block_lookup(const std::size_t key) const {
  if (!spatial_block_by_tuple_dense_.empty()) {
    if (key >= spatial_block_by_tuple_dense_.size())
      throw std::out_of_range("selected integral spatial tuple is invalid");
    return spatial_block_by_tuple_dense_[key];
  }
  const auto found = spatial_block_by_tuple_.find(key);
  return found == spatial_block_by_tuple_.end() ? missing : found->second;
}

std::size_t SelectedIntegralStore::spin_block_index(
    const std::array<OrbitalAddress, 4>& resolved) const {
  const auto block = spin_block_lookup(address_tuple_key(resolved));
  if (block == missing)
    throw std::logic_error(
        "selected CCSD requested an undeclared ERI block");
  return block;
}

std::size_t SelectedIntegralStore::spatial_block_index(
    const std::array<OrbitalAddress, 4>& resolved) const {
  const auto block = spatial_block_lookup(address_tuple_key(resolved));
  if (block == missing)
    throw std::logic_error(
        "selected CCSD spatial ERI component is outside the declared "
        "dependency closure");
  return block;
}

std::size_t SelectedIntegralStore::spatial_element_index(
    const std::size_t block,
    const std::array<OrbitalAddress, 4>& resolved) const {
  const auto& descriptor = spatial_descriptors_.at(block);
  if (resolved[0].local >= descriptor.extents[0])
    throw std::logic_error(
        "selected integral local orbital is outside its block");
  std::size_t result = resolved[0].local;
  for (std::size_t axis = 1; axis < 4; ++axis) {
    if (resolved[axis].local >= descriptor.extents[axis])
      throw std::logic_error(
          "selected integral local orbital is outside its block");
    result = result * descriptor.extents[axis] + resolved[axis].local;
  }
  if (result >= descriptor.element_count)
    throw std::logic_error("selected integral local offset is invalid");
  return result;
}

std::size_t SelectedIntegralStore::spatial_element_index_unchecked(
    const std::size_t block,
    const std::array<OrbitalAddress, 4>& resolved) const noexcept {
  const auto& extents = spatial_descriptors_[block].extents;
  return ((resolved[0].local * extents[1] + resolved[1].local) *
              extents[2] +
          resolved[2].local) *
             extents[3] +
         resolved[3].local;
}

std::size_t SelectedIntegralStore::spatial_orbital(
    const std::size_t spin_orbital) const {
  if (spin_orbital < occupied_)
    return spin_orbital / 2;
  if (spin_orbital >= occupied_ + virtuals_)
    throw std::out_of_range(
        "selected integral spin orbital is out of range");
  return spatial_occupied_ + (spin_orbital - occupied_) / 2;
}

double SelectedIntegralStore::spatial_value(
    const std::array<std::size_t, 4>& orbitals) const {
  const auto resolved =
      canonical_spatial_addresses(spatial_addresses(orbitals));
  const auto block = spatial_block_index(resolved);
  return blocks_[block][spatial_element_index(block, resolved)];
}

double SelectedIntegralStore::spatial_value_unchecked(
    const std::array<OrbitalAddress, 4>& addresses) const {
  const auto resolved = canonical_spatial_addresses(addresses);
  const auto block = spatial_block_lookup(address_tuple_key(resolved));
  if (block == missing)
    throw std::logic_error(
        "selected CCSD spatial ERI component is outside the declared "
        "dependency closure");
  return blocks_[block][spatial_element_index_unchecked(block, resolved)];
}

SelectedIntegralPanelCache::SelectedIntegralPanelCache(
    const SelectedIntegralStore& source, const std::size_t maximum_bytes)
    : source_(&source), spatial_occupied_(source.spatial_dimensions()[0]),
      spatial_virtuals_(source.spatial_dimensions()[1]),
      spin_occupied_(checked_product(
          2, spatial_occupied_,
          "selected integral panel spin occupied count overflows")),
      spin_virtuals_(checked_product(
          2, spatial_virtuals_,
          "selected integral panel spin virtual count overflows")) {
  const auto make_panel = [&](const std::array<unsigned char, 4>& spaces) {
    Panel panel;
    panel.spaces = spaces;
    std::size_t elements = 1;
    for (std::size_t axis = 0; axis < 4; ++axis) {
      panel.extents[axis] = spaces[axis] == 0U ? spatial_occupied_
                                               : spatial_virtuals_;
      elements = checked_product(
          elements, panel.extents[axis],
          "selected integral panel extent overflows");
    }
    const auto bytes_per_element = sizeof(double) + sizeof(std::uint8_t);
    const auto bytes = checked_product(
        elements, bytes_per_element,
        "selected integral panel byte count overflows");
    if (bytes > std::numeric_limits<std::size_t>::max() - required_bytes_)
      throw std::overflow_error(
          "selected integral panel-cache byte count overflows");
    required_bytes_ += bytes;
    return panel;
  };
  oooo_ = make_panel({0U, 0U, 0U, 0U});
  ooov_ = make_panel({0U, 0U, 0U, 1U});
  oovv_ = make_panel({0U, 0U, 1U, 1U});
  ovov_ = make_panel({0U, 1U, 0U, 1U});
  ovvv_ = make_panel({0U, 1U, 1U, 1U});
  if (maximum_bytes == 0 || required_bytes_ > maximum_bytes) return;

  const auto fill = [&](Panel& panel) {
    std::size_t elements = 1;
    for (const auto extent : panel.extents)
      elements = checked_product(
          elements, extent,
          "selected integral panel extent overflows");
    panel.values.resize(elements);
    panel.present.resize(elements);
    std::size_t populated = 0;
    if (elements >
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
      throw std::overflow_error(
          "selected integral panel extent exceeds ptrdiff_t");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : populated)
#endif
    for (std::ptrdiff_t flat = 0;
         flat < static_cast<std::ptrdiff_t>(elements); ++flat) {
      auto remainder = static_cast<std::size_t>(flat);
      std::array<std::size_t, 4> orbitals{};
      for (std::size_t reversed = 0; reversed < 4; ++reversed) {
        const auto axis = 3 - reversed;
        const auto local = remainder % panel.extents[axis];
        remainder /= panel.extents[axis];
        orbitals[axis] = panel.spaces[axis] == 0U
                             ? local
                             : spatial_occupied_ + local;
      }
      if (source.contains_spatial_chemist(
              orbitals[0], orbitals[1], orbitals[2], orbitals[3])) {
        panel.values[static_cast<std::size_t>(flat)] =
            source.spatial_chemist(orbitals[0], orbitals[1], orbitals[2],
                                   orbitals[3]);
        panel.present[static_cast<std::size_t>(flat)] = 1U;
        ++populated;
      }
    }
    if (populated >
        std::numeric_limits<std::size_t>::max() - populated_elements_)
      throw std::overflow_error(
          "selected integral panel populated count overflows");
    populated_elements_ += populated;
  };
  fill(oooo_);
  fill(ooov_);
  fill(oovv_);
  fill(ovov_);
  fill(ovvv_);
  allocated_bytes_ = required_bytes_;
  enabled_ = true;
}

bool SelectedIntegralPanelCache::enabled() const noexcept {
  return enabled_;
}

std::size_t SelectedIntegralPanelCache::required_bytes() const noexcept {
  return required_bytes_;
}

std::size_t SelectedIntegralPanelCache::allocated_bytes() const noexcept {
  return allocated_bytes_;
}

std::size_t
SelectedIntegralPanelCache::populated_elements() const noexcept {
  return populated_elements_;
}

std::size_t SelectedIntegralPanelCache::spatial_orbital(
    const std::size_t spin_orbital) const {
  if (spin_orbital < spin_occupied_) return spin_orbital / 2;
  if (spin_orbital >= spin_occupied_ + spin_virtuals_)
    throw std::out_of_range(
        "selected integral panel-cache spin orbital is out of range");
  return spatial_occupied_ + (spin_orbital - spin_occupied_) / 2;
}

double SelectedIntegralPanelCache::spatial_value(
    const std::array<std::size_t, 4>& orbitals) const {
  std::array<unsigned char, 4> spaces{};
  unsigned int input_pattern = 0;
  for (std::size_t axis = 0; axis < 4; ++axis) {
    if (orbitals[axis] >= spatial_occupied_ + spatial_virtuals_)
      throw std::out_of_range(
          "selected integral panel-cache spatial orbital is out of range");
    spaces[axis] = orbitals[axis] < spatial_occupied_ ? 0U : 1U;
    input_pattern = 2U * input_pattern + spaces[axis];
  }
  const auto& permutation = chemists_eri_permutations[
      canonical_space_permutation_by_pattern[input_pattern]];
  std::array<std::size_t, 4> resolved{};
  std::array<unsigned char, 4> resolved_spaces{};
  unsigned int pattern = 0;
  for (std::size_t axis = 0; axis < 4; ++axis) {
    resolved[axis] = orbitals[permutation[axis]];
    resolved_spaces[axis] = spaces[permutation[axis]];
    pattern = 2U * pattern + resolved_spaces[axis];
  }
  const Panel* panel = nullptr;
  switch (pattern) {
  case 0U:
    panel = &oooo_;
    break;
  case 1U:
    panel = &ooov_;
    break;
  case 3U:
    panel = &oovv_;
    break;
  case 5U:
    panel = &ovov_;
    break;
  case 7U:
    panel = &ovvv_;
    break;
  case 15U:
    return source_->spatial_chemist(
        resolved[0], resolved[1], resolved[2], resolved[3]);
  default:
    throw std::logic_error(
        "selected integral panel-cache canonical space pattern is invalid");
  }
  std::size_t flat = 0;
  for (std::size_t axis = 0; axis < 4; ++axis) {
    const auto local = resolved_spaces[axis] == 0U
                           ? resolved[axis]
                           : resolved[axis] - spatial_occupied_;
    if (local >= panel->extents[axis])
      throw std::logic_error(
          "selected integral panel-cache local orbital is invalid");
    flat = flat * panel->extents[axis] + local;
  }
  if (flat >= panel->present.size() || panel->present[flat] == 0U)
    throw std::logic_error(
        "selected CCSD panel cache requested an undeclared spatial ERI");
  return panel->values[flat];
}

double SelectedIntegralPanelCache::operator()(
    const std::size_t p, const std::size_t q, const std::size_t r,
    const std::size_t s) const {
  if (!enabled_)
    throw std::logic_error("selected integral panel cache is disabled");
  if (!source_->contains(p, q, r, s))
    throw std::logic_error("selected CCSD panel cache requested an "
                           "undeclared ERI block (" +
                           std::to_string(p) + "," + std::to_string(q) +
                           "," + std::to_string(r) + "," +
                           std::to_string(s) + ")");
  const auto spatial_p = spatial_orbital(p);
  const auto spatial_q = spatial_orbital(q);
  const auto spatial_r = spatial_orbital(r);
  const auto spatial_s = spatial_orbital(s);
  double value = 0.0;
  if (p % 2 == r % 2 && q % 2 == s % 2)
    value += spatial_value(
        {spatial_p, spatial_r, spatial_q, spatial_s});
  if (p % 2 == s % 2 && q % 2 == r % 2)
    value -= spatial_value(
        {spatial_p, spatial_s, spatial_q, spatial_r});
  return value;
}

SelectedDoublesLayout::SelectedDoublesLayout(
    const SelectedCcsdManifoldPlan& plan)
    : monomer_count_(plan.monomer_count),
      blocks_(plan.doubles_blocks),
      distributed_layout_(block_sizes(blocks_)),
      occupied_orbitals_by_monomer_(plan.occupied_orbitals_by_monomer),
      virtual_orbitals_by_monomer_(plan.virtual_orbitals_by_monomer) {
  if (!plan.dependency_status.ready_for_sparse_ccsd())
    throw std::invalid_argument(
        "selected doubles storage requires a closed CCSD dependency plan");
  if (monomer_count_ == 0 || blocks_.empty())
    throw std::invalid_argument("selected doubles storage plan is empty");
  for (const auto& group : plan.occupied_orbitals_by_monomer)
    occupied_ += group.size();
  for (const auto& group : plan.virtual_orbitals_by_monomer)
    virtuals_ += group.size();
  occupied_monomer_.assign(occupied_, missing);
  occupied_local_.assign(occupied_, missing);
  virtual_monomer_.assign(virtuals_, missing);
  virtual_local_.assign(virtuals_, missing);
  for (std::size_t monomer = 0; monomer < monomer_count_; ++monomer) {
    const auto& occupied_group =
        plan.occupied_orbitals_by_monomer.at(monomer);
    for (std::size_t local = 0; local < occupied_group.size(); ++local) {
      const auto orbital = occupied_group[local];
      if (orbital >= occupied_ || occupied_monomer_[orbital] != missing)
        throw std::invalid_argument(
            "selected doubles occupied orbital map is invalid");
      occupied_monomer_[orbital] = monomer;
      occupied_local_[orbital] = local;
    }
    const auto& virtual_group =
        plan.virtual_orbitals_by_monomer.at(monomer);
    for (std::size_t local = 0; local < virtual_group.size(); ++local) {
      const auto orbital = virtual_group[local];
      if (orbital >= virtuals_ || virtual_monomer_[orbital] != missing)
        throw std::invalid_argument(
            "selected doubles virtual orbital map is invalid");
      virtual_monomer_[orbital] = monomer;
      virtual_local_[orbital] = local;
    }
  }
  if (std::ranges::find(occupied_monomer_, missing) !=
          occupied_monomer_.end() ||
      std::ranges::find(virtual_monomer_, missing) != virtual_monomer_.end())
    throw std::invalid_argument(
        "selected doubles orbital map is incomplete");
  auto tuple_count = checked_product(
      monomer_count_, monomer_count_,
      "selected doubles monomer tuple count overflows");
  tuple_count = checked_product(
      tuple_count, tuple_count,
      "selected doubles monomer tuple count overflows");
  block_by_monomer_tuple_.assign(tuple_count, missing);
  std::size_t selected_elements = 0;
  for (std::size_t index = 0; index < blocks_.size(); ++index) {
    const auto tuple = monomer_tuple_index(blocks_[index].monomers);
    if (block_by_monomer_tuple_[tuple] != missing)
      throw std::invalid_argument(
          "selected doubles plan contains a duplicate block");
    block_by_monomer_tuple_[tuple] = index;
    if (blocks_[index].element_count >
        std::numeric_limits<std::size_t>::max() - selected_elements)
      throw std::overflow_error(
          "selected doubles element count overflows");
    selected_elements += blocks_[index].element_count;
  }
  if (selected_elements != plan.selected_doubles_elements ||
      selected_elements != distributed_layout_.scientific_element_count())
    throw std::invalid_argument(
        "selected doubles storage disagrees with the manifold plan");
}

std::size_t SelectedDoublesLayout::occupied() const noexcept {
  return occupied_;
}

std::size_t SelectedDoublesLayout::virtuals() const noexcept {
  return virtuals_;
}

std::size_t SelectedDoublesLayout::block_count() const noexcept {
  return blocks_.size();
}

std::size_t SelectedDoublesLayout::selected_element_count() const noexcept {
  return distributed_layout_.scientific_element_count();
}

const SelectedBlockLayout&
SelectedDoublesLayout::distributed_layout() const noexcept {
  return distributed_layout_;
}

const DoublesMonomerBlock&
SelectedDoublesLayout::block(const std::size_t index) const {
  return blocks_.at(index);
}

const std::vector<std::size_t>& SelectedDoublesLayout::axis_orbitals(
    const std::size_t block, const std::size_t axis) const {
  if (axis >= 4)
    throw std::out_of_range("selected doubles axis is out of range");
  const auto monomer = blocks_.at(block).monomers[axis];
  return axis < 2 ? occupied_orbitals_by_monomer_.at(monomer)
                  : virtual_orbitals_by_monomer_.at(monomer);
}

bool SelectedDoublesLayout::selected(const std::size_t i,
                                     const std::size_t j,
                                     const std::size_t a,
                                     const std::size_t b) const {
  if (i >= occupied_ || j >= occupied_ || a >= virtuals_ || b >= virtuals_)
    throw std::out_of_range("selected doubles orbital is out of range");
  return block_by_monomer_tuple_[monomer_tuple_index(
             {occupied_monomer_[i], occupied_monomer_[j],
              virtual_monomer_[a], virtual_monomer_[b]})] != missing;
}

std::size_t SelectedDoublesLayout::block_index(
    const std::size_t i, const std::size_t j, const std::size_t a,
    const std::size_t b) const {
  if (i >= occupied_ || j >= occupied_ || a >= virtuals_ || b >= virtuals_)
    throw std::out_of_range("selected doubles orbital is out of range");
  const auto result = block_by_monomer_tuple_[monomer_tuple_index(
      {occupied_monomer_[i], occupied_monomer_[j], virtual_monomer_[a],
       virtual_monomer_[b]})];
  if (result == missing)
    throw std::out_of_range("doubles orbital tuple is outside the selection");
  return result;
}

std::size_t SelectedDoublesLayout::element_index(
    const std::size_t i, const std::size_t j, const std::size_t a,
    const std::size_t b) const {
  const auto block_id = block_index(i, j, a, b);
  const auto& descriptor = blocks_[block_id];
  const std::array<std::size_t, 4> local{
      occupied_local_.at(i), occupied_local_.at(j), virtual_local_.at(a),
      virtual_local_.at(b)};
  std::size_t result = local[0];
  for (std::size_t axis = 1; axis < 4; ++axis)
    result = result * descriptor.extents[axis] + local[axis];
  if (result >= descriptor.element_count)
    throw std::logic_error("selected doubles local offset is invalid");
  return result;
}

std::vector<std::vector<double>>
SelectedDoublesLayout::pack(const Tensor4& dense) const {
  if (dense.dimensions() !=
      std::array<std::size_t, 4>{occupied_, occupied_, virtuals_, virtuals_})
    throw std::invalid_argument("selected doubles dense dimensions mismatch");
  std::vector<std::vector<double>> result;
  result.reserve(blocks_.size());
  for (const auto& descriptor : blocks_)
    result.emplace_back(descriptor.element_count, 0.0);
  for (std::size_t i = 0; i < occupied_; ++i)
    for (std::size_t j = 0; j < occupied_; ++j)
      for (std::size_t a = 0; a < virtuals_; ++a)
        for (std::size_t b = 0; b < virtuals_; ++b) {
          if (!selected(i, j, a, b))
            continue;
          const auto value = dense(i, j, a, b);
          if (!std::isfinite(value))
            throw std::invalid_argument(
                "selected doubles dense source contains a non-finite value");
          result[block_index(i, j, a, b)][element_index(i, j, a, b)] = value;
        }
  return result;
}

Tensor4 SelectedDoublesLayout::unpack(
    const std::vector<std::vector<double>>& blocks) const {
  if (blocks.size() != blocks_.size())
    throw std::invalid_argument("selected doubles block count mismatch");
  Tensor4 result({occupied_, occupied_, virtuals_, virtuals_});
  for (std::size_t i = 0; i < occupied_; ++i)
    for (std::size_t j = 0; j < occupied_; ++j)
      for (std::size_t a = 0; a < virtuals_; ++a)
        for (std::size_t b = 0; b < virtuals_; ++b) {
          if (!selected(i, j, a, b))
            continue;
          const auto block_id = block_index(i, j, a, b);
          if (blocks[block_id].size() != blocks_[block_id].element_count)
            throw std::invalid_argument("selected doubles block size mismatch");
          const auto value = blocks[block_id][element_index(i, j, a, b)];
          if (!std::isfinite(value))
            throw std::invalid_argument(
                "selected doubles block contains a non-finite value");
          result(i, j, a, b) = value;
        }
  return result;
}

std::size_t SelectedDoublesLayout::monomer_tuple_index(
    const std::array<std::size_t, 4>& labels) const {
  std::size_t result = 0;
  for (const auto label : labels) {
    if (label >= monomer_count_)
      throw std::out_of_range("selected doubles monomer is out of range");
    result = result * monomer_count_ + label;
  }
  return result;
}

SelectedDoublesView::SelectedDoublesView(
    const SelectedDoublesLayout& layout,
    const std::vector<std::vector<double>>& blocks)
    : layout_(&layout), blocks_(&blocks) {
  if (blocks.size() != layout.block_count())
    throw std::invalid_argument("selected T2 view block-count mismatch");
  for (std::size_t block = 0; block < blocks.size(); ++block)
    if (blocks[block].size() != layout.block(block).element_count)
      throw std::invalid_argument("selected T2 view block-size mismatch");
}

std::array<std::size_t, 4>
SelectedDoublesView::dimensions() const noexcept {
  return {layout_->occupied(), layout_->occupied(), layout_->virtuals(),
          layout_->virtuals()};
}

std::size_t SelectedDoublesView::size() const noexcept {
  return layout_->occupied() * layout_->occupied() * layout_->virtuals() *
         layout_->virtuals();
}

double SelectedDoublesView::operator()(const std::size_t i,
                                       const std::size_t j,
                                       const std::size_t a,
                                       const std::size_t b) const {
  if (!layout_->selected(i, j, a, b))
    return 0.0;
  const auto block = layout_->block_index(i, j, a, b);
  return blocks_->at(block).at(layout_->element_index(i, j, a, b));
}

bool SelectedDoublesView::all_finite() const noexcept {
  for (const auto& block : *blocks_)
    for (const auto value : block)
      if (!std::isfinite(value))
        return false;
  return true;
}

SelectedIntermediateLayout::SelectedIntermediateLayout(
    const SelectedCcsdManifoldPlan& plan,
    std::vector<CcsdIntermediateMonomerBlock> blocks)
    : monomer_count_(plan.monomer_count), blocks_(std::move(blocks)),
      distributed_layout_(intermediate_block_sizes(blocks_)) {
  if (!plan.dependency_status.ready_for_sparse_ccsd() ||
      monomer_count_ == 0 || blocks_.empty())
    throw std::invalid_argument(
        "selected intermediate storage requires a closed nonempty plan");
  spaces_ = blocks_.front().spaces;
  for (std::size_t axis = 0; axis < 4; ++axis) {
    const auto& groups = spaces_[axis] == CcsdOrbitalSpace::occupied
        ? plan.occupied_orbitals_by_monomer
        : plan.virtual_orbitals_by_monomer;
    orbitals_by_monomer_[axis] = groups;
    std::size_t orbital_count = 0;
    for (const auto& group : groups)
      orbital_count += group.size();
    orbital_monomer_[axis].assign(orbital_count, missing);
    orbital_local_[axis].assign(orbital_count, missing);
    for (std::size_t monomer = 0; monomer < groups.size(); ++monomer)
      for (std::size_t local = 0; local < groups[monomer].size(); ++local) {
        const auto orbital = groups[monomer][local];
        if (orbital >= orbital_count ||
            orbital_monomer_[axis][orbital] != missing)
          throw std::invalid_argument(
              "selected intermediate orbital map is invalid");
        orbital_monomer_[axis][orbital] = monomer;
        orbital_local_[axis][orbital] = local;
      }
    if (std::ranges::find(orbital_monomer_[axis], missing) !=
        orbital_monomer_[axis].end())
      throw std::invalid_argument(
          "selected intermediate orbital map is incomplete");
  }
  auto tuple_count = checked_product(
      monomer_count_, monomer_count_,
      "selected intermediate monomer tuple count overflows");
  tuple_count = checked_product(
      tuple_count, tuple_count,
      "selected intermediate monomer tuple count overflows");
  block_by_tuple_.assign(tuple_count, missing);
  for (std::size_t index = 0; index < blocks_.size(); ++index) {
    const auto& block = blocks_[index];
    if (block.spaces != spaces_)
      throw std::invalid_argument(
          "selected intermediate block spaces are inconsistent");
    const auto tuple = tuple_index(block.monomers);
    if (block_by_tuple_[tuple] != missing)
      throw std::invalid_argument(
          "selected intermediate plan contains a duplicate block");
    std::size_t expected = 1;
    for (std::size_t axis = 0; axis < 4; ++axis) {
      const auto extent =
          orbitals_by_monomer_[axis].at(block.monomers[axis]).size();
      if (block.extents[axis] != extent)
        throw std::invalid_argument(
            "selected intermediate block extent disagrees with plan");
      expected = checked_product(
          expected, extent,
          "selected intermediate block element count overflows");
    }
    if (expected != block.element_count)
      throw std::invalid_argument(
          "selected intermediate block element count is invalid");
    block_by_tuple_[tuple] = index;
  }
}

std::size_t SelectedIntermediateLayout::block_count() const noexcept {
  return blocks_.size();
}

const SelectedBlockLayout&
SelectedIntermediateLayout::distributed_layout() const noexcept {
  return distributed_layout_;
}

const CcsdIntermediateMonomerBlock&
SelectedIntermediateLayout::block(const std::size_t index) const {
  return blocks_.at(index);
}

bool SelectedIntermediateLayout::selected(
    const std::array<std::size_t, 4>& orbitals) const {
  std::array<std::size_t, 4> labels{};
  for (std::size_t axis = 0; axis < 4; ++axis) {
    if (orbitals[axis] >= orbital_monomer_[axis].size())
      throw std::out_of_range(
          "selected intermediate orbital is out of range");
    labels[axis] = orbital_monomer_[axis][orbitals[axis]];
  }
  return block_by_tuple_[tuple_index(labels)] != missing;
}

std::size_t SelectedIntermediateLayout::block_index(
    const std::array<std::size_t, 4>& orbitals) const {
  std::array<std::size_t, 4> labels{};
  for (std::size_t axis = 0; axis < 4; ++axis) {
    if (orbitals[axis] >= orbital_monomer_[axis].size())
      throw std::out_of_range(
          "selected intermediate orbital is out of range");
    labels[axis] = orbital_monomer_[axis][orbitals[axis]];
  }
  const auto result = block_by_tuple_[tuple_index(labels)];
  if (result == missing)
    throw std::out_of_range(
        "intermediate orbital tuple is outside the selection");
  return result;
}

std::size_t SelectedIntermediateLayout::element_index(
    const std::array<std::size_t, 4>& orbitals) const {
  const auto block_id = block_index(orbitals);
  const auto& descriptor = blocks_[block_id];
  std::size_t result = orbital_local_[0].at(orbitals[0]);
  for (std::size_t axis = 1; axis < 4; ++axis)
    result = result * descriptor.extents[axis] +
             orbital_local_[axis].at(orbitals[axis]);
  if (result >= descriptor.element_count)
    throw std::logic_error(
        "selected intermediate local offset is invalid");
  return result;
}

const std::vector<std::size_t>& SelectedIntermediateLayout::axis_orbitals(
    const std::size_t block, const std::size_t axis) const {
  if (axis >= 4)
    throw std::out_of_range("selected intermediate axis is out of range");
  return orbitals_by_monomer_[axis].at(blocks_.at(block).monomers[axis]);
}

std::size_t SelectedIntermediateLayout::tuple_index(
    const std::array<std::size_t, 4>& labels) const {
  std::size_t result = 0;
  for (const auto label : labels) {
    if (label >= monomer_count_)
      throw std::out_of_range(
          "selected intermediate monomer is out of range");
    result = result * monomer_count_ + label;
  }
  return result;
}

struct SelectedDoublesPhaseCache::Implementation {
  Implementation(const SelectedDoublesLayout& input_layout,
                 const GlobalArraysBlockTensor& input_values,
                 const std::size_t input_maximum_bytes)
      : layout(&input_layout), values(&input_values),
        maximum_bytes(input_maximum_bytes) {}

  const SelectedDoublesLayout* layout{};
  const GlobalArraysBlockTensor* values{};
  std::size_t maximum_bytes{};
  std::size_t cached_bytes{};
  std::unordered_map<std::size_t, std::vector<double>> blocks;
  SelectedDoublesCacheStatistics statistics;
};

SelectedDoublesPhaseCache::SelectedDoublesPhaseCache(
    const SelectedDoublesLayout& layout,
    const GlobalArraysBlockTensor& values,
    const std::size_t maximum_bytes)
    : implementation_(std::make_unique<Implementation>(
          layout, values, maximum_bytes)) {
  if (maximum_bytes == 0)
    throw std::invalid_argument(
        "selected doubles phase-cache limit must be nonzero");
  if (values.layout().block_count() != layout.block_count())
    throw std::invalid_argument(
        "selected doubles phase-cache block count mismatch");
  for (std::size_t block = 0; block < layout.block_count(); ++block)
    if (values.layout().block_size(block) !=
        layout.block(block).element_count)
      throw std::invalid_argument(
          "selected doubles phase-cache block size mismatch");
}

SelectedDoublesPhaseCache::~SelectedDoublesPhaseCache() = default;

void SelectedDoublesPhaseCache::prepare(std::vector<std::size_t> blocks) {
  std::sort(blocks.begin(), blocks.end());
  blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
  std::size_t required_bytes = 0;
  for (const auto block : blocks) {
    const auto elements = implementation_->layout->block(block).element_count;
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
        elements * sizeof(double) >
            std::numeric_limits<std::size_t>::max() - required_bytes)
      throw std::overflow_error(
          "selected doubles phase-cache size overflows");
    required_bytes += elements * sizeof(double);
  }
  if (required_bytes > implementation_->maximum_bytes)
    throw std::runtime_error(
        "selected doubles phase requires " +
        std::to_string(required_bytes) +
        " cache bytes, exceeding the explicit limit of " +
        std::to_string(implementation_->maximum_bytes));
  implementation_->blocks.clear();
  implementation_->blocks.reserve(blocks.size());
  for (const auto block : blocks) {
    auto values = implementation_->values->get_block(block);
    implementation_->statistics.fetched_blocks += 1;
    implementation_->statistics.fetched_bytes +=
        values.size() * sizeof(double);
    implementation_->blocks.emplace(block, std::move(values));
  }
  implementation_->cached_bytes = required_bytes;
  implementation_->statistics.peak_cached_bytes = std::max(
      implementation_->statistics.peak_cached_bytes, required_bytes);
  implementation_->statistics.prepared_phases += 1;
}

void SelectedDoublesPhaseCache::clear() noexcept {
  implementation_->blocks.clear();
  implementation_->cached_bytes = 0;
}

double SelectedDoublesPhaseCache::value(
    const std::size_t i, const std::size_t j, const std::size_t a,
    const std::size_t b) const {
  if (!implementation_->layout->selected(i, j, a, b))
    return 0.0;
  const auto block = implementation_->layout->block_index(i, j, a, b);
  const auto found = implementation_->blocks.find(block);
  if (found == implementation_->blocks.end())
    throw std::logic_error(
        "selected doubles phase-cache miss; hidden GA fetch is forbidden");
  return found->second.at(
      implementation_->layout->element_index(i, j, a, b));
}

SelectedDoublesCacheStatistics
SelectedDoublesPhaseCache::statistics() const noexcept {
  return implementation_->statistics;
}

struct SelectedIntermediatePhaseCache::Implementation {
  Implementation(const SelectedIntermediateLayout& input_layout,
                 const GlobalArraysBlockTensor& input_values,
                 const std::size_t input_maximum_bytes)
      : layout(&input_layout), values(&input_values),
        maximum_bytes(input_maximum_bytes) {}

  const SelectedIntermediateLayout* layout{};
  const GlobalArraysBlockTensor* values{};
  std::size_t maximum_bytes{};
  std::size_t cached_bytes{};
  std::unordered_map<std::size_t, std::vector<double>> blocks;
  SelectedDoublesCacheStatistics statistics;
};

SelectedIntermediatePhaseCache::SelectedIntermediatePhaseCache(
    const SelectedIntermediateLayout& layout,
    const GlobalArraysBlockTensor& values,
    const std::size_t maximum_bytes)
    : implementation_(std::make_unique<Implementation>(
          layout, values, maximum_bytes)) {
  if (maximum_bytes == 0)
    throw std::invalid_argument(
        "selected intermediate phase-cache limit must be nonzero");
  if (values.layout().block_count() != layout.block_count())
    throw std::invalid_argument(
        "selected intermediate phase-cache block count mismatch");
  for (std::size_t block = 0; block < layout.block_count(); ++block)
    if (values.layout().block_size(block) !=
        layout.block(block).element_count)
      throw std::invalid_argument(
          "selected intermediate phase-cache block size mismatch");
}

SelectedIntermediatePhaseCache::~SelectedIntermediatePhaseCache() = default;

void SelectedIntermediatePhaseCache::prepare(
    std::vector<std::size_t> blocks) {
  std::sort(blocks.begin(), blocks.end());
  blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
  std::size_t required_bytes = 0;
  for (const auto block : blocks) {
    const auto elements = implementation_->layout->block(block).element_count;
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
        elements * sizeof(double) >
            std::numeric_limits<std::size_t>::max() - required_bytes)
      throw std::overflow_error(
          "selected intermediate phase-cache size overflows");
    required_bytes += elements * sizeof(double);
  }
  if (required_bytes > implementation_->maximum_bytes)
    throw std::runtime_error(
        "selected intermediate phase requires " +
        std::to_string(required_bytes) +
        " cache bytes, exceeding the explicit limit of " +
        std::to_string(implementation_->maximum_bytes));
  implementation_->blocks.clear();
  implementation_->blocks.reserve(blocks.size());
  for (const auto block : blocks) {
    auto values = implementation_->values->get_block(block);
    implementation_->statistics.fetched_blocks += 1;
    implementation_->statistics.fetched_bytes +=
        values.size() * sizeof(double);
    implementation_->blocks.emplace(block, std::move(values));
  }
  implementation_->cached_bytes = required_bytes;
  implementation_->statistics.peak_cached_bytes = std::max(
      implementation_->statistics.peak_cached_bytes, required_bytes);
  implementation_->statistics.prepared_phases += 1;
}

void SelectedIntermediatePhaseCache::clear() noexcept {
  implementation_->blocks.clear();
  implementation_->cached_bytes = 0;
}

double SelectedIntermediatePhaseCache::value(
    const std::array<std::size_t, 4>& orbitals) const {
  if (!implementation_->layout->selected(orbitals))
    return 0.0;
  const auto block = implementation_->layout->block_index(orbitals);
  const auto found = implementation_->blocks.find(block);
  if (found == implementation_->blocks.end())
    throw std::logic_error(
        "selected intermediate phase-cache miss; hidden GA fetch is "
        "forbidden");
  return found->second.at(
      implementation_->layout->element_index(orbitals));
}

SelectedDoublesCacheStatistics
SelectedIntermediatePhaseCache::statistics() const noexcept {
  return implementation_->statistics;
}

} // namespace localized_cc
