#pragma once

#include "localized_cc/distributed_block_tensor.hpp"
#include "localized_cc/frozen_core.hpp"
#include "localized_cc/selected_ccsd_manifold.hpp"
#include "localized_cc/tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace localized_cc {

using SelectedSpatialIntegralBlockProducer = std::function<
    std::vector<std::vector<double>>(
        const std::vector<CcsdIntegralMonomerBlock>&,
        const std::vector<std::vector<std::size_t>>&,
        const std::vector<std::vector<std::size_t>>&)>;

// Fail-closed, read-only storage for the spatial chemists' ERI blocks needed
// by the selected spin-orbital CCSD equation dependency closure.  A requested
// antisymmetrized spin integral is reconstructed from at most two spatial
// blocks, without storing the sixteen-fold spin expansion. The dense
// constructor packs an exact SpatialRhfHamiltonian and provides a numerical
// reference for the selected-block producer constructor.
// Values outside the declared spin closure are not represented and an
// attempted access throws instead of silently returning zero or falling back
// to a dense ERI.
class SelectedIntegralStore {
public:
  SelectedIntegralStore(const SelectedCcsdManifoldPlan& plan,
                        const SpatialRhfHamiltonian& dense_oracle);
  SelectedIntegralStore(
      const SelectedCcsdManifoldPlan& plan, std::size_t spatial_occupied,
      std::size_t spatial_virtuals,
      const SelectedSpatialIntegralBlockProducer& producer);

  [[nodiscard]] std::array<std::size_t, 4> dimensions() const noexcept;
  [[nodiscard]] std::size_t declared_spin_block_count() const noexcept;
  [[nodiscard]] std::size_t declared_spin_element_count() const noexcept;
  [[nodiscard]] std::size_t block_count() const noexcept;
  [[nodiscard]] std::size_t spatial_component_request_count() const noexcept;
  [[nodiscard]] std::size_t selected_element_count() const noexcept;
  [[nodiscard]] std::size_t selected_bytes() const noexcept;
  [[nodiscard]] std::size_t tuple_lookup_bytes() const noexcept;
  [[nodiscard]] bool contains(std::size_t p, std::size_t q,
                              std::size_t r, std::size_t s) const;
  [[nodiscard]] std::array<std::size_t, 2>
  spatial_dimensions() const noexcept;
  [[nodiscard]] bool contains_spatial_chemist(
      std::size_t p, std::size_t q, std::size_t r,
      std::size_t s) const;
  [[nodiscard]] double spatial_chemist(std::size_t p, std::size_t q,
                                       std::size_t r,
                                       std::size_t s) const;
  [[nodiscard]] double operator()(std::size_t p, std::size_t q,
                                  std::size_t r, std::size_t s) const;
  [[nodiscard]] bool all_finite() const noexcept;

private:
  static constexpr std::size_t missing =
      std::numeric_limits<std::size_t>::max();

  struct OrbitalAddress {
    CcsdOrbitalSpace space{};
    std::size_t monomer{};
    std::size_t local{};
    std::size_t token{};
  };

  [[nodiscard]] std::size_t tuple_key(
      const std::array<CcsdOrbitalSpace, 4>& spaces,
      const std::array<std::size_t, 4>& monomers) const;
  [[nodiscard]] std::array<OrbitalAddress, 4> spin_addresses(
      std::size_t p, std::size_t q, std::size_t r,
      std::size_t s) const;
  [[nodiscard]] std::array<OrbitalAddress, 4> spatial_addresses(
      const std::array<std::size_t, 4>& orbitals) const;
  [[nodiscard]] std::array<OrbitalAddress, 4> canonical_spatial_addresses(
      const std::array<OrbitalAddress, 4>& addresses) const;
  [[nodiscard]] std::size_t address_tuple_key(
      const std::array<OrbitalAddress, 4>& addresses) const;
  [[nodiscard]] std::size_t spin_block_lookup(std::size_t key) const;
  [[nodiscard]] std::size_t spatial_block_lookup(std::size_t key) const;
  [[nodiscard]] std::size_t spin_block_index(
      const std::array<OrbitalAddress, 4>& addresses) const;
  [[nodiscard]] std::size_t spatial_block_index(
      const std::array<OrbitalAddress, 4>& addresses) const;
  [[nodiscard]] std::size_t spatial_element_index(
      std::size_t block,
      const std::array<OrbitalAddress, 4>& addresses) const;
  [[nodiscard]] std::size_t spatial_element_index_unchecked(
      std::size_t block,
      const std::array<OrbitalAddress, 4>& addresses) const noexcept;
  [[nodiscard]] std::size_t spatial_orbital(std::size_t spin_orbital) const;
  [[nodiscard]] double spatial_value(
      const std::array<std::size_t, 4>& orbitals) const;
  [[nodiscard]] double spatial_value_unchecked(
      const std::array<OrbitalAddress, 4>& addresses) const;

  std::size_t monomer_count_{};
  std::size_t occupied_{};
  std::size_t virtuals_{};
  std::size_t spatial_occupied_{};
  std::size_t spatial_virtuals_{};
  std::vector<CcsdIntegralMonomerBlock> spatial_descriptors_;
  std::vector<std::vector<double>> blocks_;
  std::unordered_map<std::size_t, std::size_t> spin_block_by_tuple_;
  std::unordered_map<std::size_t, std::size_t> spatial_block_by_tuple_;
  std::vector<std::size_t> spin_block_by_tuple_dense_;
  std::vector<std::size_t> spatial_block_by_tuple_dense_;
  std::vector<std::uint8_t> canonical_permutation_by_tuple_;
  std::vector<OrbitalAddress> spin_orbitals_;
  std::vector<OrbitalAddress> spatial_orbitals_;
  std::vector<std::size_t> spin_to_spatial_;
  std::size_t declared_spin_blocks_{};
  std::size_t spatial_component_requests_{};
  std::size_t declared_spin_elements_{};
  std::size_t selected_elements_{};
};

// Bounded, read-only, equation-pattern panels for the non-VVVV spatial ERIs
// repeatedly consumed by the tiled CCSD intermediate and doubles kernels.
// The cache is built once from a fail-closed SelectedIntegralStore, retains a
// presence byte for every panel element, and never materializes an AO N^4
// tensor.  If all five exact panels do not fit the caller's byte cap, the
// cache remains disabled and the solver uses SelectedIntegralStore directly.
class SelectedIntegralPanelCache {
public:
  SelectedIntegralPanelCache(const SelectedIntegralStore& source,
                             std::size_t maximum_bytes);

  [[nodiscard]] bool enabled() const noexcept;
  [[nodiscard]] std::size_t required_bytes() const noexcept;
  [[nodiscard]] std::size_t allocated_bytes() const noexcept;
  [[nodiscard]] std::size_t populated_elements() const noexcept;
  [[nodiscard]] double operator()(std::size_t p, std::size_t q,
                                  std::size_t r, std::size_t s) const;

private:
  struct Panel {
    std::array<std::size_t, 4> extents{};
    std::array<unsigned char, 4> spaces{};
    std::vector<double> values;
    std::vector<std::uint8_t> present;
  };

  [[nodiscard]] std::size_t spatial_orbital(
      std::size_t spin_orbital) const;
  [[nodiscard]] double spatial_value(
      const std::array<std::size_t, 4>& orbitals) const;

  const SelectedIntegralStore* source_{};
  std::size_t spatial_occupied_{};
  std::size_t spatial_virtuals_{};
  std::size_t spin_occupied_{};
  std::size_t spin_virtuals_{};
  std::size_t required_bytes_{};
  std::size_t allocated_bytes_{};
  std::size_t populated_elements_{};
  bool enabled_{};
  Panel oooo_;
  Panel ooov_;
  Panel oovv_;
  Panel ovov_;
  Panel ovvv_;
};

// Deterministic orbital-to-monomer-block addressing for selected T2, tau,
// residual, and DIIS vectors. It contains metadata only; scientific values
// live either in rank-local block vectors or GlobalArraysBlockTensor.
class SelectedDoublesLayout {
public:
  explicit SelectedDoublesLayout(const SelectedCcsdManifoldPlan& plan);

  [[nodiscard]] std::size_t occupied() const noexcept;
  [[nodiscard]] std::size_t virtuals() const noexcept;
  [[nodiscard]] std::size_t block_count() const noexcept;
  [[nodiscard]] std::size_t selected_element_count() const noexcept;
  [[nodiscard]] const SelectedBlockLayout& distributed_layout() const noexcept;
  [[nodiscard]] const DoublesMonomerBlock& block(std::size_t index) const;
  [[nodiscard]] const std::vector<std::size_t>& axis_orbitals(
      std::size_t block, std::size_t axis) const;

  [[nodiscard]] bool selected(std::size_t i, std::size_t j,
                              std::size_t a, std::size_t b) const;
  [[nodiscard]] std::size_t block_index(std::size_t i, std::size_t j,
                                        std::size_t a,
                                        std::size_t b) const;
  [[nodiscard]] std::size_t element_index(std::size_t i, std::size_t j,
                                          std::size_t a,
                                          std::size_t b) const;

  [[nodiscard]] std::vector<std::vector<double>>
  pack(const Tensor4& dense) const;
  [[nodiscard]] Tensor4
  unpack(const std::vector<std::vector<double>>& blocks) const;

private:
  static constexpr std::size_t missing =
      std::numeric_limits<std::size_t>::max();
  [[nodiscard]] std::size_t monomer_tuple_index(
      const std::array<std::size_t, 4>& labels) const;

  std::size_t monomer_count_{};
  std::size_t occupied_{};
  std::size_t virtuals_{};
  std::vector<DoublesMonomerBlock> blocks_;
  SelectedBlockLayout distributed_layout_;
  std::vector<std::size_t> block_by_monomer_tuple_;
  std::vector<std::size_t> occupied_monomer_;
  std::vector<std::size_t> occupied_local_;
  std::vector<std::size_t> virtual_monomer_;
  std::vector<std::size_t> virtual_local_;
  std::vector<std::vector<std::size_t>> occupied_orbitals_by_monomer_;
  std::vector<std::vector<std::size_t>> virtual_orbitals_by_monomer_;
};

// Read-only indexed view of selected T2 blocks. Amplitudes outside the
// selected manifold are exactly zero by definition of the projected model.
// This view lets downstream moment builders consume converged block storage
// without materializing a dense o-o-v-v tensor.
class SelectedDoublesView {
public:
  SelectedDoublesView(const SelectedDoublesLayout& layout,
                      const std::vector<std::vector<double>>& blocks);

  [[nodiscard]] std::array<std::size_t, 4> dimensions() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] double operator()(std::size_t i, std::size_t j,
                                  std::size_t a, std::size_t b) const;
  [[nodiscard]] bool all_finite() const noexcept;

private:
  const SelectedDoublesLayout* layout_{};
  const std::vector<std::vector<double>>* blocks_{};
};

// Deterministic addressing shared by W_mnij, W_mbej, and streamed W_abef
// blocks. Axis orbital indices are relative to their declared occupied or
// virtual spaces; no spin-orbital offset is accepted at this boundary.
class SelectedIntermediateLayout {
public:
  SelectedIntermediateLayout(
      const SelectedCcsdManifoldPlan& plan,
      std::vector<CcsdIntermediateMonomerBlock> blocks);

  [[nodiscard]] std::size_t block_count() const noexcept;
  [[nodiscard]] const SelectedBlockLayout& distributed_layout() const noexcept;
  [[nodiscard]] const CcsdIntermediateMonomerBlock&
  block(std::size_t index) const;
  [[nodiscard]] bool selected(
      const std::array<std::size_t, 4>& orbitals) const;
  [[nodiscard]] std::size_t block_index(
      const std::array<std::size_t, 4>& orbitals) const;
  [[nodiscard]] std::size_t element_index(
      const std::array<std::size_t, 4>& orbitals) const;
  [[nodiscard]] const std::vector<std::size_t>& axis_orbitals(
      std::size_t block, std::size_t axis) const;

private:
  static constexpr std::size_t missing =
      std::numeric_limits<std::size_t>::max();
  [[nodiscard]] std::size_t tuple_index(
      const std::array<std::size_t, 4>& labels) const;

  std::size_t monomer_count_{};
  std::array<CcsdOrbitalSpace, 4> spaces_{};
  std::vector<CcsdIntermediateMonomerBlock> blocks_;
  SelectedBlockLayout distributed_layout_;
  std::vector<std::size_t> block_by_tuple_;
  std::array<std::vector<std::size_t>, 4> orbital_monomer_;
  std::array<std::vector<std::size_t>, 4> orbital_local_;
  std::array<std::vector<std::vector<std::size_t>>, 4> orbitals_by_monomer_;
};

struct SelectedDoublesCacheStatistics {
  std::size_t prepared_phases{};
  std::size_t fetched_blocks{};
  std::size_t fetched_bytes{};
  std::size_t peak_cached_bytes{};
};

// A phase cache makes the no-MPI-in-inner-loop rule executable. The caller
// derives all source block IDs for one residual/intermediate phase, prepares
// them collectively before entering OpenMP kernels, and may then perform only
// read-only scalar lookup. A cache miss throws instead of issuing hidden GA
// traffic from a contraction loop.
class SelectedDoublesPhaseCache {
public:
  SelectedDoublesPhaseCache(const SelectedDoublesLayout& layout,
                            const GlobalArraysBlockTensor& values,
                            std::size_t maximum_bytes);
  ~SelectedDoublesPhaseCache();

  SelectedDoublesPhaseCache(const SelectedDoublesPhaseCache&) = delete;
  SelectedDoublesPhaseCache& operator=(
      const SelectedDoublesPhaseCache&) = delete;
  SelectedDoublesPhaseCache(SelectedDoublesPhaseCache&&) = delete;
  SelectedDoublesPhaseCache& operator=(SelectedDoublesPhaseCache&&) = delete;

  void prepare(std::vector<std::size_t> blocks);
  void clear() noexcept;
  [[nodiscard]] double value(std::size_t i, std::size_t j,
                             std::size_t a, std::size_t b) const;
  [[nodiscard]] SelectedDoublesCacheStatistics statistics() const noexcept;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

// Fail-closed read-only cache for one selected W/tau intermediate phase. The
// declared block list comes directly from the equation dependency closure.
// prepare() is the only operation that may fetch GA blocks; value() performs
// local lookup and throws on an undeclared selected block.
class SelectedIntermediatePhaseCache {
public:
  SelectedIntermediatePhaseCache(
      const SelectedIntermediateLayout& layout,
      const GlobalArraysBlockTensor& values, std::size_t maximum_bytes);
  ~SelectedIntermediatePhaseCache();

  SelectedIntermediatePhaseCache(const SelectedIntermediatePhaseCache&) =
      delete;
  SelectedIntermediatePhaseCache& operator=(
      const SelectedIntermediatePhaseCache&) = delete;
  SelectedIntermediatePhaseCache(SelectedIntermediatePhaseCache&&) = delete;
  SelectedIntermediatePhaseCache& operator=(
      SelectedIntermediatePhaseCache&&) = delete;

  void prepare(std::vector<std::size_t> blocks);
  void clear() noexcept;
  [[nodiscard]] double value(
      const std::array<std::size_t, 4>& orbitals) const;
  [[nodiscard]] double operator()(
      const std::array<std::size_t, 4>& orbitals) const {
    return value(orbitals);
  }
  [[nodiscard]] SelectedDoublesCacheStatistics statistics() const noexcept;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

} // namespace localized_cc
