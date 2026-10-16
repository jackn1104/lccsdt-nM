#include "localized_cc/frozen_core.hpp"
#include "localized_cc/parallel.hpp"
#include "localized_cc/semidirect_hamiltonian.hpp"
#include "localized_cc/selected_ccsd_manifold.hpp"
#include "localized_cc/selected_semidirect_hamiltonian.hpp"

#include "modernqc/integrals/integral_provider.hpp"
#include "modernqc/molecule/molecule.hpp"
#include "modernqc/scf/rhf.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

modernqc::linalg::Matrix columns(const modernqc::linalg::Matrix& source,
                                 const std::size_t first,
                                 const std::size_t count) {
  modernqc::linalg::Matrix result(source.rows(), count);
  for (std::size_t row = 0; row < source.rows(); ++row)
    for (std::size_t column = 0; column < count; ++column)
      result(row, column) = source(row, first + column);
  return result;
}

localized_cc::Matrix copy_matrix(const modernqc::linalg::Matrix& source) {
  localized_cc::Matrix result({source.rows(), source.columns()});
  for (std::size_t row = 0; row < source.rows(); ++row)
    for (std::size_t column = 0; column < source.columns(); ++column)
      result(row, column) = source(row, column);
  return result;
}

modernqc::scf::RhfOptions rhf_options() {
  modernqc::scf::RhfOptions result;
  result.energy_tolerance = 1.0e-11;
  result.density_rms_tolerance = 1.0e-9;
  result.commutator_rms_tolerance = 1.0e-8;
  return result;
}

double maximum_error(const std::vector<double>& left,
                     const std::vector<double>& right) {
  if (left.size() != right.size())
    throw std::invalid_argument("semidirect comparison size mismatch");
  double result = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index)
    result = std::max(result, std::abs(left[index] - right[index]));
  return result;
}

} // namespace

int main(int argc, char** argv) {
  localized_cc::parallel::Runtime parallel(argc, argv);
  try {
    using modernqc::molecule::Atom;
    const modernqc::molecule::Molecule water(
        std::vector<Atom>{{"O", 8, {0.0, 0.0, 0.0}},
                          {"H", 1, {1.43233673, 0.0, 1.10715266}},
                          {"H", 1, {-1.43233673, 0.0, 1.10715266}}},
        0, 1);
    modernqc::integrals::Libint2IntegralProvider integrals(
        water, "sto-3g", true);
    const auto rhf = modernqc::scf::run_rhf(water, integrals, rhf_options());
    constexpr std::size_t frozen_core = 1;
    const auto occupied = columns(rhf.coefficients, frozen_core, 4);
    const auto virtuals = columns(rhf.coefficients, 5,
                                  rhf.coefficients.columns() - 5);

    const auto dense_ao =
        integrals.compute_validation_ao_eris(8U * 1024U * 1024U);
    localized_cc::Tensor4 dense_eri(
        {dense_ao.ao_count, dense_ao.ao_count, dense_ao.ao_count,
         dense_ao.ao_count});
    dense_eri.data() = dense_ao.values;
    auto dense = localized_cc::build_frozen_core_spatial_hamiltonian(
        localized_cc::FrozenCoreInput{
            copy_matrix(columns(rhf.coefficients, 0, frozen_core)),
            copy_matrix(occupied), copy_matrix(virtuals),
            copy_matrix(rhf.overlap), copy_matrix(rhf.core_hamiltonian),
            std::move(dense_eri), rhf.nuclear_repulsion_energy,
            rhf.total_energy, true});
    const auto semidirect =
        localized_cc::build_semidirect_spatial_hamiltonian(
            rhf, integrals, occupied, virtuals, frozen_core,
            32U * 1024U * 1024U);
    const std::vector<int> occupied_labels(2 * occupied.columns(), 0);
    const std::vector<int> virtual_labels(2 * virtuals.columns(), 0);
    localized_cc::SelectedCcsdManifoldOptions selected_options;
    selected_options.maximum_monomers = 1;
    const auto selected_plan =
        localized_cc::plan_selected_ccsd_manifold(
            occupied_labels, virtual_labels, selected_options);
    localized_cc::SelectedSemidirectTransformDiagnostics
        selected_diagnostics;
    const auto selected =
        localized_cc::build_selected_semidirect_spatial_hamiltonian(
            rhf, integrals, occupied, virtuals, selected_plan,
            32U * 1024U * 1024U, selected_diagnostics);
    bool rejected_zero_slice_cap = false;
    try {
      static_cast<void>(localized_cc::build_semidirect_spatial_hamiltonian(
          rhf, integrals, occupied, virtuals, frozen_core,
          32U * 1024U * 1024U, 1.0e-10, 1.0e-8, 0));
    } catch (const std::invalid_argument&) {
      rejected_zero_slice_cap = true;
    }
    if (!rejected_zero_slice_cap)
      throw std::runtime_error(
          "semidirect Hamiltonian accepted a zero cooperative slice cap");

    const double fock_error = maximum_error(
        dense.fock.data(), semidirect.hamiltonian.fock.data());
    const double hcore_error = maximum_error(
        dense.hcore.data(), semidirect.hamiltonian.hcore.data());
    const double eri_error = maximum_error(
        dense.eri_chemist.data(),
        semidirect.hamiltonian.eri_chemist.data());
    const double reference_error = std::abs(
        dense.reference_energy - semidirect.hamiltonian.reference_energy);
    const localized_cc::SpinOrbitalEriView dense_spin(dense);
    double selected_eri_error = 0.0;
    std::size_t selected_eri_comparisons = 0;
    std::size_t undeclared_eri_quartets = 0;
    const auto spin_orbitals = 2 * (dense.nocc + dense.nvirt);
    for (std::size_t p = 0; p < spin_orbitals; ++p)
      for (std::size_t q = 0; q < spin_orbitals; ++q)
        for (std::size_t r = 0; r < spin_orbitals; ++r)
          for (std::size_t s = 0; s < spin_orbitals; ++s) {
            if (!selected.integrals.contains(p, q, r, s)) {
              ++undeclared_eri_quartets;
              continue;
            }
            selected_eri_error = std::max(
                selected_eri_error,
                std::abs(selected.integrals(p, q, r, s) -
                         dense_spin(p, q, r, s)));
            ++selected_eri_comparisons;
          }
    if (fock_error > 2.0e-11 || hcore_error > 2.0e-12 ||
        eri_error > 2.0e-11 || reference_error > 2.0e-11 ||
        semidirect.diagnostics.evaluated_shell_quartets !=
            semidirect.diagnostics.unique_shell_quartets ||
        selected_eri_error > 2.0e-11 ||
        selected_eri_comparisons == 0 || undeclared_eri_quartets == 0 ||
        selected_diagnostics.selected_integral_bytes == 0 ||
        selected_diagnostics.requested_spatial_integral_blocks == 0 ||
        selected_diagnostics.symmetry_unique_spatial_integral_blocks == 0 ||
        selected_diagnostics.symmetry_unique_spatial_integral_blocks >=
            selected_diagnostics.requested_spatial_integral_blocks ||
        selected_diagnostics.selected_integral_tuple_lookup_bytes == 0 ||
        selected_diagnostics.avoided_full_active_mo_eri_bytes == 0 ||
        selected_diagnostics.estimated_peak_bytes >
            32U * 1024U * 1024U ||
        semidirect.diagnostics.estimated_peak_bytes >=
            6 * dense_ao.values.size() * sizeof(double))
      throw std::runtime_error(
          "semidirect Hamiltonian does not match the dense oracle");
    if (localized_cc::parallel::world().size > 1) {
      const auto one_first_index_slice_bytes =
          occupied.rows() * occupied.rows() * occupied.rows() *
          sizeof(double);
      const auto multigroup =
          localized_cc::build_semidirect_spatial_hamiltonian(
              rhf, integrals, occupied, virtuals, frozen_core,
              32U * 1024U * 1024U, 1.0e-10, 1.0e-8,
              2 * one_first_index_slice_bytes);
      if (multigroup.diagnostics.cooperative_first_index_groups < 2 ||
          maximum_error(dense.fock.data(),
                        multigroup.hamiltonian.fock.data()) > 2.0e-11 ||
          maximum_error(dense.eri_chemist.data(),
                        multigroup.hamiltonian.eri_chemist.data()) > 2.0e-11)
        throw std::runtime_error(
            "multi-group semidirect Hamiltonian does not match the dense "
            "oracle");
    }
    if (localized_cc::parallel::world().rank == 0) {
      std::cout.setf(std::ios::scientific);
      std::cout << "semidirect dense Fock error " << fock_error << '\n'
                << "semidirect dense hcore error " << hcore_error << '\n'
                << "semidirect dense ERI error " << eri_error << '\n'
                << "semidirect dense RHF error " << reference_error << '\n';
      std::cout << "selected semidirect dense ERI error "
                << selected_eri_error << '\n'
                << "selected semidirect declared comparisons "
                << selected_eri_comparisons << '\n'
                << "selected semidirect undeclared quartets "
                << undeclared_eri_quartets << '\n'
                << "selected semidirect bytes "
                << selected_diagnostics.selected_integral_bytes << '\n'
                << "selected semidirect requested spatial blocks "
                << selected_diagnostics.requested_spatial_integral_blocks
                << '\n'
                << "selected semidirect symmetry-unique spatial blocks "
                << selected_diagnostics
                       .symmetry_unique_spatial_integral_blocks
                << '\n'
                << "selected semidirect tuple lookup bytes "
                << selected_diagnostics.selected_integral_tuple_lookup_bytes
                << '\n'
                << "avoided full active MO ERI bytes "
                << selected_diagnostics.avoided_full_active_mo_eri_bytes
                << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    if (localized_cc::parallel::world().rank == 0)
      std::cerr << "semidirect Hamiltonian test failure: " << error.what()
                << '\n';
    return 1;
  }
}
