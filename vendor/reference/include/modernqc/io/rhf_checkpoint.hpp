#pragma once

#include "modernqc/basis/basis_set.hpp"
#include "modernqc/linalg/matrix.hpp"
#include "modernqc/localization/pipek_mezey.hpp"
#include "modernqc/molecule/molecule.hpp"
#include "modernqc/mp2/laplace_fit.hpp"
#include "modernqc/mp2/laplace_mp2.hpp"
#include "modernqc/mp2/tiled_laplace_mp2.hpp"
#include "modernqc/scf/rhf.hpp"

#include <cstddef>
#include <filesystem>
#include <vector>

namespace modernqc::io {

struct CheckpointRuntime {
  int mpi_size{1};
  int openmp_threads{1};
};

struct CanonicalRhfCheckpoint {
  double total_energy;
  double electronic_energy;
  linalg::Matrix density;
  linalg::Matrix fock;
  linalg::Matrix core_hamiltonian;
  linalg::Matrix overlap;
  linalg::Matrix orthogonalizer;
  linalg::Matrix coefficients;
  std::vector<double> orbital_energies;
  std::vector<double> occupations;
  std::vector<int> frozen_core_mask;
  std::vector<std::size_t> active_occupied_indices;
  std::vector<std::size_t> virtual_indices;
  double orbital_orthonormality_maximum_error;
};

struct LocalizedRhfCheckpoint {
  CanonicalRhfCheckpoint canonical;
  localization::PipekMezeyResult occupied;
  localization::PipekMezeyResult virtual_space;
};

struct LaplaceMp2Checkpoint {
  mp2::LaplaceFitResult fit;
  mp2::IncoreLaplaceMp2Result energy;
};

void write_rhf_checkpoint(const std::filesystem::path& path,
                          const molecule::Molecule& molecule,
                          const basis::BasisSet& basis,
                          const scf::RhfResult& result, bool frozen_core,
                          CheckpointRuntime runtime, bool overwrite,
                          const localization::PipekMezeyResult*
                              occupied_localization = nullptr,
                          const localization::PipekMezeyResult*
                              virtual_localization = nullptr,
                          const mp2::LaplaceFitResult*
                              laplace_fit = nullptr,
                          const mp2::IncoreLaplaceMp2Result*
                              laplace_mp2 = nullptr,
                          const mp2::TiledLaplaceMp2Result*
                              tiled_laplace_mp2 = nullptr);

[[nodiscard]] linalg::Matrix read_rhf_restart_density(
    const std::filesystem::path& path,
    const molecule::Molecule& expected_molecule,
    const basis::BasisSet& expected_basis);

[[nodiscard]] CanonicalRhfCheckpoint read_canonical_rhf_checkpoint(
    const std::filesystem::path& path,
    const molecule::Molecule& expected_molecule,
    const basis::BasisSet& expected_basis);

// Restore a converged same-geometry, same-basis RHF result without executing
// another SCF iteration. All scientific matrices and stored diagnostics are
// validated before the result is returned.
[[nodiscard]] scf::RhfResult read_rhf_result_checkpoint(
    const std::filesystem::path& path,
    const molecule::Molecule& expected_molecule,
    const basis::BasisSet& expected_basis);

[[nodiscard]] LocalizedRhfCheckpoint read_localized_rhf_checkpoint(
    const std::filesystem::path& path,
    const molecule::Molecule& expected_molecule,
    const basis::BasisSet& expected_basis);

[[nodiscard]] LaplaceMp2Checkpoint read_laplace_mp2_checkpoint(
    const std::filesystem::path& path);

// Transactional Phase 11 sidecar. Periodic writes replace the destination
// only after the temporary HDF5 file has been flushed and read back.
void write_tiled_laplace_mp2_restart(
    const std::filesystem::path& path,
    const mp2::TiledLaplaceMp2Progress& progress, bool overwrite);

[[nodiscard]] mp2::TiledLaplaceMp2Progress
read_tiled_laplace_mp2_restart(const std::filesystem::path& path);

}  // namespace modernqc::io
