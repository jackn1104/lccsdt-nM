# Notices and provenance

The project-owned code is distributed under the MIT license in LICENSE.
This does not replace third-party copyrights or licenses.

The files in `vendor/reference/` are an unchanged source/header subset of
ModernQC, obtained from the LMP2 repository at commit
`5645f78a94b53dcd5720ff1bc23b6f045b655e0f`. Their BSD-3-Clause license and
copyright are preserved in `vendor/reference/LICENSE`.

The CCSD/triples numerical source snapshot is
`c7bd288b6edb825fe75bff99b0ade9a9adb1e0cf`. Internal identifiers are retained
to avoid unnecessary numerical and checkpoint-format changes. NWChem's legacy
CCSD module at commit `37975061856208c94946e6f59afa8b26906a4ac3` was an
algorithmic reference, particularly its spatial-pair and OpenMP organization:
https://github.com/nwchemgit/nwchem/tree/37975061856208c94946e6f59afa8b26906a4ac3/src/ccsd
NWChem is distributed under ECL-2.0; its solver source is not vendored here.
The C++ spatial executor and ISO_C_BINDING Fortran kernels implement this
project's CCSD equations, not a copied standalone NWChem solver. This does not
claim NWChem performance parity. The original Python prototype is not included.

Global Arrays is not enabled or linked by this package. Its existing notice
is retained for the optional, inactive source adapter. Eigen, Libint2,
BLAS/LAPACK and optional MPI/OpenMP/Fortran runtimes are external build
dependencies, not relicensed by this package. Preserve their applicable
notices when redistributing their binaries or sources.

The water Z-matrix test fixture originates from the NWChem QA water test;
the four-water reference XYZ retains its original provenance comment.
The W2/aVDZ validation XYZ and component energies are from the archived
full-dimer record `w2_ccd_cct_ccq_mbe.56823943.out`, transcribed from the
existing reference inventory, not calculated during package preparation.

Packaging changes: standalone build configuration, public command-line
wrapper, run manifests, examples and documentation. The numerical kernels
have not been renamed or rewritten for this package. No authorship or
performance claim is inferred from cosmetic interface names.
