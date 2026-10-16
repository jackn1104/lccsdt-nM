# Legacy-style mixed-language CCSD(T) backend

This module is the Fortran scientific-kernel layer for the localized CCSD(T)
program. C++ retains ownership of orbital localization, monomer assignment,
screening policy, checkpoint metadata, CSV output, and the M1--M4 CCSD plus
M1--M6 `(T)` decomposition.

## Source reference and provenance

The algorithmic and distributed-memory reference is the NWChem legacy
`src/ccsd` module at commit
`37975061856208c94946e6f59afa8b26906a4ac3`, audited on 2026-08-05:

<https://github.com/nwchemgit/nwchem/tree/37975061856208c94946e6f59afa8b26906a4ac3/src/ccsd>

NWChem is distributed under the Educational Community License 2.0. No NWChem
source file is copied into this repository at this phase. The new
`legacy_energy_kernels.F90` is an original ISO_C_BINDING implementation of the
spin-orbital equations already documented in this repository. If NWChem source
is incorporated later, its copyright headers, license, modification notices,
and provenance will be preserved.

## Why the NWChem directory cannot be linked by itself

The legacy entry point is not a standalone tensor library. Its driver directly
uses:

- NWChem RTDB task state and SCF `movecs` files;
- geometry, basis, integral, and Schwarz-screening services;
- MA stack allocation and distributed Global Arrays;
- NWChem parallel file I/O and CCSD restart files;
- task schedulers, timing, diagnostics, and error handling.

Consequently, copying `src/ccsd/*.F` into this CMake project would neither link
nor reproduce the legacy memory behavior. The production port preserves the
ownership and task boundaries by replacing each NWChem service with an
explicit backend interface. Phase 14 now supplies the first such production
dependency: Global Arrays v5.9.2 with MPI-3 one-sided transport.

The original C++ `GlobalArraysPairTensor` adapts the pinned NWChem internal-
transformation `g_nt2`/`g_nht2` layout: `(ia,jb)` global dimensions with
`(v,v)` chunks and occupied-pair task ownership. It is distinct from the
alternative `g_t2` flattened-vector layout in the same module. NWChem source
remains a pinned algorithmic reference; it has not been pasted into a
non-linkable standalone module.

## Implemented ABI

Configure with:

```text
-DCCSD_ENABLE_LEGACY_FORTRAN_BACKEND=ON
```

The current ABI performs the CCSD correlation-energy contraction, the
conventional triples `L*X/36` contraction, and the dominant spin-orbital
`W_abef` CCSD intermediate in Fortran. `W_abef` costs `O(o^2 v^4)` in the
present equations and uses bounds-check-free row-major offsets. Production
now streams one `(a,b)` block with `v^2` binary64 elements and contracts it
directly into rank-owned residual slices; the former replicated `v^4` tensor
and its global reduction are eliminated. A full-tensor Fortran kernel remains
only as an independent element-wise test reference. When enabled, the normal
CCSD and noncanonical triples solvers call the Fortran kernels; when disabled,
the existing C++ reference contractions remain active. The ABI makes
collective execution explicit: in distributed energy mode C++ passes the MPI
rank and rank count, Fortran assigns flattened work cyclically, and C++
executes one collective sum after the kernel. The normal production energy
functions use collective-free local mode because some callers, including the
noncanonical triples partition, intentionally execute on rank zero only.
Fortran never calls MPI in an innermost contraction loop.

This is the first validated boundary, not a claim of NWChem performance parity.
The next kernels to move are the remaining distributed T2 residual and
intermediates, DIIS/restart storage, semidirect integral transformation, and
streamed canonical triples driver. The Global Arrays runtime and pair store now
exist, but full parity still requires direct residual/DIIS ownership, a
spin-adapted spatial-orbital path, semidirect integral transformation, and
W1/W2 timing comparisons.

An additional driver-level admission route is selected with
`CCSD_SOLVER_BASIS=canonical` and
`CCSD_SOLVER_BACKEND=legacy-spatial`. It retains the exact closed-shell
pair-symmetric spatial T1/T2 state and spatial DIIS vectors throughout the
nonlinear solve. Spin amplitudes are materialized only at checkpoint and final
PM-rotation boundaries so the established checkpoint, partition, and Laplace
triples interfaces remain unchanged. The current residual evaluator is still
the retained spin-orbital equation oracle. Therefore this selector validates
the storage, restart, and scientific service boundaries required by the
legacy NWChem architecture; it does not yet provide the final tiled spatial
residual contractions or claim NWChem timing parity. The Perlmutter admission
runner is `examples/nersc/validate_legacy_spatial_ccsd_w2_interactive.sh`.

## Array and energy conventions

- C++ owns allocations and passes contiguous binary64 arrays.
- Fortran uses explicit row-major offsets matching `localized_cc::Tensor`; no
  implicit transpose or hidden copy occurs at the ABI.
- CCSD uses spin-orbital antisymmetrized ERIs and
  `F_ia t_i^a + (1/4)<ij||ab>t_ij^ab + (1/2)<ij||ab>t_i^a t_j^b`.
- Conventional triples uses the full-index `L*X/36` convention.
- Fortran and unrelated BLAS/OpenMP thread counts must be recorded in every
  benchmark because reduction order can cause bounded binary64 differences.
