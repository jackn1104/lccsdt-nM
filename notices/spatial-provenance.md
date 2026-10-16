# Native spatial pair executor: NERSC admission candidate

## Scope and provenance

`CCSD_SOLVER_BACKEND=legacy-spatial` plus `CCSD_NATIVE_SPATIAL_PAIRS=1`
selects the candidate. The default remains the existing spin-expanded
residual evaluator. This is an original refactoring of this repository's
Stanton-factorized residual equations, **not copied NWChem source** and not
a claim of NWChem timing parity.

The NWChem reference is the legacy `src/ccsd` module at
`37975061856208c94946e6f59afa8b26906a4ac3`, in particular
[`ccsd_itm_omp.F`](https://github.com/nwchemgit/nwchem/blob/37975061856208c94946e6f59afa8b26906a4ac3/src/ccsd/ccsd_itm_omp.F).
It works with spatial pair arrays, thread buffers, GA ownership, and dense
linear algebra, but also requires NWChem MA/RTDB/integral/task services. It
cannot be pasted into this CMake library without porting those interfaces.
Its ECL-2.0 license was inspected; no upstream source is vendored by this
patch. The [OpenMP documentation](https://nwchemgit.github.io/CCSD.html#openmp)
describes CCSD threading and a preliminary 2--4-thread recommendation, not
a guarantee that our one-rank/128-thread execution is optimal.

## Equation/storage contract

Spatial amplitudes use `T(i,j,a,b)=T(j,i,b,a)`, not individual occupied/virtual
antisymmetry. Define `tau(i,j,a,b)=T(i,j,a,b)+t(i,a)t(j,b)`. The new executor
never creates iterative spin-expanded T1, T2, tau, or four-index spin
intermediates. Spin selection rules are still used as scalar accessors when
building two independent channels of the mixed intermediate; this is not
spin tensor allocation or a restriction of the excitation space.

The independent channels satisfy
`W(m_sigma,b_tau,e_rho,j_eta) = A(m,b,e,j) delta_sigma,rho delta_tau,eta
+ B(m,b,e,j) delta_sigma,eta delta_tau,rho`.
They are built once per residual evaluation, then contracted using spatial
`(i,a)` matrix rows. The occupied W_mnij term is likewise a spatial pair
matrix product. W_abef is built in bounded spatial external-pair panels,
with its OOVV dressing and final tau contraction evaluated by DGEMM.
The complete four-index VVVV intermediate is not allocated.

Every contraction retains all internal sums and off-diagonal localized Fock
couplings. No coefficient cutoff, monomer truncation, DF/RI, or changed
convergence criterion is introduced. PM symmetry is admitted by the existing
Hamiltonian covariance test; spatial amplitude covariance is checked before
representative W_abef task reuse. The signed map reconstructs every output.
Pair outputs and cached channels use MPI ownership plus rank-local OpenMP;
MPI collectives occur outside the contraction loops. The amplitude state,
Hamiltonian, and reduced intermediate matrices remain replicated.

The existing direct R1 and small-Fock residual kernels remain; they do not
allocate spin amplitudes but still use scalar spin selection and rebuild some
small intermediates. Mixed-channel construction and bare panel packing are
also not fully DGEMM-factorized. These are remaining performance limits,
not reasons to claim NWChem parity. No timing improvement has been measured.

DIIS stores spatial vectors, uses the configured Gram backend, and now honors
rank-zero extrapolation plus broadcast. Failed root DIIS propagates before
the amplitude broadcast. Checkpoints still use the established spin convention
at write/restart boundaries; the final triples/nM adapter still expands once.
Neither the Laplace (T) equations nor the partition implementation is changed.

## Resource and timing semantics

The conservative transient pair-workspace bound is
`8 * (16*(o*v)^2 + o^4)` bytes per rank, bounded by
`CCSD_WABEF_INVARIANT_CACHE_MIB`. W_abef thread panels separately use at most
`CCSD_WABEF_TILE_MIB`; the runner sets these to 65536 and 512 MiB.
This is not a total process-memory estimate: Hamiltonian, amplitudes, DIIS,
small Fock matrices, and checkpoint/triples adapters are additional.
No bare invariant spin cache is built. Native channel matrices are rebuilt
per residual, not persisted across iterations.

The existing `CCSD spatial residual workspace` field measures the old
solve-lifetime spin workspace and is zero here; it does NOT mean zero memory.
W_abef total wall time is measured. Its legacy build/BLAS subphase counters
are not instrumented in the candidate and zero is not a measured duration.
Native task/contraction counters describe global work, not a per-rank sum.

## Admission and commands

No executable tests or molecular calculations have been run on the Mac.
Non-OpenMP C++ syntax checks and shell syntax checks passed. Apple Clang
rejected the OpenMP driver flag and crashed when its OpenMP frontend was
invoked directly for syntax checking; OpenMP compilation is therefore still
unverified and assigned to the NERSC GNU build. No executable was produced.
Syntax-only checking is not numerical or performance acceptance. Added gates:

- deterministic nonseparable ERIs, unequal o/v dimensions, noncanonical Fock
  blocks and pair-symmetric amplitudes at scales 1e-4, 1e-2 and 1e-1;
- residual infinity error <=1e-12 + 1e-12*max(reference,candidate norms), with
  both infinity and two-norm differences reported;
- tile-cap variation, insufficient-workspace rejection, converged energy and
  amplitude-only restart comparisons;
- PM-symmetric residuals and nonlinear CCSD/nM comparisons against the
  unchanged executor; existing Laplace tests continue to run.

Within the existing one-node/256-logical-CPU NERSC allocation, after building
the updated sources, set these before the W4 runner:

```bash
export LCCSDT_NODES=1
export LCCSDT_COMPARE_NATIVE_SPATIAL=1
bash examples/nersc/validate_w4_pm_symmetry_interactive.sh
```

The `symmetry` arm uses native pairs, and `reference` uses the old executor;
both now enable PM task symmetry. Each arm records `executor.txt`, logs,
checkpoints and fresh timings. This mode must use fresh CCSD/triples work to
measure speed: either start fresh, or resume from a stopped **pre-CCSD**
checkpoint in each arm. Do not copy a running checkpoint. A converged CCSD or
completed triples restart is not a timing test and the comparison rejects it.
All numerical gates remain unchanged. Do not promote to default until the
NERSC equation gates, molecular energies/nM closure, and timing comparison
pass; a failure stops before any expensive molecular run.
