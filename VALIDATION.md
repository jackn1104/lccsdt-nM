# Release validation status

## Laptop checks performed on 2026-10-15

This source preview has a tested laptop path, not general cluster certification.
Apple Silicon, 24 GiB RAM, AppleClang 16.0.0, Libint2 2.13.1, Eigen 5.0.1,
OpenBLAS 0.3.34 (LP64), libomp 22.1.8 and Boost 1.90.0 were used.
Debug and Release each passed all six CTest suites: CCSD, Laplace triples,
semidirect Hamiltonian, PM symmetry, PM symmetry triples and launcher contracts.
The seven Python launcher cases are grouped into one CTest suite.

The exact bundled W2 geometry was verified by SHA-256
`29e5c7938bf9085928a5d16fc2f159dcb0406257ca6075ef517a31cbef7369df`.
The molecular run used four OpenMP threads, one BLAS thread, frozen oxygen 1s,
full PM nM, native spatial CCSD in localized orbitals, no density fitting,
CCSD energy-change and RMS-residual thresholds of 1e-8, AO Schwarz threshold
1e-12, and seven-point Laplace triples with exact closed-shell spin-flip reuse.

| Quantity | Calculated (Eh) | Calculated minus stored reference (Eh) |
|---|---:|---:|
| RHF | -152.087407652935 | +1.201400e-8 |
| CCSD correlation | -0.457815879677 | -9.795405e-9 |
| (T) | -0.011021441322 | -1.556273e-9 |
| Total CCSD(T) | -152.556244973934 | +6.623111e-10 |

All four component checks passed the unchanged 1e-7 Eh gate. The stored
canonical energies are transcribed references with provenance in
`examples/w2-avdz-reference.json`; no new external reference calculation was run.
CCSD converged in 29 iterations. Its final RMS residual measures were
1.707996e-9 and 5.233732e-10; the raw L2 norms are not those stopping measures.
The reported Laplace quadrature bound was 1.498822332229e-8 Eh.

| PM class | CCSD correlation (Eh) | (T) (Eh) |
|---|---:|---:|
| M1 | -0.437665458036 | -0.009654995847 |
| M2 | -0.020150421641 | -0.001366445475 |

Reported partition closure errors were -3.33e-16 Eh for CCSD and zero at
printed precision for (T). W2 has no M3-M6 contribution; higher-class coverage
is exercised by synthetic equation fixtures, not by this molecular example.
There is no independent canonical PM nM reference in the supplied data.

Measured molecular wall time was 290.026 s, including 248.858 s CCSD and
32.587 s triples. These are one laptop's observations, not scaling claims.
A second run using copied checkpoints restored RHF, PM, Hamiltonian and
converged CCSD, reported zero CCSD iterations, and recomputed triples in
33.660 s total. All four energies and all W2 M1/M2 values agreed at the
12-decimal printed precision (1e-12 Eh comparison gate).

Raw evidence is retained locally under `runs/w2-validation.crS9dU/`, including
`calculation/W2_VALIDATED.json` and `restart-check/RESTART_VALIDATED.json`.
Generated runs, binaries and private checkpoints are excluded from source archives.

AddressSanitizer is **blocked**, not passed: the installed runtime aborts
before application startup with `sanitizer_malloc_mac.inc:189`. An empty
instrumented `int main() { return 0; }` reproduces the same failure without
any solver or numerical libraries. The failed ASan logs and reproducer are
preserved under the earlier `runs/w2-validation.PXzBK9/` and `build/`.

UndefinedBehaviorSanitizer was run separately with `-fsanitize=undefined
-fno-omit-frame-pointer` and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`.
All six suites passed (18.06 s); no undefined-behavior diagnostics were reported.
This does not replace the blocked ASan check. The sanitizer run covered small
equation fixtures, not the full W2 molecular calculation or external libraries.

## Remaining release gates

Before claiming wider production support, on a validation machine:

1. Configure a Debug build and run `ctest --test-dir build --output-on-failure`.
2. Repeat for a Release build; verify compiler, integral-library basis data,
   BLAS integer ABI and compatible threading runtimes.
3. Check MPI equations on allocated compute nodes, not a login node:

```bash
srun --nodes=1 --ntasks=4 --cpus-per-task=1 ./build/ccsd_tests
srun --nodes=1 --ntasks=4 --cpus-per-task=1 ./build/laplace_triples_tests
srun --nodes=1 --ntasks=4 --cpus-per-task=1 ./build/pm_symmetry_triples_tests
```

4. Repeat a small full molecular run at one/four ranks. Compare finite total
   and PM nM energies to an independent reference (1e-7 Eh target), with
   identical geometry, core convention and saved PM orbitals. Compare
   point-group on/off only for an exactly verified geometry; otherwise
   require explicit rejection. Compare spin two-sector/four-sector equations.
5. On Linux, interrupt a disk run after durable rows/points. Restart it and
   compare against uninterrupted results at the same settings. Confirm the
   triples-only restart reports zero CCSD iterations and reuses moment rows.
6. Test insufficient memory/disk, changed geometry or binary, malformed input,
   missing checkpoints and simultaneous launches. Failures must not overwrite
   evidence or write COMPLETE.
7. Run compiler sanitizers on the validation host. Diagnostic runtime setup
   is site-specific; do not assume a loader preload suitable for every cluster.

Known limits: replicated resident Hamiltonian and spatial CCSD arrays;
no general linear-scaling claim; not every basis/cluster fits a laptop or the
example cluster allocation. Disk moment persistence is Linux-only. Point-group
and disk propagation cannot be combined. The seven-point rule may reject
a large denominator range. Some internal experimental modules remain linked
to support the molecular driver but are not exposed by the public launcher.

The retained kernel-provenance documents describe historical stages. They
are preserved as attribution, not as current build instructions or validation
evidence. A final provenance and redistribution review is still required
before publishing this preview as a supported release.
