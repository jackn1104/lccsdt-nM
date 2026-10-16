# lccsdt

Full-space localized CCSD and orbital-invariant Laplace `(T)` for neutral,
closed-shell water clusters. Occupied and virtual orbitals are localized
separately with Pipek–Mezey; symmetric Löwdin populations assign monomer
labels. The default freezes oxygen 1s orbitals and requires assignment
confidence of at least 0.8. No density fitting or monomer-span truncation is
used. CCSD reports M1–M4 and `(T)` reports M1–M6, up to the number of waters.

**Source preview with laptop validation:** Debug and Release equation tests,
full-space W2/aVDZ, and a copied-checkpoint triples-only restart passed on
Apple Silicon. MPI, Linux disk persistence and larger molecular cases have
not been validated for this standalone package. See VALIDATION.md before production.

## Build

Requirements: CMake 3.22+, a C++20 compiler, Eigen3, Libint2, Boost headers, LP64 BLAS/LAPACK,
and Python 3.9+ for the launcher. MPI and OpenMP are optional. Fortran kernels
are optional (`-DLCCSDT_FORTRAN=ON`). Dependencies are not downloaded.
Install the integral library's basis data, including `ano-rcc` for PM
localization and the requested orbital basis. Set `LIBINT_DATA_PATH` if
your integral library installation requires it.

Laptops (serial or shared-memory):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

Use `-DCMAKE_PREFIX_PATH=/path/to/dependencies` if libraries are not found.
For separate installations, set `-DEigen3_DIR=...` and `-DLibint2_DIR=...`.
OpenMP is used when available; a compiler without OpenMP remains serial.

Clusters:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLCCSDT_MPI=ON
cmake --build build -j 8
```

Load the site's compiler/MPI/library environment first. For a compiler
wrapper that supplies serial BLAS/LAPACK, add
`-DCMAKE_CXX_COMPILER=CC -DLCCSDT_WRAPPER_BLAS=ON`. Do not enable threaded
BLAS beneath OpenMP without a deliberate benchmark. Run the numerical tests
on compute nodes; see VALIDATION.md.

## Laptop example

```bash
./lccsdt run examples/water.xyz --basis sto-3g --threads 2 --memory-mib 2048 --output runs/water
```

This is a small demonstration geometry, not a canonical-energy reference.
Larger augmented bases may exceed a laptop's memory. `--memory-mib` limits
additional numerical workspace **per rank**, not total process RSS. Resident
integrals, amplitudes, runtime state and temporary arrays require additional
memory. There is no promise that arbitrary cluster/basis combinations fit.

### Explicit W2/aVDZ laptop validation

The bundled `examples/w2-avdz.xyz` is the supplied CCSD(T)/aVDZ-optimized
dimer input geometry. On a macOS laptop with the listed Homebrew dependencies
already installed and at least 16 GiB RAM, run:

```bash
bash scripts/validate-laptop-w2.sh
```

This is an explicit foreground calculation: it builds Debug and Release,
runs equation/launcher tests, then computes full-space W2/aVDZ with PM nM
and closed-shell spin symmetry. It uses at most four physical threads and
checks all four canonical energy components to 1e-7 Eh. Existing runs are
not overwritten. It does not install dependencies or publish anything.
It then copies the completed checkpoints and verifies a triples-only restart
with zero CCSD iterations and total/M1/M2 agreement to 1e-12 Eh.
The canonical reference has no PM nM components: those are checked for sum
closure here and by equation tests, not against invented reference values.

## Other geometries and basis sets

Supply an XYZ in angstrom, with each monomer grouped as O,H,H. The geometry
is used exactly as supplied: the program does not optimize or symmetrize it.
Use basis-matched optimized geometries if that is your comparison protocol.

Basis names: `sto-3g`, `cc-pVDZ`, `aug-cc-pVDZ`, `aug-cc-pVTZ`,
`aug-cc-pVQZ`; short names `ccdz`, `avdz`, `avtz`, `avqz` also work.
Names are case-insensitive. Other elements/fragments, open shells, charged
systems and arbitrary point groups are not exposed by this launcher.

## Cluster example

The supplied Slurm script requests two CPU nodes, four ranks per node and
32 physical threads per rank. Its 64 CPUs/task assumes two hardware threads
per physical core; adapt these together for your cluster. Its memory request
is 460000 MiB per node. The executable receives 65536 MiB additional
workspace per rank, leaving room for resident allocations. This is a plan,
not a certified RSS bound.

```bash
sbatch --account=YOUR_ACCOUNT examples/cluster.slurm /absolute/path/w6.xyz aug-cc-pVDZ /absolute/path/new-run
```

For another geometry/basis, change the arguments, for example:

```bash
sbatch --account=YOUR_ACCOUNT examples/cluster.slurm /absolute/path/w3.xyz aug-cc-pVTZ /absolute/path/new-w3-run
```

Edit partition/QoS/constraint/module settings to match your site. The script
does not contain a private account, fixed scratch path or automatic module
selection. Multi-node disk execution requires a shared filesystem visible
at the same absolute path on all nodes. Check quota, not only `df` free space.
The disk allowance defaults to 262144 MiB per rank and can be changed with
`--scratch-mib`; two nodes with eight ranks can therefore need substantial
scratch. It is a cap, not a reservation supplied by the scheduler.

## Symmetry

The default `--symmetry spin` uses verified closed-shell spin inversion to
compute two representative triples sectors, with full nM multiplicities.
Spin-aware moments and exact virtual-exchange reuse remain enabled. This
does not assume that different water molecules are geometrically equivalent.

`--symmetry point-group --storage memory` enables symmetry-adapted PM
localization and verified point-group task reuse. Candidate mappings exist
for particular 3-, 4-, 6- and 8-water geometries, not every cluster of those
sizes. Atom, orbital and Hamiltonian covariance checks must pass; coordinates
are never snapped and ambiguous orbital maps fail. This mode uses the
four-sector representation and requires memory for its resident triples.

**Point-group reuse is not implemented for the disk-backed propagation path.**
The launcher rejects that combination rather than silently ignoring it.
No claim is made that every possible symmetry is exploited.

## Checkpoints and restart

Each run stores `run.json`, the input geometry, uniquely named live logs,
RHF/PM/CCSD/Hamiltonian checkpoints, and (for disk mode) moment row files and
sector/point checkpoints. Keep the entire run directory. Do not run two
writers against the same checkpoints or move individual files.

```bash
./lccsdt restart /absolute/path/run
```

To require saved converged CCSD and run only the remaining triples:

```bash
./lccsdt restart /absolute/path/run --triples-only
```

For the supplied two-node Slurm layout:

```bash
sbatch --account=YOUR_ACCOUNT examples/restart.slurm /absolute/path/run
```

Restart preserves the original scientific settings, executable checksum and
MPI layout. Verification reads are still necessary. Completed durable moment
rows and Laplace points are reused; unfinished rows/points may be repeated.
Memory-mode moment construction is not durable. Sector/point checkpoints are
Linux-only; macOS retains RHF, PM, Hamiltonian and CCSD checkpoints but recomputes
triples after an interruption. Old development checkpoints
are not automatically imported by this new interface.

## Accuracy and output

Defaults: CCSD convergence threshold `1e-8`, RMS residual measure, AO Schwarz
threshold `1e-12`, seven minimax Laplace points with requested quadrature
energy bound `1e-7 Eh`. If the quadrature is insufficient, the calculation
must fail rather than silently weaken the bound. These settings do not
constitute a universal `1e-7 Eh` bound on the complete CCSD(T) error.

The launcher verifies finite reported energies, total-energy arithmetic and
nM sum closure to `1e-7 Eh`. `COMPLETE` means exit zero and these output
checks passed; it does not certify agreement with an independent reference.
Validate total and each nM class against a same-geometry, same-core,
same-basis reference before scientific use. nM components depend on the PM
orbital gauge; compare them using the same localized orbitals and labels.

License: MIT for project-owned code; required component terms and provenance
are preserved separately in NOTICES.md. The runtime numerical kernels use
C++20, with optional Fortran contractions; Python only launches and checks runs.
