# lccsdt-nM

Full-space localized CCSD and Laplace-transformed (T) for neutral, closed-shell
water clusters, with Pipek–Mezey orbitals, PM nM energy partitions and symmetry
reuse. Oxygen 1s orbitals are frozen by default.

## Build

Requires CMake 3.22+, C++20, Python 3.9+, Eigen3, Libint2, Boost headers and
LP64 BLAS/LAPACK. OpenMP is optional. Install the requested basis and `ano-rcc`
basis data; set `LIBINT_DATA_PATH` if needed.

```bash
git clone https://github.com/jackn1104/lccsdt-nM.git
cd lccsdt-nM
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

If dependencies are not found, add `-DCMAKE_PREFIX_PATH=/path/to/dependencies`
to the configure command.

## Run on a laptop

```bash
./lccsdt run examples/w2-avdz.xyz --basis avdz --threads 4 --memory-mib 4096 --storage memory --output runs/w2-avdz
```

Use XYZ coordinates in angstrom, ordered O,H,H for each water. Coordinates
are not optimized or symmetrized. Supported bases: `sto-3g`, `cc-pVDZ`,
`aug-cc-pVDZ`, `aug-cc-pVTZ`, `aug-cc-pVQZ` (aliases: `ccdz`, `avdz`, `avtz`,
`avqz`). Choose a new output directory for each calculation.

`--memory-mib` is additional workspace **per rank**, not a total memory limit.
Leave RAM for integrals and amplitudes; large clusters/bases may not fit.

## Run on a cluster

Load your site's compiler, MPI and library environment, then build with MPI:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLCCSDT_MPI=ON
cmake --build build -j 8
sbatch --account=YOUR_ACCOUNT examples/cluster.slurm /absolute/path/cluster.xyz avdz /absolute/path/new-run
```

Run tests on compute nodes. For Cray wrappers providing serial BLAS/LAPACK,
add `-DCMAKE_CXX_COMPILER=CC -DLCCSDT_WRAPPER_BLAS=ON` when configuring.

The Slurm template uses 2 nodes, 4 ranks/node, 32 threads/rank and 460000 MiB
RAM/node. Its 64 CPUs/task assumes two hardware threads/core. Adjust both
the allocation and launcher settings for your hardware. Disk-backed triples
require Linux and sufficient shared scratch space.

## Restart and output

Keep the entire run directory and the original executable/launcher. Never
run two writers on the same checkpoints.

```bash
./lccsdt restart /absolute/path/run
./lccsdt restart /absolute/path/run --triples-only
```

The second command requires converged saved CCSD and does not redo CCSD.
For cluster triples restarts, use `examples/restart.slurm` with an allocation
matching the saved run.

Progress and energies are saved in `output-*.out`; diagnostics in
`progress-*.log`. Add `--verbose` to display diagnostics.
[Example W2/aVDZ output](examples/w2-avdz.out).

Defaults: full PM nM, closed-shell spin symmetry, CCSD threshold `1e-8`,
AO screening `1e-12`, and requested Laplace quadrature bound `1e-7` hartree.
Optional `--symmetry point-group --storage memory` requires verified geometry,
orbital and Hamiltonian symmetry; it is not available for every cluster.

[MIT license](LICENSE). See [notices](notices/) and
[bundled dependency license](vendor/reference/LICENSE).
