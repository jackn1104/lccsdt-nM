# 0.1.0 source preview

Prepared as a separate directory, without development Git history, private
campaigns, checkpoints, job logs, personal paths or account credentials.

The standalone CMake build, launcher, documentation, launcher contract tests,
and examples are new. Validation added semantics-preserving OpenMP capture
portability fixes, explicit Boost header discovery, a platform guard for
Linux-only sector persistence, and correct two-sector restart-state routing.
No equations, geometry or numerical tolerances were changed. Required
component licenses and historical provenance remain in the package.

The initial packaging snapshot matched the development numerical source.
The documented portability/restart changes now intentionally differ from it;
the development worktree itself remains unchanged. Generated local validation
logs contain absolute paths and are deliberately excluded from source archives.

Debug and Release builds and all six equation/launcher test suites passed.
W2/aVDZ and copied-checkpoint triples-only restart passed; see VALIDATION.md
for numerical values and evidence. All six suites also passed UBSan.
ASan is blocked by a reproduced runtime
startup failure. MPI, Linux disk restart and larger molecular cases remain
unvalidated for this standalone package. This public source preview preserves
the validated numerical code; publication does not certify those remaining
gates. Existing live cluster calculations were not accessed or changed.

The source is prepared for supported closed-shell water-cluster calculations,
not a general open-shell or arbitrary-fragment quantum-chemistry program.
Preserving full PM nM does not mean each nM contribution is independently
decoupled from other excitation classes in the CCSD equations.
