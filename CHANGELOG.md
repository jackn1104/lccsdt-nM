# Changelog

## 0.1.0 preview — 2026-10-15

- Added the standalone build, full-space water-cluster launcher and reference W2 example.
- Fixed AppleClang OpenMP captures without changing contractions or equations.
- Located the integral library's required Boost headers explicitly.
- Prevented use of the Linux-only durable sector writer on macOS.
- Kept combined Laplace point state out of the two-sector restart path; retained
  converged CCSD and separately fingerprinted sector checkpoints where available.
- Added copied-checkpoint W2 restart regression checks to laptop validation.
- Passed Debug/Release equation tests, W2/aVDZ canonical component agreement at
  1e-7 Eh, PM nM closure and triples-only restart at 1e-12 Eh printed precision.
- Documented the independent ASan runtime startup failure and outstanding cluster gates.
- Passed all six test suites with UndefinedBehaviorSanitizer; ASan remains blocked.
