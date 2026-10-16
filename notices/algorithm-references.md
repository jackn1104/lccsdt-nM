# Implementation references

Development records identify NWChem as an algorithmic reference, including
its legacy CCSD module at commit
`37975061856208c94946e6f59afa8b26906a4ac3`:

- [`ccsd_iterdrv2.F`](https://github.com/nwchemgit/nwchem/blob/37975061856208c94946e6f59afa8b26906a4ac3/src/ccsd/ccsd_iterdrv2.F)
  and [`ccsd_itm_omp.F`](https://github.com/nwchemgit/nwchem/blob/37975061856208c94946e6f59afa8b26906a4ac3/src/ccsd/ccsd_itm_omp.F):
  composite occupied–virtual pair storage, spatial contractions and parallel
  work organization. The optional pair-array implementation specifically
  follows the `(ia,jb)` layout with `(v,v)` chunks used by `g_nt2/g_nht2`.
- The [TCE triples routines](https://github.com/nwchemgit/nwchem/tree/37975061856208c94946e6f59afa8b26906a4ac3/src/tce/ccsd_t):
  doubles/singles moment, permutation and signed-denominator conventions.
- The [CCSD convergence documentation](https://nwchemgit.github.io/CCSD.html#thresh-convergence-threshold):
  a convergence-convention reference. This package defines its own residual
  norms explicitly; equal threshold values need not imply identical stopping
  criteria across programs.

These references do not mean that this package links the NWChem solver or
has the same execution architecture or performance. NWChem's source is
distributed under [ECL-2.0](https://github.com/nwchemgit/nwchem/blob/37975061856208c94946e6f59afa8b26906a4ac3/LICENSE.TXT).
This reference note does not replace any license or copyright notice required
for redistributed third-party material.

NWChem names in output-format readers and example-data provenance identify
the source format or calculation, not the implementation of this solver.
