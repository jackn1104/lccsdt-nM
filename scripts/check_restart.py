"""Compare a copied-checkpoint W2 triples-only restart with its complete run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re


def completed_log(directory: Path) -> str:
    marker = (directory / "COMPLETE").read_text()
    name = next(line.removeprefix("log=") for line in marker.splitlines()
                if line.startswith("log="))
    if Path(name).name != name:
        raise ValueError("invalid completion log name")
    return (directory / name).read_text()


def energy(text: str, label: str) -> float:
    match = re.search(r"^" + re.escape(label) + r"\s+([-+0-9.eE]+)\s+Eh\s*$",
                      text, re.MULTILINE)
    if match is None or not math.isfinite(float(match[1])):
        raise ValueError(f"missing or nonfinite energy: {label}")
    return float(match[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("restart", type=Path)
    args = parser.parse_args()
    before = completed_log(args.reference)
    after = completed_log(args.restart)
    for marker in ("RHF complete; restored from checkpoint", "AO-to-MO skipped",
                   "converged localized CCSD checkpoint restored",
                   "closed-shell exact spin-flip amplitude gate passed"):
        if marker not in after:
            raise ValueError(f"restart did not confirm: {marker}")
    if not re.search(r"^CCSD iterations this run\s+0\s*$", after, re.MULTILINE):
        raise ValueError("restart recomputed CCSD")
    differences = {}
    for label in ("RHF", "localized CCSD corr", "localized (T)", "localized CCSD(T)",
                  "CCSD M1", "CCSD M2", "(T) M1", "(T) M2"):
        difference = energy(after, label) - energy(before, label)
        if abs(difference) > 1e-12:
            raise ValueError(f"restart differs for {label}: {difference} Eh")
        differences[label] = difference
        print(f"{label}: restart difference={difference:+.3e} Eh")
    with (args.restart / "RESTART_VALIDATED.json").open("x") as output:
        json.dump(dict(ccsd_iterations=0, tolerance_hartree=1e-12,
                       differences_hartree=differences), output, indent=2, sort_keys=True)
        output.write("\n")
    print("W2 TRIPLES-ONLY RESTART PASSED: zero CCSD iterations; total and M1/M2 agree")


if __name__ == "__main__":
    main()
