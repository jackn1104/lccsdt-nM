"""Check a completed W2 run against the geometry-matched canonical energies."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import math

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    reference = json.loads((ROOT / "examples/w2-avdz-reference.json").read_text())
    geometry_hash = hashlib.sha256((run / "geometry.xyz").read_bytes()).hexdigest()
    if geometry_hash != reference["geometry_sha256"]:
        raise ValueError("not the reference W2 CCSD(T)/aVDZ input geometry")
    config = json.loads((run / "run.json").read_text())
    if config["basis"] != "avdz" or config["symmetry"] != "spin" or config["stage"] != "full":
        raise ValueError("validation requires full W2/aVDZ with spin symmetry")
    marker = (run / "COMPLETE").read_text()
    log_name = next(line.removeprefix("log=") for line in marker.splitlines() if line.startswith("log="))
    if Path(log_name).name != log_name:
        raise ValueError("invalid completion log name")
    text = (run / log_name).read_text()
    results = {}
    for label, expected in reference["energies"].items():
        match = re.search(r"^" + re.escape(label) + r"\s+([-+0-9.eE]+)\s+Eh\s*$", text, re.MULTILINE)
        if match is None:
            raise ValueError(f"missing energy: {label}")
        actual = float(match[1])
        delta = actual - expected
        passed = math.isfinite(actual) and abs(delta) <= reference["tolerance"]
        results[label] = dict(calculated=actual, reference=expected, difference=delta, passed=passed)
        print(f"{label}: calculated={actual:.12f} reference={expected:.12f} delta={delta:+.3e} Eh passed={passed}")
    if not all(item["passed"] for item in results.values()):
        raise ValueError("W2 canonical component gate failed at 1e-7 Eh; do not publish as validated")
    if "closed-shell exact spin-flip amplitude gate passed" not in text:
        raise ValueError("closed-shell spin-symmetry admission marker missing")
    report = dict(geometry_sha256=geometry_hash, tolerance_hartree=reference["tolerance"],
                  energies=results, log=log_name,
                  scope="W2 components; nM closure checked by launcher; not an independent nM reference")
    with (run / "W2_VALIDATED.json").open("x") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
    print(f"W2 VALIDATION PASSED: {run / 'W2_VALIDATED.json'}")


if __name__ == "__main__":
    main()
