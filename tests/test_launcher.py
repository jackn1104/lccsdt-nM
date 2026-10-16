"""Public-interface contract tests; no molecular calculation is launched."""
import importlib.machinery
import importlib.util
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("lccsdt_cli", str(ROOT / "lccsdt"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)


class LauncherContract(unittest.TestCase):
    def config(self):
        return dict(storage="memory", symmetry="spin", waters=1, launcher="serial",
                    nodes=1, ranks_per_node=1, cpus_per_task=2, threads=2,
                    stage="full", memory_mib=2048, scratch_mib=4096, identity="fixture",
                    engine="/example/build/lccsdt_engine", basis="avdz")

    def test_geometry(self):
        self.assertEqual(cli.check_geometry(ROOT / "examples/water.xyz"), 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.xyz"
            for content in ("2\nbad\nO 0 0 0\nH 0 0 1\n",
                            "3\nbad\nO nan 0 0\nH 0 0 1\nH 0 1 0\n",
                            "3\nbad\nH 0 0 0\nO 0 0 1\nH 0 1 0\n"):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    cli.check_geometry(path)

    def test_conflicting_symmetry(self):
        cfg = self.config()
        cfg.update(symmetry="point-group", storage="disk", waters=4)
        with self.assertRaises(ValueError):
            cli.check_options(cfg)

    def test_full_model_and_restart(self):
        cfg = self.config()
        env = cli.environment(cfg, Path("/run"), True)
        self.assertEqual(env["CCSD_SELECTED_MAXIMUM_MONOMERS"], "full")
        self.assertEqual(env["CCSD_REQUIRE_CONVERGED_RESTART"], "1")
        self.assertEqual(env["CCSD_REFINE_SAVED_CCSD"], "0")
        self.assertEqual(env["CCSD_TRIPLES_TWO_SPIN_SECTORS"], "1")
        self.assertEqual(env["CCSD_CONVERGENCE_THRESHOLD"], "1e-8")
        self.assertEqual(env["CCSD_LAPLACE_TRIPLES_ERROR_TOLERANCE"], "1e-7")
        self.assertIn("required", cli.command(cfg, Path("/run"), True))

    def test_slurm_shape(self):
        cfg = self.config()
        cfg.update(launcher="slurm", nodes=2, ranks_per_node=4, threads=32, cpus_per_task=64)
        cmd = cli.command(cfg, Path("/run"), False)
        self.assertIn("--ntasks=8", cmd)
        self.assertIn("--cpus-per-task=64", cmd)
        self.assertIn("--nodes=2", cmd)

    def test_sector_checkpoint_platform(self):
        cfg = self.config()
        with patch.object(cli.sys, "platform", "darwin"):
            env = cli.environment(cfg, Path("/run"), False)
            self.assertNotIn("CCSD_TRIPLES_SECTOR_CHECKPOINT", env)
            self.assertEqual(env["CCSD_TRIPLES_TWO_SPIN_SECTORS"], "1")
        with patch.object(cli.sys, "platform", "linux"):
            env = cli.environment(cfg, Path("/run"), False)
            self.assertEqual(env["CCSD_TRIPLES_SECTOR_CHECKPOINT"], "/run/triples-sectors.chk")

    def test_ccsd_stage_mapping(self):
        cfg = self.config()
        cfg["stage"] = "ccsd"
        self.assertEqual(cli.environment(cfg, Path("/run"), False)["CCSD_EXECUTION_STAGE"], "ccsd-only")
        self.assertEqual(cli.environment(cfg, Path("/run"), True)["CCSD_EXECUTION_STAGE"], "full")

    def test_output_contract(self):
        text = ("RHF -1.0 Eh\nlocalized CCSD corr -0.1 Eh\nCCSD M1 -0.1 Eh\n"
                "localized (T) -0.01 Eh\n(T) M1 -0.01 Eh\nlocalized CCSD(T) -1.11 Eh\n")
        cli.check_results(text, 1, "full")
        with self.assertRaises(RuntimeError):
            cli.check_results(text.replace("(T) M1 -0.01", "(T) M1 -0.02"), 1, "full")
        with self.assertRaises(RuntimeError):
            cli.check_results(text.replace("localized (T) -0.01", "localized (T) nan"), 1, "full")

    def test_concise_ccsd(self):
        output = cli.ConciseOutput()
        line = ("progress localized CCSD iteration 12; E=-0.45 Eh; dE=1e-9 Eh; "
                "R1=1e-6; R2=2e-6; R1_measure=1e-9; R2_measure=2e-9; "
                "backend=replicated exact; wall=20 s")
        self.assertEqual(output.line(line), "CCSD iteration 12: E_corr=-0.45 Eh; "
                         "dE=1e-9 Eh; R1(RMS)=1e-9; R2(RMS)=2e-9")
        self.assertIn("CCSD convergence check 12", output.line(line.replace(
            "iteration 12", "convergence check after iteration 12")))
        self.assertIn("R1(L2)=1e-6", output.line(line.replace(
            "R1_measure=1e-9; R2_measure=2e-9; ", "")))
        self.assertEqual(output.line("progress localized CCSD converged"), "CCSD converged")
        self.assertIn("reused", output.line("progress converged localized CCSD checkpoint "
                                           "restored; CCSD iterations this run=0"))

    def test_concise_laplace(self):
        output = cli.ConciseOutput()
        self.assertIsNone(output.line("progress Laplace triples point 3/7; sector=1 started"))
        self.assertEqual(output.line("progress Laplace triples sector 1 complete"),
                         "Laplace (T) sector 1: point 3/7 complete")
        self.assertEqual(output.line("progress triples sector 0 point 7 restored; propagation skipped"),
                         "Laplace (T) sector 0: point 7 restored")
        self.assertEqual(output.line("progress orbital-invariant Laplace triples point 3/7 complete; "
                                    "node=0.5, weight=0.1, seconds=2, M1=-0.01, M2=-0.02, "
                                    "M3=0, M4=0, M5=0, M6=0"),
                         "Laplace (T) point 3/7: energy contribution=-3.000000000000e-02 Eh")

    def test_concise_summary_and_errors(self):
        output = cli.ConciseOutput()
        for line in ("RHF -152.0 Eh", "CCSD M4 -0.001 Eh", "(T) M5 -0.002 Eh",
                     "(T) M6 -0.003 Eh", "wall time 12.5 s", "CCSD convergence threshold 1e-8",
                     "fatal: checkpoint mismatch", "srun: error: task failed",
                     "WARNING: limited memory", "AddressSanitizer:DEADLYSIGNAL"):
            self.assertIsNotNone(output.line(line), line)
        for line in ("CCSD kernel total this run 12 s", "progress RHF calculation started",
                     "CCSD W_abef invariant cache 100 bytes", "", "unmeasured timing nan"):
            self.assertIsNone(output.line(line), line)

    def test_stream_modes_and_failure(self):
        raw = "progress RHF calculation started\nRHF -1.0 Eh\nwarning: example warning\n"
        with tempfile.TemporaryDirectory() as tmp:
            concise_files = []
            for verbose in (False, True):
                log = Path(tmp) / f"raw-{verbose}.log"
                out = Path(tmp) / f"clean-{verbose}.out"
                terminal = io.StringIO()
                command = [sys.executable, "-c", f"print({raw!r}, end='')"]
                with contextlib.redirect_stdout(terminal):
                    self.assertEqual(cli.stream_output(command, dict(os.environ), log, out, verbose), 0)
                self.assertEqual(log.read_text(), raw)
                concise_files.append(out.read_text())
                self.assertEqual(terminal.getvalue(), raw if verbose else out.read_text())
                with self.assertRaises(FileExistsError):
                    cli.stream_output(command, dict(os.environ), log, out, verbose)
            self.assertEqual(*concise_files)
            log, out = Path(tmp) / "failure.log", Path(tmp) / "failure.out"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.stream_output([sys.executable, "-c", "raise SystemExit(3)"],
                                                  dict(os.environ), log, out, False), 3)
            self.assertIn("exited with status 3", out.read_text())


if __name__ == "__main__":
    unittest.main()
